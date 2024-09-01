from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # CORS 라이브러리 임포트
from chatbot_service import get_chat_response, get_score_from_intent, ask_phq9_question, phq9_questions, evaluate_overall_depression, upload_and_predict, summarize_depression_analysis, analyze_overall_chat
from models import ChatRequest, ChatResponse
import os
import datetime
import requests
import pyttsx3
import ffmpeg
import soundfile as sf
import jwt
import re

# 환경변수에서 JWT 시크릿 키를 가져옴
JWT_SECRET = os.getenv('JWT_SECRET', "")
JWT_ALGORITHM = 'HS512'

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 세션을 대체할 딕셔너리로 상태 관리 (메모리 내 저장)
user_sessions = {}

TARGET_SERVER_URL = 'https://api.joyfully.o-r.kr/api/v1/weather/score'  # 데이터를 전송할 대상 서버의 URL
WAVE_OUTPUT_FILENAME = "./audio/record.wav"  # 클라이언트로부터 받은 오디오 파일 저장 경로
TTS_OUTPUT_FILENAME = "./audio/response.mp3"  # TTS로 생성된 음성 파일 저장 경로

def remove_tags(html_content):
    # HTML, HEAD, BODY 태그를 제거하는 정규 표현식
    html_content = re.sub(r'</?(html|head|body)>', '', html_content, flags=re.IGNORECASE)
    # 개행 문자 제거
    html_content = html_content.replace('\n', '')
    html_content = html_content.replace('<!DOCTYPE html>', '')
    return html_content

def decode_jwt_token(token):
    """JWT 토큰 디코딩 및 검증"""
    try:
        # print(token)
        payload = jwt.decode(jwt=token, jey=JWT_SECRET, algorithms=JWT_ALGORITHM, options={"verify_signature": False})
        # print(payload)
        return payload['memberId']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        print(e)
        return None

def get_user_session(user_id):
    """사용자 세션을 가져오거나 새로 생성"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'phq9_index': 0,
            'phq9_scores': [],
            'completed_phq9': False,
            'chat_history': [],
            'phq9_score': 0
        }
    return user_sessions[user_id]

def get_token_from_header():
    """Authorization 헤더에서 JWT 토큰 추출"""
    auth_header = request.headers.get('Authorization', None)
    if auth_header:
        return auth_header.split(" ")[1]  # 'Bearer <token>'에서 token 부분만 추출
    return None

@app.route('/api/chatbot/start', methods=['POST'])
def start_chat():
    # print(request.headers)
    token = get_token_from_header()
    if not token:
        return jsonify({'error': 'Authorization token is required'}), 400

    user_id = decode_jwt_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid or expired token'}), 401

    # 사용자 세션 초기화
    user_sessions[user_id] = {
        'phq9_index': 0,
        'phq9_scores': [],
        'completed_phq9': False,
        'chat_history': [],
        'phq9_score': 0
    }

    return jsonify({'message': '새로운 채팅이 시작되었습니다.\n설문을 시작합니다.\n설문에 답해주세요.', 'user_id': user_id})

def process_chat_message(token, message):
    """메시지(텍스트 또는 음성 변환 텍스트)를 처리하는 함수"""
    user_id = decode_jwt_token(token)
    if not user_id:
        return {'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}, 403

    # 사용자 세션 가져오기
    session = get_user_session(user_id)

    session['chat_history'].append({'role': 'user', 'content': message})  # 채팅 내역에 추가

    if not session['completed_phq9']:
        phq9_index = session['phq9_index']
        phq9_scores = session['phq9_scores']

        score = get_score_from_intent(message)
        phq9_scores.append(score)
        session['phq9_index'] = phq9_index + 1
        session['phq9_scores'] = phq9_scores

        if phq9_index < len(phq9_questions):
            next_question = ask_phq9_question(phq9_index)
            session['chat_history'].append({'role': 'assistant', 'content': next_question})  # 채팅 내역에 추가
            return {'response': next_question, 'current_score': score, 'total_score': sum(phq9_scores)}, 200
        else:
            total_score = sum(phq9_scores)
            session['completed_phq9'] = True
            session['phq9_score'] = total_score
            result = {
                'response': "PHQ-9 질문이 완료되었습니다. 이제 일상적인 대화를 나눌 수 있습니다.",
                'total_score': total_score,
                'assessment': assess_depression(total_score)
            }
            session['chat_history'].append({'role': 'assistant', 'content': result['response']})  # 채팅 내역에 추가
            return result, 200
    else:
        chat_request = ChatRequest(message=message)
        chat_response = get_chat_response(session['chat_history'], chat_request)
        session['chat_history'].append({'role': 'assistant', 'content': chat_response.response})  # 챗봇 응답 추가
        # return {'response': chat_response.to_dict()}, 200
        return {'response': chat_response.response}, 200

@app.route('/api/chatbot/chat', methods=['POST'])
def chat():
    token = get_token_from_header()
    if not token:
        return jsonify({'error': 'Authorization token is required'}), 400

    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'Message field is required'}), 400

    result, status_code = process_chat_message(token, data['message'])
    return jsonify(result), status_code

@app.route('/api/chatbot/voice', methods=['POST'])
def voice_chat():
    """클라이언트로부터 음성 파일을 받아 처리하고 OpenAI API로 전송 후, 응답을 음성 파일로 반환"""
    token = get_token_from_header()
    if not token:
        return jsonify({'error': 'Authorization token is required'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'Audio file is required'}), 400
    
    user_id = decode_jwt_token(token)
    if not user_id:
        return jsonify({'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}), 403

    audio_file = request.files["audio"]
    audio_file.save(WAVE_OUTPUT_FILENAME)

    corrected_text = upload_and_predict(WAVE_OUTPUT_FILENAME)
    result, status_code = process_chat_message(token, corrected_text)

    response = {
        "recognizedText": corrected_text,
        "response": result
    }
    
    return jsonify(response), status_code

@app.route('/api/chatbot/end', methods=['POST'])
def end_chat():
    token = get_token_from_header()
    if not token:
        return jsonify({'error': 'Authorization token is required'}), 400
    
    user_id = decode_jwt_token(token)
    if not user_id:
        return jsonify({'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}), 401

    session = user_sessions.get(user_id, None)
    if not session:
        return jsonify({'error': '세션을 찾을 수 없습니다. 새로운 세션을 시작하세요.'}), 403

    chat_history = session['chat_history']
    overall_assessment = evaluate_overall_depression(chat_history)
    analyze_chat = analyze_overall_chat(chat_history)

    data_to_send = {
        'userId': user_id,
        'overallScore': overall_assessment,
        'phq9Score': session.get('phq9_score'),
        'summary': analyze_chat
    }
    if session['completed_phq9']:
        try:
            response = requests.post(TARGET_SERVER_URL, json=data_to_send)
            server_response = response.json()
            print("back server response 200")
        except requests.exceptions.JSONDecodeError:
            server_response = "백서버 연결 실패요 ㅅㄱ"
            print("back server response error")
    else:
        server_response = "선생님 얘 설문 다 안하고 도망갔어요."
        print(server_response)

    # 세션 종료 시 사용자 세션 데이터 삭제
    del user_sessions[user_id]

    return jsonify({
        'response': '채팅이 종료되었습니다.',
        'score': overall_assessment,
        'summary': analyze_chat,
        'server_response': server_response
    })

@app.route('/api/chatbot/analyze', methods=['POST'])
def analyze_depression_trend():
    """우울증 점수와 분석을 받아 요약된 분석을 생성"""
    data = request.json
    print(data)

    if 'weather_list' not in data or not isinstance(data['weather_list'], list):
        return jsonify({'error': 'weather_list must be a list of weather data'}), 400

    weather_list = data['weather_list']

    parsed_string = ', '.join(
        f"날짜 : {item['date']}/채팅 내용 분석 : {item['result']}/채팅 점수 : {item['score']}/phq설문 점수 : { item['phq_score']}"
        for item in weather_list
    )

    summary = summarize_depression_analysis(parsed_string)
    summary_remove_tag = remove_tags(summary)

    return jsonify({'summary': summary_remove_tag})

def assess_depression(total_score: int) -> str:
    if total_score < 5:
        return "우울증이 없는 상태입니다."
    elif total_score < 10:
        return "경미한 우울증이 의심됩니다."
    elif total_score < 15:
        return "중간 정도의 우울증이 의심됩니다."
    elif total_score < 20:        
        return "치료를 요하는 정도의 우울증이 의심됩니다."
    else:
        return "심한 우울증이 의심됩니다."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)