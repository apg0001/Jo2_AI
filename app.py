# from flask import Flask, request, jsonify, session, send_file, make_response
# from flask_cors import CORS  # CORS 라이브러리 임포트
# from flask_session import Session
# from chatbot_service import get_chat_response, get_score_from_intent, ask_phq9_question, phq9_questions, evaluate_overall_depression, upload_and_predict, summarize_depression_analysis, analyze_overall_chat
# from models import ChatRequest, ChatResponse
# import os
# import datetime
# import requests
# import pyttsx3
# import ffmpeg
# import soundfile as sf

# app = Flask(__name__)
# CORS(app, supports_credentials=True)
# # CORS(app, origins=["http://localhost:3000", "https://your-frontend-domain.com"])

# app.config['SECRET_KEY'] = 'supersecretkey'
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(
#     minutes=60)  # 세션 타임아웃 60분 설정
# app.config['SESSION_COOKIE_SAMESITE'] = 'None'
# # app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS에서만 동작, 로컬 개발 시 False로 설정 가능
# # app.config['SESSION_COOKIE_SECURE'] = False
# app.config['SESSION_COOKIE_HTTPONLY'] = False # HttpOnly 속성을 False로 설정
# Session(app)

# TARGET_SERVER_URL = 'https://api.joyfully.o-r.kr/api/v1/weather/score'  # 데이터를 전송할 대상 서버의 URL
# WAVE_OUTPUT_FILENAME = "./audio/record.wav"  # 클라이언트로부터 받은 오디오 파일 저장 경로
# TTS_OUTPUT_FILENAME = "./audio/response.mp3"  # TTS로 생성된 음성 파일 저장 경로


# @app.before_request
# def make_session_permanent():
#     session.permanent = True

# def is_wav_format(file_path):
#     try:
#         with sf.SoundFile(file_path) as file:
#             return file.format == 'WAV'
#     except Exception as e:
#         print(f"파일 형식을 확인하는 중 오류 발생: {e}")
#         return False

# def convert_to_wav(input_file, output_file):
#     try:
#         ffmpeg.input(input_file).output(output_file, acodec='pcm_s16le', ar='44100').run()
#         print(f"파일이 {output_file}로 변환되었습니다.")
#     except Exception as e:
#         print(f"파일 변환 중 오류 발생: {e}")

# @app.route('/api/chatbot/start', methods=['POST'])
# def start_chat():
#     data = request.json
#     if 'user_id' not in data:
#         return jsonify({'error': 'User ID is required'}), 400

#     session.clear()  # 이전 세션을 지우고 새 세션을 시작
#     session['user_id'] = data['user_id']  # 사용자의 ID 저장
#     session['phq9_index'] = 0
#     session['phq9_scores'] = []
#     session['completed_phq9'] = False
#     session['chat_history'] = []  # 채팅 내역 초기화
#     return jsonify({'message': '새로운 채팅이 시작되었습니다.\n설문을 시작합니다.\n설문에 답해주세요.', 'user_id': session['user_id']})
#     # response = make_response("새로운 세션이 시작되었습니다. user_id: " + str(session['user_id']))
#     # response.set_cookie('user_id', str(session.get('user_id')), httponly=False)
#     # for header, value in response.headers.items():
#     #     print(f'{header}: {value}')
#     # print(response.get_data(as_text=True))
#     # return response


# def process_chat_message(message):
#     """메시지(텍스트 또는 음성 변환 텍스트)를 처리하는 함수"""
    
#     if 'user_id' not in session:
#         print("세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.")
#         return {'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}, 403

#     print("processing chat,,,: ", session['user_id'])
#     if 'phq9_index' not in session:
#         print("세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.")
#         return {'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}, 403

#     session['chat_history'].append(
#         {'role': 'user', 'content': message})  # 채팅 내역에 추가

#     if not session['completed_phq9']:
#         phq9_index = session['phq9_index']
#         phq9_scores = session['phq9_scores']

#         score = get_score_from_intent(message)
#         phq9_scores.append(score)
#         session['phq9_index'] = phq9_index + 1
#         session['phq9_scores'] = phq9_scores

#         if phq9_index < len(phq9_questions):
#             next_question = ask_phq9_question(phq9_index)
#             session['chat_history'].append(
#                 {'role': 'assistant', 'content': next_question})  # 채팅 내역에 추가
#             return {'response': next_question, 'current_score': score, 'total_score': sum(phq9_scores)}, 200
#         else:
#             total_score = sum(phq9_scores)
#             session['completed_phq9'] = True
#             result = {
#                 'response': "PHQ-9 질문이 완료되었습니다. 이제 일상적인 대화를 나눌 수 있습니다.",
#                 'total_score': total_score,
#                 'assessment': assess_depression(total_score)
#             }
#             session['phq9_score'] = total_score
#             session['chat_history'].append(
#                 {'role': 'assistant', 'content': result['response']})  # 채팅 내역에 추가
#             return result, 200
#     else:
#         chat_request = ChatRequest(message=message)  # ChatRequest 객체 생성
#         chat_response = get_chat_response(chat_request)
#         # chat_response = get_chat_response(message)
#         session['chat_history'].append(
#             {'role': 'assistant', 'content': chat_response.response})  # 챗봇 응답 추가
#         return {'response': chat_response.to_dict()}, 200


# @app.route('/api/chatbot/chat', methods=['POST'])
# def chat():
#     print(request.headers)
#     data = request.json
#     print(data)
#     if 'message' not in data:
#         return jsonify({'error': 'Message field is required'}), 400

#     result, status_code = process_chat_message(data['message'])
#     return jsonify(result), status_code


# @app.route('/api/chatbot/voice', methods=['POST'])
# def voice_chat():
#     """클라이언트로부터 음성 파일을 받아 처리하고 OpenAI API로 전송 후, 응답을 음성 파일로 반환"""
#     if 'audio' not in request.files:
#         return jsonify({'error': 'Audio file is required'}), 400
    
#     # 테스트
#     # print(request.headers)
#     # data = request.files
#     # print(data)

#     # # 음성 파일 저장
#     # audio_file = request.files['file']
#     # audio_file.save(WAVE_OUTPUT_FILENAME)

#     audio_file = request.files["audio"]
#     # audio_path = os.path.join("audio", "input.wav")
#     audio_file.save(WAVE_OUTPUT_FILENAME)
    
#     # 파일이 이미 WAV 포맷인지 확인
#     # if is_wav_format(WAVE_OUTPUT_FILENAME):
#     #     print("파일이 이미 WAV 포맷입니다. 변환하지 않습니다.")
#     #     # output_file = input_file  # 변환하지 않고 기존 파일을 그대로 사용
#     # else:
#     #     # 파일을 WAV로 변환
#     #     print("파일이 WAV 포맷이 아닙니다. 파일을 변환합니다.")
#     #     convert_to_wav(WAVE_OUTPUT_FILENAME, WAVE_OUTPUT_FILENAME)

#     # 변환된 (혹은 기존의) 파일을 사용하는 코드 작성
#     try:
#         data, samplerate = sf.read(WAVE_OUTPUT_FILENAME)
#         print("파일을 성공적으로 읽었습니다!")
#         # 여기서 data와 samplerate를 사용하여 추가 작업을 수행할 수 있습니다.
#     except Exception as e:
#         print(f"파일을 읽는 중 오류 발생: {e}")

#     # 음성 인식 및 텍스트 추론 수행
#     corrected_text = upload_and_predict(WAVE_OUTPUT_FILENAME)
#     # corrected_text = upload_and_predict(WAVE_OUTPUT_FILENAME)

#     result, status_code = process_chat_message(corrected_text)

#     # if status_code == 200:
#     #     # TTS 변환
#     #     tts_engine = pyttsx3.init()
#     #     tts_engine.save_to_file(result['response'], TTS_OUTPUT_FILENAME)
#     #     tts_engine.runAndWait()

#     #     # 음성 파일 반환
#     #     return send_file(TTS_OUTPUT_FILENAME, mimetype='audio/mp3')

#     # return jsonify(result), status_code

#     response = {
#         "recognizedText": corrected_text,
#         "response": result
#     }
    
#     print(response)

#     return jsonify(response), status_code


# @app.route('/api/chatbot/end', methods=['POST'])
# def end_chat():
#     chat_history = session.get('chat_history', [])
#     overall_assessment = evaluate_overall_depression(chat_history)
#     analyze_chat = analyze_overall_chat(chat_history)

#     data_to_send = {
#         'userId': session.get('user_id'),  # 세션에 저장된 사용자 ID
#         # 'session_id': session.sid,
#         'overallScore': overall_assessment,
#         'phq9Score': session.get('phq9_score'),
#         # 'chat_history': chat_history
#         'summary': analyze_chat
#     }

#     print(chat_history)

#     print(data_to_send)

#     # 다른 서버로 데이터 전송
#     try:
#         response = requests.post(TARGET_SERVER_URL, json=data_to_send)
#         server_response = response.json()  # JSON 응답 파싱 시도
#     except requests.exceptions.JSONDecodeError:
#         server_response = "백서버 연결 실패요 ㅅㄱ"  # JSON 파싱 실패 시, 텍스트 응답 반환
#         print("response error")

#     session.clear()  # 세션 데이터를 삭제하여 세션을 종료합니다.
#     return jsonify({
#         'response': '채팅이 종료되었습니다. 세션이 종료되었습니다.',
#         'score': overall_assessment,
#         'summary': analyze_chat,
#         'server_response': server_response
#     })

# @app.route('/api/chatbot/analyze', methods=['POST'])
# def analyze_depression_trend():
#     """우울증 점수와 분석을 받아 요약된 분석을 생성"""
#     data = request.json

#     if 'weatherList' not in data or not isinstance(data['weatherList'], list):
#         return jsonify({'error': 'weatherList must be a list of weather data'}), 400

#     weather_list = data['weatherList']

#     # 리스트의 각 항목을 특정 형식의 문자열로 변환
#     parsed_string = ', '.join(
#         f"날짜 : {item['createdAt']}/요일 : {item['dayofweek']}/채팅 내용 분석 : {item['result']}/채팅 점수 : {item['score']}/phq설문 점수 : { item['phq9Score']}"
#         for item in weather_list
#     )

#     # 이 문자열을 summarize_depression_analysis 함수로 전달
#     summary = summarize_depression_analysis(parsed_string)

#     return jsonify({'summary': summary})


# def assess_depression(total_score: int) -> str:
#     if total_score < 5:
#         return "우울증이 없는 상태입니다."
#     elif total_score < 10:
#         return "경미한 우울증이 의심됩니다."
#     elif total_score < 15:
#         return "중간 정도의 우울증이 의심됩니다."
#     elif total_score < 20:
#         return "치료를 요하는 정도의 우울증이 의심됩니다."
#     else:
#         return "심한 우울증이 의심됩니다."


# if __name__ == '__main__':
#     # app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
#     app.run(host='0.0.0.0', port=5000)

# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS  # CORS 라이브러리 임포트
# from chatbot_service import get_chat_response, get_score_from_intent, ask_phq9_question, phq9_questions, evaluate_overall_depression, upload_and_predict, summarize_depression_analysis, analyze_overall_chat
# from models import ChatRequest, ChatResponse
# import os
# import requests
# import jwt

# # 환경변수에서 JWT 시크릿 키를 가져옴
# JWT_SECRET = os.getenv('JWT_SECRET')
# JWT_ALGORITHM = 'HMAC'

# app = Flask(__name__)
# CORS(app, supports_credentials=True)

# # 세션을 대체할 딕셔너리로 상태 관리 (메모리 내 저장)
# user_sessions = {}

# TARGET_SERVER_URL = 'https://api.joyfully.o-r.kr/api/v1/weather/score'  # 데이터를 전송할 대상 서버의 URL
# WAVE_OUTPUT_FILENAME = "./audio/record.wav"  # 클라이언트로부터 받은 오디오 파일 저장 경로
# TTS_OUTPUT_FILENAME = "./audio/response.mp3"  # TTS로 생성된 음성 파일 저장 경로

# def decode_jwt_token(token):
#     """JWT 토큰 디코딩 및 검증"""
#     try:
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         return payload['MEMBER_ID']
#     except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
#         return None

# def get_user_session(user_id):
#     """사용자 세션을 가져오거나 새로 생성"""
#     if user_id not in user_sessions:
#         user_sessions[user_id] = {
#             'phq9_index': 0,
#             'phq9_scores': [],
#             'completed_phq9': False,
#             'chat_history': [],
#             'phq9_score': 0
#         }
#     return user_sessions[user_id]

# @app.route('/api/chatbot/start', methods=['POST'])
# def start_chat():
#     print(request.headers)
#     data = request.json
#     if 'Authorization' not in request.headers:
#         return jsonify({'error': 'Token is required'}), 400

#     user_id = decode_jwt_token(request.headers['Authorization'])
#     if not user_id:
#         return jsonify({'error': 'Invalid or expired token'}), 401

#     # 사용자 세션 초기화
#     user_sessions[user_id] = {
#         'phq9_index': 0,
#         'phq9_scores': [],
#         'completed_phq9': False,
#         'chat_history': [],
#         'phq9_score': 0
#     }

#     return jsonify({'message': '새로운 채팅이 시작되었습니다.\n설문을 시작합니다.\n설문에 답해주세요.', 'user_id': user_id})

# def process_chat_message(token, message):
#     """메시지(텍스트 또는 음성 변환 텍스트)를 처리하는 함수"""
#     user_id = decode_jwt_token(token)
#     if not user_id:
#         return {'error': '토큰이 만료되었거나 유효하지 않습니다. 새로운 토큰을 시작하세요.'}, 401

#     # 사용자 세션 가져오기
#     session = get_user_session(user_id)

#     session['chat_history'].append({'role': 'user', 'content': message})  # 채팅 내역에 추가

#     if not session['completed_phq9']:
#         phq9_index = session['phq9_index']
#         phq9_scores = session['phq9_scores']

#         score = get_score_from_intent(message)
#         phq9_scores.append(score)
#         session['phq9_index'] = phq9_index + 1
#         session['phq9_scores'] = phq9_scores

#         if phq9_index < len(phq9_questions):
#             next_question = ask_phq9_question(phq9_index)
#             session['chat_history'].append({'role': 'assistant', 'content': next_question})  # 채팅 내역에 추가
#             return {'response': next_question, 'current_score': score, 'total_score': sum(phq9_scores)}, 200
#         else:
#             total_score = sum(phq9_scores)
#             session['completed_phq9'] = True
#             session['phq9_score'] = total_score
#             result = {
#                 'response': "PHQ-9 질문이 완료되었습니다. 이제 일상적인 대화를 나눌 수 있습니다.",
#                 'total_score': total_score,
#                 'assessment': assess_depression(total_score)
#             }
#             session['chat_history'].append({'role': 'assistant', 'content': result['response']})  # 채팅 내역에 추가
#             return result, 200
#     else:
#         chat_request = ChatRequest(message=message)
#         chat_response = get_chat_response(chat_request)
#         session['chat_history'].append({'role': 'assistant', 'content': chat_response.response})  # 챗봇 응답 추가
#         return {'response': chat_response.to_dict()}, 200

# @app.route('/api/chatbot/chat', methods=['POST'])
# def chat():
#     data = request.json
#     if 'message' not in data:
#         return jsonify({'error': 'Messagefields are required'}), 400
#     if 'Authorization' not in request.headers:
#         return jsonify({'error': 'Token fields are required'}), 400
        

#     result, status_code = process_chat_message(data['token'], data['message'])
#     return jsonify(result), status_code

# @app.route('/api/chatbot/voice', methods=['POST'])
# def voice_chat():
#     """클라이언트로부터 음성 파일을 받아 처리하고 OpenAI API로 전송 후, 응답을 음성 파일로 반환"""
#     if 'audio' not in request.files or 'token' not in request.form:
#         return jsonify({'error': 'Audio file and token are required'}), 400
    
#     token = request.headers['Authorization']
#     user_id = decode_jwt_token(token)
#     if not user_id:
#         return jsonify({'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}), 403

#     audio_file = request.files["audio"]
#     audio_file.save(WAVE_OUTPUT_FILENAME)

#     corrected_text = upload_and_predict(WAVE_OUTPUT_FILENAME)
#     result, status_code = process_chat_message(token, corrected_text)

#     response = {
#         "recognizedText": corrected_text,
#         "response": result
#     }
    
#     return jsonify(response), status_code

# @app.route('/api/chatbot/end', methods=['POST'])
# def end_chat():
#     data = request.json
#     if 'Authorization' not in request.headers:
#         return jsonify({'error': 'Token is required'}), 401
    
#     user_id = decode_jwt_token(request.headers['Authorization'])
#     if not user_id:
#         return jsonify({'error': '세션이 만료되었거나 유효하지 않습니다. 새로운 세션을 시작하세요.'}), 401

#     session = user_sessions.get(user_id, None)
#     if not session:
#         return jsonify({'error': '세션을 찾을 수 없습니다. 새로운 세션을 시작하세요.'}), 401

#     chat_history = session['chat_history']
#     overall_assessment = evaluate_overall_depression(chat_history)
#     analyze_chat = analyze_overall_chat(chat_history)

#     data_to_send = {
#         'userId': user_id,
#         'overallScore': overall_assessment,
#         'phq9Score': session.get('phq9_score'),
#         'summary': analyze_chat
#     }

#     try:
#         response = requests.post(TARGET_SERVER_URL, json=data_to_send)
#         server_response = response.json()
#     except requests.exceptions.JSONDecodeError:
#         server_response = "백서버 연결 실패요 ㅅㄱ"
#         print("response error")

#     # 세션 종료 시 사용자 세션 데이터 삭제
#     del user_sessions[user_id]

#     return jsonify({
#         'response': '채팅이 종료되었습니다.',
#         'score': overall_assessment,
#         'summary': analyze_chat,
#         'server_response': server_response
#     })

# @app.route('/api/chatbot/analyze', methods=['POST'])
# def analyze_depression_trend():
#     """우울증 점수와 분석을 받아 요약된 분석을 생성"""
#     data = request.json

#     if 'weatherList' not in data or not isinstance(data['weatherList'], list):
#         return jsonify({'error': 'weatherList must be a list of weather data'}), 400

#     weather_list = data['weatherList']

#     parsed_string = ', '.join(
#         f"날짜 : {item['createdAt']}/요일 : {item['dayofweek']}/채팅 내용 분석 : {item['result']}/채팅 점수 : {item['score']}/phq설문 점수 : { item['phq9Score']}"
#         for item in weather_list
#     )

#     summary = summarize_depression_analysis(parsed_string)

#     return jsonify({'summary': summary})

# def assess_depression(total_score: int) -> str:
#     if total_score < 5:
#         return "우울증이 없는 상태입니다."
#     elif total_score < 10:
#         return "경미한 우울증이 의심됩니다."
#     elif total_score < 15:
#         return "중간 정도의 우울증이 의심됩니다."
#     elif total_score < 20:
#         return "치료를 요하는 정도의 우울증이 의심됩니다."
#     else:
#         return "심한 우울증이 의심됩니다."

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

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

# 환경변수에서 JWT 시크릿 키를 가져옴
JWT_SECRET = os.getenv('JWT_SECRET', 'your_jwt_secret')
JWT_ALGORITHM = 'HMAC'

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 세션을 대체할 딕셔너리로 상태 관리 (메모리 내 저장)
user_sessions = {}

TARGET_SERVER_URL = 'https://api.joyfully.o-r.kr/api/v1/weather/score'  # 데이터를 전송할 대상 서버의 URL
WAVE_OUTPUT_FILENAME = "./audio/record.wav"  # 클라이언트로부터 받은 오디오 파일 저장 경로
TTS_OUTPUT_FILENAME = "./audio/response.mp3"  # TTS로 생성된 음성 파일 저장 경로

def decode_jwt_token(token):
    """JWT 토큰 디코딩 및 검증"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload['userId']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
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
        chat_response = get_chat_response(chat_request)
        session['chat_history'].append({'role': 'assistant', 'content': chat_response.response})  # 챗봇 응답 추가
        return {'response': chat_response.to_dict()}, 200

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

    try:
        response = requests.post(TARGET_SERVER_URL, json=data_to_send)
        server_response = response.json()
    except requests.exceptions.JSONDecodeError:
        server_response = "백서버 연결 실패요 ㅅㄱ"
        print("response error")

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

    if 'weatherList' not in data or not isinstance(data['weatherList'], list):
        return jsonify({'error': 'weatherList must be a list of weather data'}), 400

    weather_list = data['weatherList']

    parsed_string = ', '.join(
        f"날짜 : {item['createdAt']}/요일 : {item['dayofweek']}/채팅 내용 분석 : {item['result']}/채팅 점수 : {item['score']}/phq설문 점수 : { item['phq9Score']}"
        for item in weather_list
    )

    summary = summarize_depression_analysis(parsed_string)

    return jsonify({'summary': summary})

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