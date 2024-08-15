import librosa
import torch
import time
import numpy as np
from torch import Tensor
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

# #init language model
# # from transformers import TFBertForMaskedLM
# from transformers import AutoModelForMaskedLM
# from transformers import AutoTokenizer
# from transformers import FillMaskPipeline

# # model = TFBertForMaskedLM.from_pretrained("klue/bert-base", from_pt=True)
# model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
# tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
# pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

# def lm(script):
#     sep = script.split()

#     cnt = len(sep)
#     print("\nwords: ", cnt, "\n")

#     for i in range(cnt):
#         #print(i, " : ")
#         temp = sep[i]
#         sep[i] = "[MASK]"
#         sc = " ".join(sep)

#         flag = False

#         lst=pip(sc)

#         for j in range(5):
#             if temp == lst[j]["token_str"]:
#                 flag = True
        
#         if flag:
#             sep[i] = temp
#         else:
#             sep[i] = lst[0]["token_str"]

#     return " ".join(sep)

start = time.time()

audio = OmegaConf.create(
    {
        "name": "melspectogram",
        "sample_rate": 16000,
        "frame_length": 20.0,
        "frame_shift": 10.0,
        "del_silence": False,
        "num_mels": 80,
        "apply_spec_augment": True,
        "apply_noise_augment": False,
        "apply_time_stretch_augment": False,
        "apply_joining_augment": False,
    }
)

infer = OmegaConf.create(
    {
        "use_cuda": True,
        "dataset_path": "???",
        # "checkpoint_path": "C:/Users/lab1080/Desktop/rasa_sr/openspeech/models/joint_ctc_las.ckpt",
        # "checkpoint_path": "C:/Users/lab1080/Desktop/rasa_sr/openspeech/models/las.ckpt",
        "checkpoint_path": "C:/Users/lab1080/Desktop/rasa_sr/openspeech/models/multi_head_las.ckpt",
        "manifest_file_path": "???",
        "result_path": "???",
        "num_workers": "4",
        "batch_size": "32",
        "beam_size": "1",
    }
)

model = OmegaConf.create(
    {
        #"model_name": "joint_ctc_listen_attend_spell",
        # "model_name": "listen_attend_spell",
        "model_name": "listen_attend_spell_with_multi_head",
        "num_encoder_layers": 3,
        "num_decoder_layers": 2,
        # "hidden_state_dim": 196,
        "hidden_state_dim": 512,
        "encoder_dropout_p": 0.3,
        "encoder_bidirectional": True,
        "rnn_type": "lstm",
        # "joint_ctc_attention": True,
        "joint_ctc_attention": False,
        # "max_length": 96,
        "max_length": 128,
        # "num_attention_heads": 1,
        "num_attention_heads": 4,
        "decoder_dropout_p": 0.2,
        # "decoder_attn_mechanism": "loc",
        # "decoder_attn_mechanism": "dot",
        "decoder_attn_mechanism": "multi-head",
        "teacher_forcing_ratio:": 1.0,
        "optimizer": "adam",
    }
)

tokenizer = OmegaConf.create(
    {
        "sos_token": "<sos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "blank_token": "<blank>",
        "encoding": "utf-8",
        "unit": "kspon_character",
        "vocab_path": "C:/Users/lab1080/Desktop/rasa_sr/openspeech/aihub_labels.csv",
    }
)

trainer = OmegaConf.create(
    {"gradient_clip_val" : 5.0,
     "device": "gpu",
     "name": "gpu"}
)

criterion = OmegaConf.create(
    {"criterion_name": "cross_entropy"}
)

configs = OmegaConf.create(
    {"audio": audio, "infer": infer, "model": model, "tokenizer": tokenizer, "trainer": trainer, "criterion" : criterion}
)

tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

model = MODEL_REGISTRY[configs.model.model_name]
model = model.load_from_checkpoint(configs.infer.checkpoint_path, configs=configs, tokenizer=tokenizer)

use_cuda = configs.infer.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")
print("device: ", device, "\n\n")
model.to(device)

print("\ninit time: ", time.time()-start)

def transform_input(signal):
    print("\n...start transform input...")

    melspectrogram = librosa.feature.melspectrogram(
            #default n_fft: 2048, hop_length=512
            #512, 160이 가장 정확한 듯
            y=signal,
            sr=configs.audio.sample_rate,
            n_mels=configs.audio.num_mels,
            n_fft=512,
            hop_length=160,
        )
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    print("\n...end transform input...")
    return melspectrogram

def parse_audio(filepath: str) -> Tensor:
    print("\n...start parse audio...")

    signal, sr = librosa.load(filepath, sr=None)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
    feature = transform_input(signal)
    
    feature -= feature.mean()
    feature /= np.std(feature)

    feature = torch.FloatTensor(feature).transpose(0, 1)

    print("\n...end parse audio...")
    return feature


def inference(feature):
    print("\n...start inference...")
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(feature.unsqueeze(0).to(device), torch.as_tensor([feature.shape[0]], device=device))
    prediction = tokenizer.decode(outputs["predictions"].cpu().detach().numpy())

    print("\ninference time: ", time.time() - start)
    return prediction

def upload(filepath):

    # load file
    feature = parse_audio(filepath)
    
    feature = feature.to(device)

    prediction = inference(feature)
    
    print("\nprediction: ", prediction[0])
    
    #print("\nlanguage model: ", lm(prediction[0]))
    
    return {'prediction': prediction[0]}

if __name__ == "__main__":
    # rank_zero_info(OmegaConf.to_yaml(configs)) 
    
    start = time.time()
    
    # for i in range(23):
    #     dir = "C:/Users/lab1080/Desktop/rasa_sr/openspeech/test_file/audio/"
    #     upload(dir+str(i)+".wav")
        
    # upload("C:/Users/lab1080/Desktop/rasa_sr/openspeech/test_file/audio/voice.wav")
    upload("C:/Users/lab1080/Desktop/file.wav")
    
    print("\ntotal time: ", time.time() - start)