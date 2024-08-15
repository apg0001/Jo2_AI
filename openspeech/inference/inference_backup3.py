# import os
# import warnings

# import sys

# import hydra
# import torch
# from omegaconf import DictConfig

# from openspeech.data.audio.dataset import SpeechToTextDataset
# from openspeech.data.sampler import RandomSampler
# from openspeech.models import MODEL_REGISTRY
# from openspeech.tokenizers import TOKENIZER_REGISTRY
# from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
# from openspeech.dataclass import EVAL_DATACLASS_REGISTRY
# from openspeech.models import MODEL_DATACLASS_REGISTRY
# from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY
# from hydra.core.config_store import ConfigStore

# import time

# from typing import Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Sampler

# from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# #language model import and init
# import time
# import copy

# from transformers import TFBertForMaskedLM
# from transformers import AutoTokenizer
# from transformers import FillMaskPipeline
# from transformers import logging


# model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
# tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
# logging.set_verbosity_error()

# def _collate_fn(batch, pad_id: int = 0):

#     def seq_length_(p):
#         return len(p[0])

#     def target_length_(p):
#         return len(p[1])

#     # sort by sequence length for rnn.pack_padded_sequence()
#     batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

#     seq_lengths = [len(s[0]) for s in batch]
#     target_lengths = [len(s[1]) - 1 for s in batch]

#     max_seq_sample = max(batch, key=seq_length_)[0]
#     max_target_sample = max(batch, key=target_length_)[1]

#     max_seq_size = max_seq_sample.size(0)
#     max_target_size = len(max_target_sample)

#     feat_size = max_seq_sample.size(1)
#     batch_size = len(batch)

#     seqs = torch.zeros(batch_size, max_seq_size, feat_size)

#     targets = torch.zeros(batch_size, max_target_size).to(torch.long)
#     targets.fill_(pad_id)

#     for x in range(batch_size):
#         sample = batch[x]
#         tensor = sample[0]
#         target = sample[1]
#         seq_length = tensor.size(0)

#         seqs[x].narrow(0, 0, seq_length).copy_(tensor)
#         targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

#     seq_lengths = torch.IntTensor(seq_lengths)
#     target_lengths = torch.IntTensor(target_lengths)

#     return seqs, targets, seq_lengths, target_lengths


# class AudioDataLoader(DataLoader):

#     def __init__(
#         self,
#         dataset: torch.utils.data.Dataset,
#         num_workers: int,
#         batch_sampler: torch.utils.data.sampler.Sampler,
#         **kwargs,
#     ) -> None:
#         super(AudioDataLoader, self).__init__(
#             dataset=dataset,
#             num_workers=num_workers,
#             batch_sampler=batch_sampler,
#             **kwargs,
#         )
#         self.collate_fn = _collate_fn
        
# def load_dataset(manifest_file_path: str) -> Tuple[list, list]:
    
#     audio_paths = list()
#     transcripts = list()

#     with open(manifest_file_path) as f:
#         for idx, line in enumerate(f.readlines()):
#             audio_path, korean_transcript, transcript = line.split("\t")
#             transcript = transcript.replace("\n", "")

#             audio_paths.append(audio_path)
#             transcripts.append(transcript)

#     return audio_paths, transcripts

# def language_model(script):
#     sep=script.split()
#     masked_script=[]

#     pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

#     cnt = len(sep)
#     print("words: ", cnt, "\n")

#     for i in range(cnt):
#         print(i, " : ")
#         print(script)
#         temp_list = copy.deepcopy(sep)
#         temp_list[i]='[MASK]'
#         sc=' '.join(temp_list)
#         print(sc)
#         print(pip(sc)[0]['sequence'], "\n")

#     return sc
    

# def hydra_inference_init() -> None:
#     from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
#     from openspeech.dataclass import EVAL_DATACLASS_REGISTRY
#     from openspeech.models import MODEL_DATACLASS_REGISTRY
#     from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY

#     registries = {
#         "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
#         "eval": EVAL_DATACLASS_REGISTRY,
#         "model": MODEL_DATACLASS_REGISTRY,
#         "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
#     }

#     cs = ConfigStore.instance()

#     for group in registries.keys():
#         dataclass_registry = registries[group]

#         for k, v in dataclass_registry.items():
#             cs.store(group=group, name=k, node=v, provider="openspeech")


# @hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="eval")
# def hydra_main(configs: DictConfig) -> None:

#     use_cuda = configs.eval.use_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     audio_paths, transcripts = load_dataset(configs.eval.manifest_file_path)
#     tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

#     model = MODEL_REGISTRY[configs.model.model_name]
#     model = model.load_from_checkpoint(configs.eval.checkpoint_path, configs=configs, tokenizer=tokenizer)
#     model.to(device)

#     if configs.eval.beam_size > 1:
#         model.set_beam_decoder(beam_size=configs.eval.beam_size)

#     dataset = SpeechToTextDataset(
#         configs=configs,
#         dataset_path=configs.eval.dataset_path,
#         audio_paths=audio_paths,
#         transcripts=transcripts,
#         sos_id=tokenizer.sos_id,
#         eos_id=tokenizer.eos_id,
#     )
    
#     sampler = RandomSampler(data_source=dataset, batch_size=configs.eval.batch_size)
    
#     print("ckpt")
#     ckpt = time.time()
#     data_loader = list(AudioDataLoader(
#         dataset=dataset,
#         num_workers=configs.eval.num_workers,
#         batch_sampler=sampler,
#     ))
#     print("time :", time.time() - ckpt)
#     ckpt = time.time()
        
#     input_sentence = ""
#     #batch = next(iter(data_loader))
#     #print(sys.getsizeof(batch))
    
#     with torch.no_grad():
#         # inputs, targets, input_lengths, target_lengths = batch
#         outputs = model(data_loader[0][0].to(device), data_loader[0][2].to(device))
        
#     input_sentence = tokenizer.decode(outputs["predictions"])[0]
#     print("\nprediction\t: ", input_sentence)

#     print("time :", time.time() - ckpt)
#     ckpt = time.time()
    
#     #lanugae model
#     language_model(input_sentence)
#     print("time :", time.time() - ckpt)

# if __name__ == "__main__":
    
#     hydra_inference_init()
#     hydra_main()



import librosa
import torch
import numpy as np
from torch import Tensor
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from omegaconf import DictConfig, OmegaConf

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

eval = OmegaConf.create(
    {
        "use_cuda": True,
        "dataset_path": "",
        "checkpoint_path": "../models/joint_ctc_las.ckpt",
        "manifest_file_path": "",
        "result_path": "",
        "num_workers": "4",
        "batch_size": "32",
        "beam_size": "1",
    }
)

model = OmegaConf.create(
    {
        "model_name": "joint_ctc_listen_attend_spell",
        "num_encoder_layers": 3,
        "frame_lengt": 2,
        "hidden_state_dim": 196,
        "encoder_dropout_p": 0.3,
        "encoder_bidirectional": True,
        "rnn_type": "lstm",
        "joint_ctc_attention": True,
        "max_length": 96,
        "num_attention_heads": 1,
        "decoder_dropout_p": 0.2,
        "decoder_attn_mechanism": "loc",
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
        "vocab_path": "../aihub_labels.csv",
    }
)

configs = OmegaConf.create(
    {"audio": audio, "eval": eval, "model": model, "tokenizer": tokenizer}
)

tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

model = MODEL_REGISTRY[configs.model.model_name]
print(model)
model = model.load_from_checkpoint(configs.eval.checkpoint_path, configs=configs, tokenizer=tokenizer)

def transform_input(signal):
    melspectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=configs.audio.sample_rate,
            n_mels=configs.audio.num_mels,
            n_fft=2048,
            hop_length=512,
        )
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    return melspectrogram

def parse_audio(filepath: str) -> Tensor:

    signal, sr = librosa.load(filepath, sr=None)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
    feature = transform_input(signal)
    
    feature -= feature.mean()
    feature /= np.std(feature)

    feature = torch.FloatTensor(feature).transpose(0, 1)
    print(feature.shape)

    return feature


def inference(feature):
    with torch.no_grad():
        outputs = model(feature.unsqueeze(0), torch.Tensor([1]).to('cuda'))
    print(outputs)
    prediction = tokenizer.decode(outputs["predictions"].cpu().detach().numpy())
    print(prediction)

    return prediction

def upload():
    filepath = "C:/Users/lab1080/Desktop/rasa_sr/openspeech/test_file/audio/voice.wav"

    # load file
    feature = parse_audio(filepath)
    
    print(feature)
    
    feature = feature.to('cuda')

    prediction = inference(feature)
    # os.remove(filepath)

    print(prediction)
    
    return {'prediction': prediction}

if __name__ == "__main__":
    upload()