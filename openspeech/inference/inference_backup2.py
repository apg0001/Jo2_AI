import os
import warnings

import hydra
import torch
from omegaconf import DictConfig

from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
from openspeech.dataclass import EVAL_DATACLASS_REGISTRY
from openspeech.models import MODEL_DATACLASS_REGISTRY
from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY
from hydra.core.config_store import ConfigStore

import time

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


#init language model
lm_model_name = "skt/kogpt2-base-v2"
lm_model = GPT2LMHeadModel.from_pretrained(lm_model_name)
lm_tokenizer = GPT2TokenizerFast.from_pretrained(lm_model_name)

def _collate_fn(batch, pad_id: int = 0):

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_workers: int,
        batch_sampler: torch.utils.data.sampler.Sampler,
        **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn
        
def load_dataset(manifest_file_path: str) -> Tuple[list, list]:
    
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, korean_transcript, transcript = line.split("\t")
            transcript = transcript.replace("\n", "")

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts

start = time.time()

def hydra_inference_init() -> None:
    from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from openspeech.dataclass import EVAL_DATACLASS_REGISTRY
    from openspeech.models import MODEL_DATACLASS_REGISTRY
    from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "eval": EVAL_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in registries.keys():
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="openspeech")


@hydra.main(config_path=os.path.join("openspeech", "configs"), config_name="eval")
def hydra_main(configs: DictConfig) -> None:

    use_cuda = configs.eval.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    audio_paths, transcripts = load_dataset(configs.eval.manifest_file_path)
    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    model = MODEL_REGISTRY[configs.model.model_name]
    model = model.load_from_checkpoint(configs.eval.checkpoint_path, configs=configs, tokenizer=tokenizer)
    model.to(device)

    if configs.eval.beam_size > 1:
        model.set_beam_decoder(beam_size=configs.eval.beam_size)

    dataset = SpeechToTextDataset(
        configs=configs,
        dataset_path=configs.eval.dataset_path,
        audio_paths=audio_paths,
        transcripts=transcripts,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    )
    
    sampler = RandomSampler(data_source=dataset, batch_size=configs.eval.batch_size)
    
    data_loader = AudioDataLoader(
        dataset=dataset,
        num_workers=configs.eval.num_workers,
        batch_sampler=sampler,
    )
        
    input_sentence = ""
    
    ckpt1 = time.time()   
    
    # for i, (batch) in enumerate(data_loader):
        
    #     print("ckpt1")
    #     print("time :", time.time() - ckpt1)
    #     ckpt2 = time.time()
        
    #     with torch.no_grad():
    #         # inputs, targets, input_lengths, target_lengths = batch
    #         outputs = model(batch[0].to(device), batch[2].to(device))
    #         print("output")
    #         print("time :", time.time() - start)
            
    #     print("ckpt2")
    #     print("time :", time.time() - ckpt2)
    #     ckpt3 = time.time()

    #     input_sentence = tokenizer.decode(outputs["predictions"])[0]
    #     print("\nprediction\t: ", input_sentence)
        
    #     print("ckpt3")
    #     print("time :", time.time() - ckpt3)

    #     #lm

    #     input_ids = lm_tokenizer.encode(input_sentence, return_tensors='pt')
    #     with torch.no_grad():
    #         print(len(input_sentence))
    #         # output_sequences = model.generate(input_ids, max_length=len(input_sentence), num_beams=5, no_repeat_ngram_size=2, top_k=0, top_p=0.95, temperature=0.7, num_return_sequences=1)
    #         output_sequences = lm_model.generate(input_ids, max_length=len(input_sentence), num_beams=5, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7, num_return_sequences=1)

    #     for generated_sequence in output_sequences:
    #         generated_sequence = generated_sequence.tolist()
    #         print("\nLanguage model result : {0}".format(lm_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
    
    print("ckpt1")
    print("time :", time.time() - ckpt1)
    ckpt2 = time.time()
    
    batch = next(iter(data_loader))
    with torch.no_grad():
        # inputs, targets, input_lengths, target_lengths = batch
        outputs = model(batch[0].to(device), batch[2].to(device))
        print("output")
        print("time :", time.time() - start)
        
    print("ckpt2")
    print("time :", time.time() - ckpt2)
    ckpt3 = time.time()

    input_sentence = tokenizer.decode(outputs["predictions"])[0]
    print("\nprediction\t: ", input_sentence)
    
    print("ckpt3")
    print("time :", time.time() - ckpt3)

    #lm

    input_ids = lm_tokenizer.encode(input_sentence, return_tensors='pt')
    with torch.no_grad():
        print(len(input_sentence))
        # output_sequences = model.generate(input_ids, max_length=len(input_sentence), num_beams=5, no_repeat_ngram_size=2, top_k=0, top_p=0.95, temperature=0.7, num_return_sequences=1)
        output_sequences = lm_model.generate(input_ids, max_length=len(input_sentence), num_beams=5, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7, num_return_sequences=1)

    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        print("\nLanguage model result : {0}".format(lm_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))

if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    start = time.time()
    print("start")
    print("time :", time.time() - start)
    
    hydra_inference_init()
    print("init")
    print("time :", time.time() - start)
    
    hydra_main()
    print("complete")
    print("time :", time.time() - start)



