import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import config
from text.japanese import text2sep_kata

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

models = dict()


def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    text = "".join(text2sep_kata(text)[0])
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
