import librosa
import matplotlib.pyplot as plt

import os
import json
import math
import sys
import argparse

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import langdetect

from scipy.io.wavfile import write
import re
from scipy import signal

# - paths
path_to_config = f"./models/{sys.argv[1]}/config.json"
path_to_model = f"./models/{sys.argv[1]}/G_{sys.argv[2]}.pth"

#- text input
input = "先生、何かお手伝いしましょうか？"

# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

hps = utils.get_hparams_from_file(f"./models/{sys.argv[1]}/config.json")

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers, #- >0 for multi speaker
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path_to_model, net_g, None)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def langdetector(text):  # from PolyLangVITS
    try:
        lang = langdetect.detect(text).lower()
        if lang == 'ko':
            return f'[KO]{text}[KO]'
        elif lang == 'ja':
            return f'[JA]{text}[JA]'
        elif lang == 'en':
            return f'[EN]{text}[EN]'
        elif lang == 'zh-cn':
            return f'[ZH]{text}[ZH]'
        else:
            return text
    except Exception as e:
        return text


output_dir = f'./vitsoutput/{sys.argv[1]}'
os.makedirs(output_dir, exist_ok=True)
speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]

command_args = sys.argv[1:]

fltstr = re.sub(r"[\[\]\(\)\{\}]", "", input)

if "--poly" in command_args:
    fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
stn_tst = get_text(fltstr, hps)

speed = 1
    
for idx, speaker in enumerate(speakers):
    sid = torch.LongTensor([idx]).cuda()
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][0,0].data.cpu().float().numpy()
    write(f'{output_dir}/{speaker}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/{speaker}.wav Generated!')
