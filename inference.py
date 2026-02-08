import torch
import torchaudio
from torchaudio import transforms
import os
import time
from model import *
import pandas as pd
import numpy as np
#import alt.audio_utility as util
from utils.utility import *
from utils.processing import *
import json
from matplotlib import pyplot as plt
import pyfar as pf
import soundfile as sf
import sounddevice as sd 
import random
import shutil


exp = "outputs/1_overfit_conf_ln" ### path to experiment directory
args_path = os.path.join(exp, "args.json")

### read args from json file
with open(args_path, "r") as f:
    args = json.load(f)

model_sr = args["samplerate"]
ir_length = args["rir_length"]
device = 'cpu'

# init neural net
ckpt_path = os.path.join(exp, f"checkpoints/model_e1500.pt")

net = ASPestNet(None, rir_length=args["rir_length"], conf_backbone=args["conf_backbone"])
weights = torch.load(ckpt_path, map_location=device) # load weights
net.load_state_dict(weights) # load weights ins nn
net = net.eval()

eval_path = "/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/train_of/"
eval_files = os.listdir(eval_path)#random.choice(os.listdir(eval_path))
eval_filepaths = [os.path.join(eval_path, eval_file) for eval_file in eval_files if eval_file.endswith('.wav')]

preds = []
HH = []

for eval_filepath in eval_filepaths:
    eval_ir, sr = torchaudio.load(eval_filepath)
    if sr != model_sr:
        tf = transforms.Resample(sr, model_sr)
        eval_ir = tf(eval_ir)
        sr = model_sr

    if eval_ir.shape[0] != 1:
        eval_ir = eval_ir.mean(dim=0, keepdim=True)

    eval_ir = pad_crop(eval_ir, sr, ir_length)
    if eval_ir.ndim == 2:        # [C, T] zu [B, C, T]
        eval_ir = eval_ir.unsqueeze(0)
    z = get_frequency_samples(120000//2+1)
    
    pred, H, _, _, params = net(eval_ir, z)
    print(f"pred mean: {torch.mean(pred)}")
    print(f"H mean: {torch.mean(H)}")
    HH.append(H)
    preds.append(pred)

preds = torch.stack(preds)
HH = torch.stack(HH)

import torch.nn as nn
mse_loss = nn.MSELoss()
mse = mse_loss(preds[0], preds[1])
mse_H = mse_loss(torch.abs(HH[0]), torch.abs(HH[1]))
#mse = torch.mean((preds - preds.mean(dim=0, keepdim=True))**2)
#mse_H = torch.mean((HH - HH.mean(dim=0, keepdim=True))**2)

print("-------------------------------------------------------")
print("preds MSE:", mse.item())
print("H MSE:", mse_H.item())

# plot
#pred_np = normalize_energy(pred)
"""pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
eval_np = eval_ir.squeeze(0).detach().cpu().numpy()

pred_real = np.abs(pred_np).astype(np.float32)

save_path = f"outputs/generated_irs/{exp.split('/')[-1]}"
os.makedirs(save_path, exist_ok=False)
sf.write(f"{save_path}/{eval_file}-pred.wav", pred_real, model_sr)
shutil.copy(eval_filepath, save_path)
#sf.write(f"{save_path}/eval.wav", eval_ir.squeeze(0), model_sr)

times = np.zeros(len(pred_np))
eval_sig = pf.Signal([eval_np.flatten(),times.flatten()],sampling_rate=model_sr, is_complex=True)
pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=model_sr, is_complex=True)

#pf.plot.time_freq(eval_sig, label="eval", alpha=0.3)
pf.plot.time_freq(pred_sig, label="pred", alpha=0.7)
plt.legend()
plt.savefig(os.path.join(save_path, f"{eval_file}-plot.pdf"))"""