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
import torch.nn as nn


exp = "outputs/1_overfit_conf_ln" ### path to experiment directory
#exp = "outputs/1_overfit_gru" ### path to experiment directory

args_path = os.path.join(exp, "args.json")

### read args from json file
with open(args_path, "r") as f:
    args = json.load(f)

model_sr = args["samplerate"]
ir_length = args["rir_length"]
device = 'cpu'

epoch = 500

# init neural net
ckpt_path = os.path.join(exp, f"checkpoints/model_e{epoch}.pt")

net = ASPestNet(None, rir_length=args["rir_length"], conf_backbone=args["conf_backbone"])
weights = torch.load(ckpt_path, map_location=device) # load weights
net.load_state_dict(weights) # load weights ins nn
net = net.eval()

eval_path = os.path.join(exp, f"audio_input")
eval_files = os.listdir(eval_path)#random.choice(os.listdir(eval_path))
eval_filepaths = [os.path.join(eval_path, eval_file) for eval_file in eval_files if eval_file.endswith('.wav')]

print(eval_filepaths)

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
    with torch.no_grad():
        pred, H, _, _, params = net(eval_ir, z)


    #pred_np = pred.squeeze().cpu().numpy()
   
    print(f"pred mean: {torch.mean(pred)}")
    print(f"H mean: {torch.mean(H)}")
    HH.append(H)
    preds.append(pred)


preds = torch.stack(preds)
HH = torch.stack(HH)

# mse_loss = nn.MSELoss()
# mse = mse_loss(preds[0], preds[1])
# mse_H = mse_loss(torch.abs(HH[0]), torch.abs(HH[1]))
mse = torch.mean((preds - preds.mean(dim=0, keepdim=True))**2)
mse_H = torch.mean((HH - HH.mean(dim=0, keepdim=True))**2)

# print("-------------------------------------------------------")
# print("preds MSE:", mse.item())
# print("H MSE:", mse_H.item())

# plot

#pred_np = pred.detach().cpu().numpy()
pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
eval_np = eval_ir.squeeze(0).detach().cpu().numpy()

pred_real = np.abs(pred_np).astype(np.float32)

#------------------------------------------------------------------------------
# plots 

times = np.zeros(len(pred_np))
eval_sig = pf.Signal([eval_np.flatten(),times.flatten()],sampling_rate=model_sr)
pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=model_sr)

pred_sig = pf.dsp.normalize(pred_sig[0])


eval_sig = eval_sig[0]
#pred_sig = pred_sig[0]

pf.plot.time_freq(eval_sig,label="eval", alpha=0.3)
pf.plot.time_freq(pred_sig,label="pred", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(save_path, f"{eval_file}-plot.pdf"))"""

speech = pf.signals.files.speech(sampling_rate=eval_sig.sampling_rate)
guitar = pf.signals.files.guitar(sampling_rate=eval_sig.sampling_rate)
drums = pf.signals.files.drums(sampling_rate=eval_sig.sampling_rate)


def convolve_signals(dry_signal,irs,plot=False):

    conv_eval = pf.dsp.normalize(pf.dsp.convolve(dry_signal,irs[0]))
    conv_pred = pf.dsp.normalize(pf.dsp.convolve(dry_signal,irs[1]))
    
    if plot == True:
        pf.plot.time_freq(dry_signal,label="dry", alpha=0.2)
        pf.plot.time_freq(conv_eval,label="eval",color='black',alpha=0.8)
        pf.plot.time_freq(conv_pred,label="pred", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return conv_eval, conv_pred

#convolve_signals(drums,[eval_sig,pred_sig],plot=True)

#------------------------------------------------------------------------------
# save files 

# create path to write audiofiles 
audio_save_path = f"outputs/{exp.split('/')[-1]}/audio_output/epoch-{epoch}"
os.makedirs(audio_save_path, exist_ok=True)
# write prediction to wav file
sf.write(f"{audio_save_path}/epoch-{epoch}.wav", pred_real, model_sr) 
# write convolved signals to wav file
signals = [speech,guitar,drums]
names = ["speech","guitar","drums"]

for signal,name in zip(signals,names):
    signal_eval, signal_pred = convolve_signals(signal,[eval_sig,pred_sig])
    sf.write(f"{audio_save_path}/convolved_eval_{name}_e{epoch}.wav", 
             signal_eval.time.flatten(), model_sr) 
    sf.write(f"{audio_save_path}/convolved_pred_{name}_e{epoch}.wav", 
             signal_pred.time.flatten(), model_sr)


