import torch
import torchaudio
from torchaudio import transforms
import os
import time
from model import DiffFDN
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


exp = "outputs/20260123-174833" ### path to experiment directory
args_path = os.path.join(exp, "args.json")

### read args from json file
with open(args_path, "r") as f:
    args = json.load(f)

model_sr = args["samplerate"]
ir_length = args["ir_length"]
device = 'cpu'

# init neural net
epoch = 10
ckpt_path = os.path.join(exp, f"checkpoints/model_e{epoch}.pt")
filepath = 'Params/'
N = args["N"]
delay_set = args["delay_set"]
filename = 'param' + '_N' + str(N) + '_d' + str(delay_set)

df = pd.read_csv(filepath+filename+'.csv', delimiter=';', nrows=N*N, dtype={'A':np.float32,'m':'Int32'})
delay_lens = torch.from_numpy(df['m'][:N].to_numpy())

net = DiffFDN(delay_lens, model_sr, ir_length) # nn init
weights = torch.load(ckpt_path, map_location=device) # load weights
net.load_state_dict(weights) # load weights ins nn
net = net.eval()

#hier wird die IR gefüttert und eine prediction gemacht:

### TODO loop über mehrere IRs, dann kann man epochen untereinander vergleichen
eval_path = "/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/eval/"
eval_file = random.choice(os.listdir(eval_path))
eval_filepath = os.path.join(eval_path, eval_file)
eval_ir, sr = torchaudio.load(eval_filepath)

### t60 berechnung, wird als gain_per_sample (γ in paper) übergeben.
### trainiert wird auf fixer t60, hier bei inference wird es als freier parameter übergeben
### paper: Santo: Optimizing tiny colorless feedback delay networks
"""t60 = eval_ir.size()[-1]
print(f"t60: {t60}")
print(f"t60-s: {t60/model_sr}")
gamma_ = 10**(-3/t60)
print(f"gamma_: {gamma_}")"""

if sr != model_sr:
    tf = transforms.Resample(sr, model_sr)
    eval_ir = tf(eval_ir)
    sr = model_sr

if eval_ir.shape[0] != 1:
    eval_ir = eval_ir.mean(dim=0, keepdim=True)

eval_ir = pad_crop(eval_ir, sr, ir_length)
if eval_ir.ndim == 2:        # [C, T] zu [B, C, T]
    eval_ir = eval_ir.unsqueeze(0)

z = get_frequency_samples(int(args['ir_length']*args['samplerate']))
t = time.time()

pred, _ = net(eval_ir, z)
print(f"generated pred in {np.round(np.abs(t-time.time()), 2)} seconds")

print(f"pred shape: {pred.shape}")

# plot
#pred_np = normalize_energy(pred)
pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
eval_np = eval_ir.squeeze(0).detach().cpu().numpy()

pred_real = np.abs(pred_np).astype(np.float32)

save_path = f"outputs/generated_irs/{exp.split('/')[-1]}/epoch{epoch}"
os.makedirs(save_path, exist_ok=True)
sf.write(f"{save_path}/{eval_file}-pred.wav", pred_real, model_sr)
shutil.copy(eval_filepath, save_path)
#sf.write(f"{save_path}/eval.wav", eval_ir.squeeze(0), model_sr)

times = np.zeros(len(pred_np))
eval_sig = pf.Signal([eval_np.flatten(),times.flatten()],sampling_rate=model_sr, is_complex=True)
pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=model_sr, is_complex=True)

#pf.plot.time_freq(eval_sig, label="eval", alpha=0.3)
pf.plot.time_freq(pred_sig, label="pred", alpha=0.7)
plt.legend()
plt.savefig(os.path.join(save_path, f"{eval_file}-plot.pdf"))