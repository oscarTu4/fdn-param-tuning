import torch
import torchaudio
from torchaudio import transforms
import os
from model import *
import numpy as np
from utils.utility import *
from utils.processing import *
import json
from matplotlib import pyplot as plt
import pyfar as pf
import soundfile as sf


# ----------------------------------------------------------------- #
# ENTER INFERENCE INFORMATION HERE                                  #
# The code will pull the correct checkpoint given your information  #
# ----------------------------------------------------------------- #

### path to training directory (this should be in your outputs folder)
exp = "outputs/+++Unirechner_Trainings/Conf 2203-1000"
# epoch you want to evaluate (training saves every even number of epochs)
epoch = 16
# path to impulse response you want to create a FDN IR from
eval_path = "/Users/oscar/Documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/train_of/SC_MC_EIG_1.wav"

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

args_path = os.path.join(exp, "args.json")

### read args from json file
with open(args_path, "r") as f:
    args = json.load(f)

model_sr = args["samplerate"]
ir_length = args["rir_length"]
conf = args["conf_backbone"]
device = 'cpu'

# init neural net
ckpt_path = os.path.join(exp, f"checkpoints/model_e{epoch}.pt")
net = ASPestNet(rir_length=ir_length, conf_backbone=conf)
# load weights
weights = torch.load(ckpt_path, map_location=device)
net.load_state_dict(weights)
net = net.eval()

#eval_files = os.listdir(eval_path)
#eval_filepaths = [os.path.join(eval_path, eval_file) for eval_file in eval_files if eval_file.endswith('.wav')]

#print(f"eval_filepaths: {eval_filepaths}")

#preds = []
#HH = []

#for eval_filepath in eval_filepaths:
eval_ir, sr = torchaudio.load(eval_path)
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

#print(f"pred mean: {torch.mean(pred)}")
#print(f"H mean: {torch.mean(H)}")
#HH.append(H)
#preds.append(pred)

# plot

pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
eval_np = eval_ir.squeeze(0).squeeze(0).detach().cpu().numpy()

pred_real = np.abs(pred_np).astype(np.float32)
eval_real = np.abs(eval_np).astype(np.float32)

#------------------------------------------------------------------------------
# plots 

times = np.zeros(len(pred_np))
eval_sig = pf.Signal([eval_np.flatten(),times.flatten()],sampling_rate=model_sr)
pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=model_sr)

pred_sig = pf.dsp.normalize(pred_sig[0])
eval_sig = eval_sig[0]

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

#------------------------------------------------------------------------------
# save files 

# save audiofiles to output folder
audio_save_path = os.path.join(exp, "local_inference")
os.makedirs(audio_save_path, exist_ok=True)
sf.write(f"{audio_save_path}/predicted_IR.wav", pred_real, model_sr)
sf.write(f"{audio_save_path}/input_IR.wav", eval_real, model_sr)
signals = [speech,guitar,drums]
names = ["speech","guitar","drums"]

# convolve signals and save to audiofile
for signal,name in zip(signals,names):
    signal_eval, signal_pred = convolve_signals(signal,[eval_sig,pred_sig])
    sf.write(f"{audio_save_path}/convolved_eval_{name}_e{epoch}.wav", 
             signal_eval.time.flatten(), model_sr) 
    sf.write(f"{audio_save_path}/convolved_pred_{name}_e{epoch}.wav", 
             signal_pred.time.flatten(), model_sr)