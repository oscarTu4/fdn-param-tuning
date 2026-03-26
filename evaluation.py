import os
import json
import torch
import torchaudio
import numpy as np
from torchaudio import transforms
from scipy.stats import linregress
from model import ASPestNet
from losses import MSSpectralLoss
from utils.processing import pad_crop
from utils.processing import get_frequency_samples
import pyfar as pf
from datetime import datetime
import pandas as pd

### ADJUST PATH ACCORDING TO FOLDER STRUCTURE AND VARIABLES ###
# variables that might need adjustment: exp, eval_path, dataset und epoch

#exp: path to folder with arg.json and checkpoints
exp = "outputs/Conf 2203-1000"
# eval_path: Path to IRs that we want to evaluate
eval_path = "/Users/flo_steig/Desktop/shoebox_freq_dep_eval/wav" 
#eval_path = "/Users/flo_steig/Desktop/MIT_Survey_IRs"
dataset = "shoebox" # change according to eval_path content
epoch = 16
files = [f for f in os.listdir(eval_path) if f.lower().endswith(".wav")]
csv_dir = os.path.join(exp, "evaluation_results") 
os.makedirs(csv_dir, exist_ok=True)

device = "cpu"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Three IRs from RIR2FDN Paper are additionally evaluated individually (for better comparison with results from RIR2FDN)
SPECIAL_IRS = {
    "lobby": "h229_Office_Lobby_1txts.wav",
    "hallway": "h042_Hallway_ElementarySchool_4txts.wav",
    "meeting": "h110_Office_MeetingRoom_1txts.wav"
}



special_results = []

# Load training params
with open(os.path.join(exp, "args.json"), "r") as f:
    args = json.load(f)

model_sr = args["samplerate"]
rir_length = args["rir_length"]

# initialize model 
ckpt_path = os.path.join(exp, f"checkpoints/model_e{epoch}.pt")
#model_name = ckpt_path.split("/")[-1] 
model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

net = ASPestNet(rir_length=args["rir_length"],conf_backbone=args["conf_backbone"])

weights = torch.load(ckpt_path, map_location=device)
net.load_state_dict(weights)
net.to(device)
net.eval()


### CENTER FREQUENCIES AND FILTERBANK ####

filterbank_pf = pf.dsp.filter.fractional_octave_bands(
    None,
    1,
    sampling_rate=model_sr,
    frequency_range=(125, 8000),
    order=4
)

nom_freq_pf, ex_freq_pf = pf.dsp.filter.fractional_octave_frequencies(
    num_fractions=1,
    frequency_range=(125, 8000),
    return_cutoff=False
)

#### FUNCTIONS FOR EVALUATION ####

# Energy Decay Rate acc. to  equation 15 and 16 in Lee et al. S. 5

def compute_stft_edr_np(ir_np, n_fft=2048):

    ir_np = np.squeeze(ir_np)

    if len(ir_np) < n_fft:
        return np.nan


    hop = n_fft // 4
    window = np.hanning(n_fft)

    # STFT 
    frames = []
    for start in range(0, len(ir_np) - n_fft +1, hop):
        frame = ir_np[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        frames.append(np.abs(spectrum) ** 2)

    energy = np.array(frames).T  # shape: [freq, time]

    # Backwards cumulative sum along time axis
    edr = np.flip(
        np.cumsum(
            np.flip(energy, axis=1),
            axis=1
        ),
        axis=1
    )

    return edr

def compute_edr_distance_np(ir_np, pred_np, n_fft=2048):

    edr_ref = compute_stft_edr_np(ir_np, n_fft)
    edr_pred = compute_stft_edr_np(pred_np, n_fft)

    if isinstance(edr_ref, float) and np.isnan(edr_ref):
        return np.nan
    if isinstance(edr_pred, float) and np.isnan(edr_pred):
        return np.nan

    eps = 1e-10

    log_ref = 10 * np.log10(np.maximum(edr_ref, eps))
    log_pred = 10 * np.log10(np.maximum(edr_pred, eps))

    edr_error = np.abs(log_ref - log_pred)

    return np.mean(edr_error)


# Schroeder Integral acc. to Schroeder, M. R. (1965)
def schroeder(ir):
    energy = ir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    edc = edc / np.max(edc)
    edc_db = 10 * np.log10(edc + 1e-12)
    return edc_db

# T30, ISO 3382-1 (2009)
def compute_t30(ir, fs):
    edc_db = schroeder(ir)
    t = np.arange(len(edc_db)) / fs
    # Look for interval from -5 to -30 
    idx = np.where((edc_db <= -5) & (edc_db >= -35))[0]
    if len(idx) < 2:
        return np.nan
    # Regression analysis over all points in the range -5 to -35 dB
    slope, intercept, _, _, _ = linregress(t[idx], edc_db[idx])
    # T30, estimated from slope 
    t30 = -60 / slope
    return t30

# Compute Clarity (C50), acc. to ISO 3382-1 (2009)
# Balance between early and late energy
def compute_c50(ir, fs):
    split = int(0.05 * fs)
    early = np.sum(ir[:split] ** 2) # first 50 ms
    late = np.sum(ir[split:] ** 2) # everything after that 
    return 10 * np.log10((early + 1e-12) / (late + 1e-12))

# DRR(Direct to Reverberant Energy Ratio), acc. to Larsen (2008)
def compute_drr(ir, fs):
    # Find Maximum in IR , Assumption: Maximum approximately direct sound
    peak = np.argmax(np.abs(ir))
    # Larson (2008): they used 3ms, using 2-3ms is common practice
    window = int(0.0015 * fs) # 3ms/2, applied on both sides
    start = max(0, peak - window)
    end = min(len(ir), peak + window)

    direct = np.sum(ir[start:end] ** 2)
    reverb = np.sum(ir[end:] ** 2)

    return 10 * np.log10((direct + 1e-12) / (reverb + 1e-12))


def octave_band_ir_pf(ir_np, fs, filterbank_pf):

    signal = pf.Signal(ir_np, sampling_rate=fs)

    # Fensterung, ggf. weglassen wegen pad_crop
    # signal = pf.dsp.time_window(signal, [0, 4.5], 
    #                             window='boxcar',
    #                             unit='s',
    #                             crop='window')

    # applying octavefilter 
    filtered = filterbank_pf.process(signal)
    return filtered

def evaluate_jnd_t30(delta_rel):
    if delta_rel < 5:
        return "below JND"
    elif delta_rel < 25:
        return "within reported JND range"
    else:
        return "above JND"

def evaluate_jnd_c50(delta):
    return "below JND" if delta < 1 else "above JND"

# def evaluate_jnd_drr(delta):
#     return "below JND" if delta < 3 else "above JND"
# commented out because depends on global level, 3 is just an example


# EVALUATION
#eval_path = os.path.join(exp, "audio_input")

delta_t30_full = []
delta_c50_full = []
delta_drr_full = []
match_losses = []
edr_errors = []
delta_t30_oct = {fc: [] for fc in nom_freq_pf}
delta_c50_oct = {fc: [] for fc in nom_freq_pf}
delta_drr_oct = {fc: [] for fc in nom_freq_pf}
delta_t30_oct_avg = []
delta_c50_oct_avg = []
delta_drr_oct_avg = []
delta_t30_rel_full = []
delta_t30_rel_oct_avg = []

criterion = MSSpectralLoss(sr=model_sr)

total_files = len(files)

#for file in files:
for idx, file in enumerate(files, 1):
    is_special = file in SPECIAL_IRS.values()
    print(f"[{idx}/{total_files}] Processing {file}")

    path = os.path.join(eval_path, file)
    ir, sr = torchaudio.load(path)

    if sr != model_sr:
        tf = transforms.Resample(sr, model_sr)
        ir = tf(ir)
        sr = model_sr

    if ir.shape[0] != 1:
        ir = ir.mean(dim=0, keepdim=True)
    #print("pad crop")
    ir = pad_crop(ir, sr, rir_length)
    ir = ir.unsqueeze(0)

    z = get_frequency_samples(120000 // 2 + 1)

    ir = ir.to(device)
    z = z.to(device)


  
    with torch.no_grad():
        pred, _, _, _, _ = net(ir, z)

    ir_np = ir.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    # Alignment of the IRs
    ref_peak = np.argmax(np.abs(ir_np))
    pred_peak = np.argmax(np.abs(pred_np))

    shift = pred_peak - ref_peak

    if shift > 0:
        # pred too late, move left
        pred_np = np.concatenate((pred_np[shift:], np.zeros(shift)))
    elif shift < 0:
        # pred too early, move right
        pred_np = np.concatenate((np.zeros(-shift), pred_np[:shift]))

    #print("Calculating metrics...")
    # EDR ERRO
    #print("Calculating EDR...")
    # edr_ref = compute_edr(ir_np)
    # edr_pred = compute_edr(pred_np)
    # edr_errors.append(np.mean(np.abs(edr_ref - edr_pred)))
    edr_distance = compute_edr_distance_np(ir_np, pred_np)
    edr_errors.append(edr_distance)
    

    # FULLBAND PARAMETERS
    #print("Calculating T30...")
    t30_ref = compute_t30(ir_np, sr)
    t30_pred = compute_t30(pred_np, sr)
    delta_t30_full.append(abs(t30_pred - t30_ref))

    #print("Calculating C50...")
    c50_ref = compute_c50(ir_np, sr)
    c50_pred = compute_c50(pred_np, sr)
    delta_c50_full.append(abs(c50_pred - c50_ref))

    #print("Calculating DRR...")
    drr_ref = compute_drr(ir_np, sr)
    drr_pred = compute_drr(pred_np, sr)
    delta_drr_full.append(abs(drr_pred - drr_ref))


    # JND PARAMETERS
    if not np.isnan(t30_ref) and t30_ref != 0:
        delta_rel = abs(t30_pred - t30_ref) / t30_ref * 100
        delta_t30_rel_full.append(delta_rel)


    # OCTAVE BAND PARAMETERS
    band_deltas_t30 = []
    band_deltas_c50 = []
    band_deltas_drr = []
    band_rel = []

    bands_ref = octave_band_ir_pf(ir_np, sr, filterbank_pf)
    bands_pred = octave_band_ir_pf(pred_np, sr, filterbank_pf)

    band_ref_t30 = []
    band_pred_t30 = []

    band_ref_c50 = []
    band_pred_c50 = []

    band_ref_drr = []
    band_pred_drr = []
    #print("calc octave bands")
    for i, fc in enumerate(nom_freq_pf):

        ir_band_ref = bands_ref[i].time.flatten()
        ir_band_pred = bands_pred[i].time.flatten()

        t30_ref_band = compute_t30(ir_band_ref, sr)
        t30_pred_band = compute_t30(ir_band_pred, sr)

        c50_ref_band = compute_c50(ir_band_ref, sr)
        c50_pred_band = compute_c50(ir_band_pred, sr)

        drr_ref_band = compute_drr(ir_band_ref, sr)
        drr_pred_band = compute_drr(ir_band_pred, sr)

        band_ref_t30.append(t30_ref_band)
        band_pred_t30.append(t30_pred_band)

        band_ref_c50.append(c50_ref_band)
        band_pred_c50.append(c50_pred_band)

        band_ref_drr.append(drr_ref_band)
        band_pred_drr.append(drr_pred_band)

        delta_t30_oct[fc].append(abs(t30_pred_band - t30_ref_band))
        delta_c50_oct[fc].append(abs(c50_pred_band - c50_ref_band))
        delta_drr_oct[fc].append(abs(drr_pred_band - drr_ref_band))

        delta_t30 = abs(t30_pred_band - t30_ref_band)
        delta_c50 = abs(c50_pred_band - c50_ref_band)
        delta_drr = abs(drr_pred_band - drr_ref_band)

        band_deltas_t30.append(delta_t30)
        band_deltas_c50.append(delta_c50)
        band_deltas_drr.append(delta_drr)

        if not np.isnan(t30_ref_band) and t30_ref_band != 0:
            band_rel.append(abs(t30_pred_band - t30_ref_band) / t30_ref_band * 100)

    if len(band_rel) > 0:
        delta_t30_rel_oct_avg.append(np.nanmean(band_rel))
    
    delta_t30_oct_avg.append(np.nanmean(band_deltas_t30))
    delta_c50_oct_avg.append(np.nanmean(band_deltas_c50))
    delta_drr_oct_avg.append(np.nanmean(band_deltas_drr))

       # MATCH LOSS
    #criterion = MSSpectralLoss(sr=model_sr)
    loss = criterion(pred, ir)
    match_losses.append(loss.item())


    if is_special:

        special_row = {
            "file": file,
            "epoch": epoch,
            "match_loss": loss.item(),
            "edr_error": edr_distance,

            # FULLBAND absolute values
            "T30_ref_full": t30_ref,
            "T30_pred_full": t30_pred,

            "C50_ref_full": c50_ref,
            "C50_pred_full": c50_pred,

            "DRR_ref_full": drr_ref,
            "DRR_pred_full": drr_pred,


            "delta_T30_full": abs(t30_pred - t30_ref),
            "delta_C50_full": abs(c50_pred - c50_ref),
            "delta_DRR_full": abs(drr_pred - drr_ref),

            "delta_T30_rel_percent": delta_rel if not np.isnan(delta_rel) else np.nan,

            "delta_T30_oct_avg": np.nanmean(band_deltas_t30),
            "delta_C50_oct_avg": np.nanmean(band_deltas_c50),
            "delta_DRR_oct_avg": np.nanmean(band_deltas_drr)
        }

        for i, fc in enumerate(nom_freq_pf):

            special_row[f"T30_ref_{int(fc)}Hz"] = band_ref_t30[i]
            special_row[f"T30_pred_{int(fc)}Hz"] = band_pred_t30[i]

            special_row[f"C50_ref_{int(fc)}Hz"] = band_ref_c50[i]
            special_row[f"C50_pred_{int(fc)}Hz"] = band_pred_c50[i]

            special_row[f"DRR_ref_{int(fc)}Hz"] = band_ref_drr[i]
            special_row[f"DRR_pred_{int(fc)}Hz"] = band_pred_drr[i]

            special_row[f"delta_T30_{int(fc)}Hz"] = band_deltas_t30[i]
            special_row[f"delta_C50_{int(fc)}Hz"] = band_deltas_c50[i]
            special_row[f"delta_DRR_{int(fc)}Hz"] = band_deltas_drr[i]

        special_results.append(special_row)


 
# save Terminal Output in txt file (in addition to csv, just for readability)
log_path = os.path.join(csv_dir, f"evaluation_results_{model_name}_{dataset}_{date_str}.txt")
log_file = open(log_path, "w")

def log_print(text=""):
    print(text)
    log_file.write(text + "\n")

log_print("\n==============================")
log_print("EVALUATION RESULTS")
log_print("==============================")
#print(f"Model: {model_name}")
log_print(f"Model: {exp}")
log_print(f"Epoch: {epoch}")
log_print(f"Dataset: {dataset}")
#print(f"Date: {date_str}")
log_print(f"Number of files: {len(files)}")
# to do add header for date etc... 

log_print(f"Median Match Loss: {np.median(match_losses):.6f}")
log_print(f"Median EDR Error: {np.median(edr_errors):.6f}")



log_print("\n--- FULLBAND RESULTS ---")

log_print(f"Median ΔT30: {np.median(delta_t30_full):.4f} s")
log_print(f"Median ΔC50: {np.median(delta_c50_full):.4f} dB")
log_print(f"Median ΔDRR: {np.median(delta_drr_full):.4f} dB")



log_print("\n--- PERCEPTUAL INTERPRETATION (FULLBAND) ---")
median_rel_t30 = np.nanmedian(delta_t30_rel_full)

log_print(f"Median relative ΔT30: {np.nanmedian(delta_t30_rel_full):.2f} %")
log_print(f"T30 perceptual rating:, {evaluate_jnd_t30(np.nanmedian(delta_t30_rel_full))}")

log_print(f"C50 perceptual rating:, {evaluate_jnd_c50(np.median(delta_c50_full))}")
#print("DRR perceptual rating:", evaluate_jnd_drr(np.median(delta_drr_full))) # not useful, since there are many different ranges

log_print("\n--- OCTAVEBAND RESULTS ---")

for fc in nom_freq_pf:
    log_print(f"{fc} Hz:")
    log_print(f"  Median ΔT30: {np.nanmedian(delta_t30_oct[fc]):.4f} s")
    log_print(f"  Median ΔC50: {np.nanmedian(delta_c50_oct[fc]):.4f} dB")
    log_print(f"  Median ΔDRR: {np.nanmedian(delta_drr_oct[fc]):.4f} dB")

log_print("\n--- OCTAVE AVERAGE RESULTS (Mean over 7 Bands, Median over Files) ---")

log_print(f"Median ΔT30 (Oct Avg): {np.nanmedian(delta_t30_oct_avg):.4f} s")
log_print(f"Median ΔC50 (Oct Avg): {np.nanmedian(delta_c50_oct_avg):.4f} dB")
log_print(f"Median ΔDRR (Oct Avg): {np.nanmedian(delta_drr_oct_avg):.4f} dB")

log_print("\n--- PERCEPTUAL INTERPRETATION (OCTAVE AVG) ---")

median_rel_oct = np.nanmedian(delta_t30_rel_oct_avg)

log_print(f"Median relative ΔT30 (Oct Avg): {median_rel_oct:.2f} %")
log_print(f"T30 Oct Avg perceptual rating:, {evaluate_jnd_t30(median_rel_oct)}")

results = {
    "date": date_str,
    "model": model_name,
    "epoch": epoch,
    "median_match_loss": np.median(match_losses),
    "median_edr_error": np.median(edr_errors),
    "median_delta_T30_full": np.median(delta_t30_full),
    "median_delta_C50_full": np.median(delta_c50_full),
    "median_delta_DRR_full": np.median(delta_drr_full),
    "median_delta_T30_oct_avg": np.nanmedian(delta_t30_oct_avg),
    "median_delta_C50_oct_avg": np.nanmedian(delta_c50_oct_avg),
    "median_delta_DRR_oct_avg": np.nanmedian(delta_drr_oct_avg),
    "median_delta_T30_rel_full_percent": np.nanmedian(delta_t30_rel_full), 
    "median_delta_T30_rel_oct_avg_percent": np.nanmedian(delta_t30_rel_oct_avg),
    "T30_full_JND_rating": evaluate_jnd_t30(np.nanmedian(delta_t30_rel_full)), 
    "T30_oct_avg_JND_rating": evaluate_jnd_t30(np.nanmedian(delta_t30_rel_oct_avg)),
    #"DRR_full_JND_rating": evaluate_jnd_drr(np.nanmedian(delta_drr_full)),
    "C50_full_JND_rating": evaluate_jnd_c50(np.nanmedian(delta_c50_full))

}
log_file.close()


for fc in nom_freq_pf:
    results[f"median_delta_T30_{int(fc)}Hz"] = np.nanmedian(delta_t30_oct[fc])
    results[f"median_delta_C50_{int(fc)}Hz"] = np.nanmedian(delta_c50_oct[fc])
    results[f"median_delta_DRR_{int(fc)}Hz"] = np.nanmedian(delta_drr_oct[fc])

df = pd.DataFrame([results])

csv_name = f"evaluation_{model_name}_{dataset}_{date_str}.csv"
csv_path = os.path.join(csv_dir, csv_name)
df.to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")


# Special IRs from RIR2FDN for individual evaluation
if len(special_results) > 0:

    log_path = os.path.join(csv_dir, f"special_irs_evaluation_results_{model_name}_{date_str}.txt")
    log_file = open(log_path, "w")

    log_print("\n==============================")
    log_print("SPECIAL IR RESULTS")
    log_print("==============================")

    log_print(f"Model: {exp}")
    log_print(f"Epoch: {epoch}")

    for row in special_results:

        log_print(f"\n--- {row['file']} ---")

        log_print(f"Match Loss: {row['match_loss']:.6f}")
        log_print(f"EDR Error: {row['edr_error']:.6f}")

        log_print("\nFULLBAND RESULTS")
        log_print(f"ΔT30: {row['delta_T30_full']:.4f} s")
        log_print(f"ΔC50: {row['delta_C50_full']:.4f} dB")
        log_print(f"ΔDRR: {row['delta_DRR_full']:.4f} dB")

        log_print("\nPERCEPTUAL INTERPRETATION")

        rel = row["delta_T30_rel_percent"]

        if not np.isnan(rel):
            log_print(f"Relative ΔT30: {rel:.2f} %")
            log_print(f"T30 rating: {evaluate_jnd_t30(rel)}")

        log_print(f"C50 rating: {evaluate_jnd_c50(row['delta_C50_full'])}")

        log_print("\nOCTAVE BAND RESULTS")

        for fc in nom_freq_pf:

            t30 = row[f"delta_T30_{int(fc)}Hz"]
            c50 = row[f"delta_C50_{int(fc)}Hz"]
            drr = row[f"delta_DRR_{int(fc)}Hz"]

            log_print(f"{fc} Hz:")
            log_print(f"  ΔT30: {t30:.4f} s")
            log_print(f"  ΔC50: {c50:.4f} dB")
            log_print(f"  ΔDRR: {drr:.4f} dB")

        log_print("\nOCTAVE AVERAGE")

        log_print(f"ΔT30 (Oct Avg): {row['delta_T30_oct_avg']:.4f} s")
        log_print(f"ΔC50 (Oct Avg): {row['delta_C50_oct_avg']:.4f} dB")
        log_print(f"ΔDRR (Oct Avg): {row['delta_DRR_oct_avg']:.4f} dB")


    log_file.close()



    df_special = pd.DataFrame(special_results)

    special_csv = os.path.join(
        csv_dir,
        f"evaluation_special_IRs_{model_name}_{date_str}.csv"
    )

    df_special.to_csv(special_csv, index=False)

    print("\nSpecial IR results saved to:", special_csv)


