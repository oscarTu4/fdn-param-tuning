import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os

# CSV files
gru_shoebox_csv = "outputs/GRU 2103-1300/evaluation_results/evaluation_%_model_e16_shoebox_2026-03-28_14-59-59.csv"
conf_shoebox_csv = "outputs/Conf 2203-1000/evaluation_results/evaluation_%_model_e16_shoebox_2026-03-28_15-16-53.csv"

gru_mit_csv = "outputs/GRU 2103-1300/evaluation_results/evaluation_%_model_e16_MIT_2026-03-28_14-50-54.csv"
conf_mit_csv = "outputs/Conf 2203-1000/evaluation_results/evaluation_%_model_e16_MIT_2026-03-28_14-40-10.csv"

# CSV files for IRs from RIR2FDN
gru_special_csv = "outputs/GRU 2103-1300/evaluation_results/evaluation_special_IRs_model_e16_2026-03-23_13-47-03.csv"
conf_special_csv = "outputs/Conf 2203-1000/evaluation_results/evaluation_special_IRs_model_e16_2026-03-23_13-52-44.csv"

# Paths to get plots in both output folders
gru_out = "outputs/GRU 2103-1300/evaluation_results/plots"
os.makedirs(gru_out, exist_ok=True)
conf_out = "outputs/Conf 2203-1000/evaluation_results/plots"
os.makedirs(conf_out, exist_ok=True)

#gru = pd.read_csv(gru_csv)
#conf = pd.read_csv(conf_csv)

gru_shoebox = pd.read_csv(gru_shoebox_csv)
conf_shoebox = pd.read_csv(conf_shoebox_csv)
gru_mit = pd.read_csv(gru_mit_csv)
conf_mit = pd.read_csv(conf_mit_csv)

### Plots for Evaluation Dataset (MIT or Shoebox) ####

# Fullband
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

metrics_full = [
    ("median_delta_T30_full","ΔT30 (s)"),
    ("median_delta_C50_full","ΔC50 (dB)"),
    ("median_delta_DRR_full","ΔDRR (dB)")
]

colors = ["tab:blue", "tab:orange"]

for i, (col, label) in enumerate(metrics_full):

    ax = axes[i]

    values = [
        gru_shoebox[col][0],
        conf_shoebox[col][0],
        gru_mit[col][0],
        conf_mit[col][0],
    ]

    x = np.array([0, 1, 3, 4])  

    ax.bar(x, values, color=[colors[0], colors[1], colors[0], colors[1]])

    ax.set_xticks(x)
    ax.set_xticklabels(["GRU", "Conf", "GRU", "Conf"])

    ax.set_title(label)
    ax.set_ylabel(label)

    def add_bracket(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y*1.05, y*1.05, y], lw=1.5, color="black")
        ax.text((x1+x2)/2, y*1.08, text, ha='center', va='bottom')
    
    ymax = max(values)

    ax.set_ylim(0, ymax * 1.3)

    bracket_y = ymax * 1.05

    add_bracket(ax, 0, 1, bracket_y, "Shoebox")
    add_bracket(ax, 3, 4, bracket_y, "MIT")

    # ymax = max(values) * 1.2

    # add_bracket(ax, 0, 1, ymax, "Shoebox")
    # add_bracket(ax, 3, 4, ymax, "MIT")

plt.tight_layout()
filename = "evaluation_fullband_barplots.png"
plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)
plt.show()

# metrics_full = [
#     ("median_delta_T30_full","ΔT30 (s)"),
#     ("median_delta_C50_full","ΔC50 (dB)"),
#     ("median_delta_DRR_full","ΔDRR (dB)")
# ]

# fig, axes = plt.subplots(1,3, figsize=(10,4))

# colors = ["tab:blue", "tab:orange"] 

# for i,(col,label) in enumerate(metrics_full):

#     ax = axes[i]

#     values = [
#         gru[col][0],
#         conf[col][0]
#     ]

#     ax.bar(["GRU","Conformer"], values, color=colors)

#     ax.set_title(label)
#     ax.set_ylabel(label)

#     # # JND Linien 
#     # if "T30" in label:
#     #     ax.axhline(0.05, color="red", linestyle="--", label="JND")

#     # if "C50" in label:
#     #     ax.axhline(1, color="red", linestyle="--")

#     # if "DRR" in label:
#     #     ax.axhline(3, color="red", linestyle="--")

# plt.tight_layout()
# filename = "evaluation_fullband_barplots.png"
# plt.savefig(os.path.join(gru_out, filename), dpi=300)
# plt.savefig(os.path.join(conf_out, filename), dpi=300)
# plt.show()

# Oktavbands
freqs = [125,250,500,1000,2000,4000,8000]

metrics = [
    ("delta_T30","ΔT30 (s)"),
    ("delta_C50","ΔC50 (dB)"),
    ("delta_DRR","ΔDRR (dB)")
]


fig, axes = plt.subplots(1,3, figsize=(14,4))

bar_width = 0.35
x = np.arange(len(freqs))

for i,(metric,label) in enumerate(metrics):

    ax = axes[i]

    gru_vals = [gru_shoebox[f"median_{metric}_{f}Hz"][0] for f in freqs]
    conf_vals = [conf_shoebox[f"median_{metric}_{f}Hz"][0] for f in freqs]

    ax.bar(x-bar_width/2, gru_vals, bar_width, label="GRU")
    ax.bar(x+bar_width/2, conf_vals, bar_width, label="Conformer")

    ax.set_xticks(x)
    ax.set_xticklabels(freqs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(label)
    ax.set_title(label)

    if i == 0:
        ax.legend()

plt.tight_layout()
filename = "evaluation_shoebox_octave_band_barplots.png"
plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)
plt.show()

fig, axes = plt.subplots(1,3, figsize=(14,4))

bar_width = 0.35
x = np.arange(len(freqs))

for i,(metric,label) in enumerate(metrics):

    ax = axes[i]

    gru_vals = [gru_mit[f"median_{metric}_{f}Hz"][0] for f in freqs]
    conf_vals = [conf_mit[f"median_{metric}_{f}Hz"][0] for f in freqs]

    ax.bar(x-bar_width/2, gru_vals, bar_width, label="GRU")
    ax.bar(x+bar_width/2, conf_vals, bar_width, label="Conformer")

    ax.set_xticks(x)
    ax.set_xticklabels(freqs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(label)
    ax.set_title(label)

    if i == 0:
        ax.legend()

plt.tight_layout()
filename = "evaluation_mit_octave_band_barplots.png"
plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)
plt.show()


# ### Plots for IRs from RIR2FDN ###

gru_special = pd.read_csv(gru_special_csv)
conf_special = pd.read_csv(conf_special_csv)

rooms = {
    "Lobby":"Lobby",
    "Hallway":"Hallway",
    "Meeting":"MeetingRoom"
}

metrics = [
    ("delta_T30_full","ΔT30 (s)"),
    ("delta_C50_full","ΔC50 (dB)"),
    ("delta_DRR_full","ΔDRR (dB)")
]


fig, axes = plt.subplots(3,3, figsize=(12,10))

for col,(metric,label) in enumerate(metrics):

    # global ymax 
    ymax = max(
        gru_special[metric].max(),
        conf_special[metric].max()
    ) * 1.1   

    for row,(room_title,key) in enumerate(rooms.items()):

        row_gru = gru_special[gru_special["file"].str.contains(key)].iloc[0]
        row_conf = conf_special[conf_special["file"].str.contains(key)].iloc[0]

        ax = axes[row,col]

        values = [
            row_gru[metric],
            row_conf[metric]
        ]

        ax.bar(["GRU","Conformer"], values, color=colors)

        ax.set_ylim(0, ymax)

        ax.set_title(f"{room_title} — {label}")
        ax.set_ylabel(label)

plt.tight_layout()
filename = "special_IR_fullband_barplots.png"
plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)
plt.show()


freqs = [125,250,500,1000,2000,4000,8000]

metrics = [
    ("delta_T30","ΔT30 (s)"),
    ("delta_C50","ΔC50 (dB)"),
    ("delta_DRR","ΔDRR (dB)")
]

fig, axes = plt.subplots(3,3, figsize=(14,10), sharey='col')

bar_width = 0.35
x = np.arange(len(freqs))

for row,(room_title,key) in enumerate(rooms.items()):

    row_gru = gru_special[gru_special["file"].str.contains(key)].iloc[0]
    row_conf = conf_special[conf_special["file"].str.contains(key)].iloc[0]

    for col,(metric,label) in enumerate(metrics):

        ax = axes[row,col]

        gru_vals = [row_gru[f"{metric}_{f}Hz"] for f in freqs]
        conf_vals = [row_conf[f"{metric}_{f}Hz"] for f in freqs]

        ax.bar(x-bar_width/2, gru_vals, bar_width, label="GRU", color="tab:blue")
        ax.bar(x+bar_width/2, conf_vals, bar_width, label="Conformer", color="tab:orange")

        ax.set_xticks(x)
        ax.set_xticklabels(freqs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(label)

        ax.set_title(f"{room_title} — {label}")

        if row == 0 and col == 0:
            ax.legend()

plt.tight_layout()
filename = "special_IR_octave_barplots.png"
plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)

plt.show()


metrics = [
    ("T30","T30 (s)"),
    ("C50","C50 (dB)"),
    ("DRR","DRR (dB)")
]

fig, axes = plt.subplots(3,3, figsize=(14,10), sharey='col')

for row,(room_title,key) in enumerate(rooms.items()):

    row_gru = gru_special[gru_special["file"].str.contains(key)].iloc[0]
    row_conf = conf_special[conf_special["file"].str.contains(key)].iloc[0]

    for col,(metric,label) in enumerate(metrics):

        ax = axes[row,col]

        ref_vals = [row_gru[f"{metric}_ref_{f}Hz"] for f in freqs]
        gru_vals = [row_gru[f"{metric}_pred_{f}Hz"] for f in freqs]
        conf_vals = [row_conf[f"{metric}_pred_{f}Hz"] for f in freqs]

        ax.plot(freqs, ref_vals, marker="o", linewidth=2, label="Reference", color="black")
        ax.plot(freqs, gru_vals, marker="o", linestyle="--", label="GRU", color="tab:blue")
        ax.plot(freqs, conf_vals, marker="o", linestyle="--", label="Conformer", color="tab:orange")

        ax.set_xscale("log")
        ax.set_xticks(freqs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_title(f"{room_title} — {label}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(label)

        if row == 0 and col == 0:
            ax.legend()

plt.tight_layout()

filename = "special_IR_octave_reference_comparison.png"

plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)

plt.show() 
#### PLOTS WITH PERCENTAGES FOR T30 (the ones we decided to use in the paper) ####
### Plot with T30 as percentages 


def add_jnd_t30(ax):
    ax.axhspan(5, 25, color="red", alpha=0.08)

def add_jnd_c50(ax):
    ax.axhline(1, linestyle="--", color="red")

fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))

colors = ["tab:blue", "tab:orange"]

# Order: GRU Shoebox, Conf Shoebox, GRU MIT, Conf MIT
# Values are from txt files
t30_percent_values = [
    19.33,  # GRU Shoebox
    10.14,  # Conf Shoebox
    137.30,  # GRU MIT
    70.67   # Conf MIT
]


x = np.array([0, 1, 3, 4])

def add_bracket(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y*1.05, y*1.05, y],
            lw=1.5, color="black")
    ax.text((x1+x2)/2, y*1.08, text,
            ha='center', va='bottom', color="black")


# T30 (%)
ax = axes[0]

ax.bar(x, t30_percent_values,
       color=[colors[0], colors[1], colors[0], colors[1]])

ax.set_xticks(x)
ax.set_xticklabels(["GRU", "Conf", "GRU", "Conf"])

ax.set_title("ΔT30 (%)")
ax.set_ylabel("ΔT30 (%)")

# JND range (5–25%)
ax.axhspan(5, 25, alpha=0.1, color= "red", label="JND")

ymax = max(t30_percent_values)
ax.set_ylim(0, ymax * 1.3)

add_bracket(ax, 0, 1, ymax * 1.05, "Shoebox")
add_bracket(ax, 3, 4, ymax * 1.05, "MIT")
ax.legend(loc="upper left", fontsize=8)

# C50 (dB)
ax = axes[1]

c50_values = [
    gru_shoebox["median_delta_C50_full"][0],
    conf_shoebox["median_delta_C50_full"][0],
    gru_mit["median_delta_C50_full"][0],
    conf_mit["median_delta_C50_full"][0],
]

ax.bar(x, c50_values,
       color=[colors[0], colors[1], colors[0], colors[1]])

ax.set_xticks(x)
ax.set_xticklabels(["GRU", "Conf", "GRU", "Conf"])

ax.set_title("ΔC50 (dB)")
ax.set_ylabel("ΔC50 (dB)")

# JND = 1 dB
ax.axhline(1, linestyle="--", color="red", label="JND")
ax.legend(loc="upper left", fontsize=8)

ymax = max(c50_values)
ax.set_ylim(0, ymax * 1.3)

add_bracket(ax, 0, 1, ymax * 1.05, "Shoebox")
add_bracket(ax, 3, 4, ymax * 1.05, "MIT")

# DRR (dB)

ax = axes[2]

drr_values = [
    gru_shoebox["median_delta_DRR_full"][0],
    conf_shoebox["median_delta_DRR_full"][0],
    gru_mit["median_delta_DRR_full"][0],
    conf_mit["median_delta_DRR_full"][0],
]

ax.bar(x, drr_values,
       color=[colors[0], colors[1], colors[0], colors[1]])

ax.set_xticks(x)
ax.set_xticklabels(["GRU", "Conf", "GRU", "Conf"])

ax.set_title("ΔDRR (dB)")
ax.set_ylabel("ΔDRR (dB)")


#ax.axhline(3, linestyle="--", color="black", label="JND")

ymax = max(drr_values)
ax.set_ylim(0, ymax * 1.3)

add_bracket(ax, 0, 1, ymax * 1.05, "Shoebox")
add_bracket(ax, 3, 4, ymax * 1.05, "MIT")


handles, labels = axes[0].get_legend_handles_labels()
for ax in axes[1:]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l

unique = dict(zip(labels, handles))

# fig.legend(unique.values(), unique.keys(),
#           loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.9])

filename = "evaluation_fullband_JND_barplots.png"

plt.savefig(os.path.join(gru_out, filename), dpi=300)
plt.savefig(os.path.join(conf_out, filename), dpi=300)

plt.show()

### Shoebox and  MIT Survey Percentage Plots Octave BAnds

datasets = [
    ("Shoebox", gru_shoebox, conf_shoebox),
    ("MIT", gru_mit, conf_mit)
]

metrics_percent = [
    ("delta_T30","ΔT30 (%)"),
    ("delta_C50","ΔC50 (dB)"),
    ("delta_DRR","ΔDRR (dB)")
]

freqs = [125,250,500,1000,2000,4000,8000]
bar_width = 0.35
x = np.arange(len(freqs))

colors = ["tab:blue", "tab:orange"]

for dataset_name, gru_df, conf_df in datasets:

    fig, axes = plt.subplots(1,3, figsize=(14,5))

    fig.suptitle(f"{dataset_name} Testset", fontsize=14)

    for i,(metric,label) in enumerate(metrics_percent):

        ax = axes[i]

        # T30 (%)

        if metric == "delta_T30":

            gru_vals = [gru_df[f"median_delta_T30_rel_{int(f)}Hz_percent"][0] for f in freqs]
            conf_vals = [conf_df[f"median_delta_T30_rel_{int(f)}Hz_percent"][0] for f in freqs]

            ax.bar(x-bar_width/2, gru_vals, bar_width, color=colors[0], label="GRU")
            ax.bar(x+bar_width/2, conf_vals, bar_width, color=colors[1], label="Conformer")

            # JND Area
            ax.axhspan(5, 25, color="red", alpha=0.08, label="JND range (5–25%)")

            ax.set_ylabel("ΔT30 (%)")

        
        # C50

        elif metric == "delta_C50":

            gru_vals = [gru_df[f"median_delta_C50_{int(f)}Hz"][0] for f in freqs]
            conf_vals = [conf_df[f"median_delta_C50_{int(f)}Hz"][0] for f in freqs]

            ax.bar(x-bar_width/2, gru_vals, bar_width, color=colors[0])
            ax.bar(x+bar_width/2, conf_vals, bar_width, color=colors[1])

            # JND Line
            ax.axhline(1, linestyle="--", color="red", label="JND (1 dB)")

            ax.set_ylabel("ΔC50 (dB)")


        # DRR
  
        elif metric == "delta_DRR":

            gru_vals = [gru_df[f"median_delta_DRR_{int(f)}Hz"][0] for f in freqs]
            conf_vals = [conf_df[f"median_delta_DRR_{int(f)}Hz"][0] for f in freqs]

            ax.bar(x-bar_width/2, gru_vals, bar_width, color=colors[0])
            ax.bar(x+bar_width/2, conf_vals, bar_width, color=colors[1])

            ax.set_ylabel("ΔDRR (dB)")

        ax.set_xticks(x)
        ax.set_xticklabels(freqs)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(label)

        if i == 0 and dataset_name == "Shoebox":
            ax.legend(loc="upper left", fontsize=8)
        if i == 1 and dataset_name == "Shoebox":
            ax.legend(loc="upper right", fontsize=8)
        if i == 0 and dataset_name == "MIT":
            ax.legend(loc="upper right", fontsize=8)
        if i == 1 and dataset_name == "MIT":
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    filename = f"evaluation_{dataset_name.lower()}_octave_band_barplots_percent.png"

    plt.savefig(os.path.join(gru_out, filename), dpi=300)
    plt.savefig(os.path.join(conf_out, filename), dpi=300)

    plt.show()