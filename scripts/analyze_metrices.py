#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ====== EDIT THIS MAPPING TO YOUR runs ======
RUNS = {
    "A1_RI_speedOFF":   r"data\22092025_0511_segcamel_train_output_epoch_50_ri\20250924_1150_unsup_outputs_river_island_01_Day_ri\metrics_k10.csv",
    "A2_RVI_speedOFF":  r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250924_0614_unsup_outputs_river_island_01_Day_rvi_nspd\metrics_k10.csv",
    "A3_RVI_speedON":   r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250924_0437_unsup_outputs_river_island_01_Day_rvi\metrics_k10.csv",
    # Optional extras:
    # "R3b_strict_deadzone": "/path/to/R3b/metrics.csv",
    # "K15": "/path/to/K15/metrics.csv",
    # "smooth2": "/path/to/smooth2/metrics.csv",
}
OUTDIR = Path(r"data\New folder")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ====== Columns your pipeline writes ======
COLS = [
    "N","K","CH","DBI","SIL",
    "NMI_speedbins",
    "velvar_wmean",
    "temporal_consistency",
    "F1_dyn@0.30mps","F1_dyn@0.50mps","F1_dyn@1.00mps"
]

def load_last_row(csv_path):
    df = pd.read_csv(csv_path)
    # if the CSV contains per-batch rows, take the last one (or mean over unique run)
    row = df.tail(1).reset_index(drop=True)
    # keep only known columns if present
    keep = [c for c in COLS if c in row.columns]
    return row[keep]

# ---- load all runs ----
records = []
for name, path in RUNS.items():
    r = load_last_row(path)
    r.insert(0, "run", name)
    records.append(r)

df = pd.concat(records, ignore_index=True)
df.to_csv(OUTDIR/"results_aggregated.csv", index=False)

# ---- normalisations for plotting (higher is better) ----
df["CH_norm"] = df["CH"] / df["CH"].max()
df["DBI_inv_norm"] = (1.0/df["DBI"]) / (1.0/df["DBI"]).max()
# Optional: invert velocity variance for “higher better”
df["velvar_inv_norm"] = (1.0/df["velvar_wmean"]) / (1.0/df["velvar_wmean"]).max()

# ------------------ FIG 1: Intrinsic ------------------
plt.figure(figsize=(8,4))
x = range(len(df))
barw = 0.25

plt.bar([i- barw for i in x], df["CH_norm"], width=barw, label="CH (norm)")
plt.bar(x,                    df["SIL"],     width=barw, label="Silhouette")
plt.bar([i+ barw for i in x], df["DBI_inv_norm"], width=barw, label="1/DBI (norm)")

plt.xticks(list(x), df["run"], rotation=20, ha="right")
plt.ylabel("Score (↑)")
plt.title("Intrinsic clustering quality")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"fig_intrinsic.png", dpi=200)

# ------------------ FIG 2: Motion ------------------
plt.figure(figsize=(8,4))
# F1 curves vs threshold
ths = [0.30, 0.50, 1.00]
for _, row in df.iterrows():
    y = [row.get(f"F1_dyn@{t:.2f}mps") for t in ths]
    plt.plot(ths, y, marker="o", label=row["run"])
plt.xlabel("|v| threshold (m/s)")
plt.ylabel("F1 dynamic/static (↑)")
plt.title("Dynamic/static consistency")
plt.xticks(ths)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"fig_motion_f1.png", dpi=200)

# Bars for NMI and (inverted) vel variance
plt.figure(figsize=(8,4))
barw = 0.35
x = range(len(df))
plt.bar([i- barw/2 for i in x], df["NMI_speedbins"], width=barw, label="NMI(|v| bins)")
plt.bar([i+ barw/2 for i in x], df["velvar_inv_norm"], width=barw, label="1/velvar (norm)")
plt.xticks(list(x), df["run"], rotation=20, ha="right")
plt.ylabel("Score (↑)")
plt.title("Motion alignment")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"fig_motion.png", dpi=200)

# ------------------ FIG 3: Temporal ------------------
plt.figure(figsize=(6,4))
plt.bar(df["run"], df["temporal_consistency"])
plt.ylabel("Temporal consistency (↑)")
plt.title("Temporal stability across frames")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTDIR/"fig_temporal.png", dpi=200)

print(f"Saved plots to: {OUTDIR.resolve()}")
