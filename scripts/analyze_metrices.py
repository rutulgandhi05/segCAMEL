#!/usr/bin/env python3
# make_results_plots.py
# Reads metrics CSVs for A1/A2/A3 (or more), aggregates, and writes plots + LaTeX table.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================== EDIT THESE PATHS ======================
RUNS = {
    # label shown in plots/table        # path to your metrics CSV (last row used)
    "A1 (RI, speed OFF)": r"data\22092025_0511_segcamel_train_output_epoch_50_ri\20250924_1150_unsup_outputs_river_island_01_Day_ri\metrics_k10_a1.csv",
    "A2 (RVI, speed OFF)": r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250924_0614_unsup_outputs_river_island_01_Day_rvi_nspd\metrics_k10_a2.csv",
    "A3 (RVI, speed ON)":  r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250924_2100_unsup_outputs_river_island_01_Day_rvi\metrics_k10_a3.csv",
}
OUTDIR = Path("data/results_figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# If you want fixed colors for each run (optional), uncomment & edit:
RUN_COLORS = {
    "A1 (RI, speed OFF)": "#7f7f7f",  # gray
    "A2 (RVI, speed OFF)": "#1f77b4", # blue
    "A3 (RVI, speed ON)":  "#ff7f0e", # orange
}
USE_CUSTOM_COLORS = True  # set False to let matplotlib choose

# Columns your pipeline writes (script is robust if some are missing)
COLS = [
    "N","K","CH","DBI","SIL",
    "NMI_speedbins",
    "velvar_wmean",
    "temporal_consistency",
    "F1_dyn@0.30mps","F1_dyn@0.50mps","F1_dyn@1.00mps"
]

F1_THRESHOLDS = [0.30, 0.50, 1.00]

# ====================== HELPERS ======================
def load_last_row(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    row = df.tail(1).reset_index(drop=True)
    keep = [c for c in COLS if c in row.columns]
    return row[keep]

def agg_runs(run_map: dict) -> pd.DataFrame:
    records = []
    for name, path in run_map.items():
        r = load_last_row(path)
        r.insert(0, "run", name)
        r.insert(1, "csv_path", path)
        records.append(r)
    df = pd.concat(records, ignore_index=True)
    # Normalisations (higher is better)
    if "CH" in df:
        df["CH_norm"] = df["CH"] / df["CH"].max()
    if "DBI" in df:
        inv_dbi = 1.0 / df["DBI"]
        df["DBI_inv_norm"] = inv_dbi / inv_dbi.max()
    if "velvar_wmean" in df:
        inv_vv = 1.0 / df["velvar_wmean"]
        df["velvar_inv_norm"] = inv_vv / inv_vv.max()
    return df

def save_csv_and_table(df: pd.DataFrame, outdir: Path):
    df.to_csv(outdir/"results_aggregated.csv", index=False)

    # LaTeX table (compact and safe if some cols missing)
    def fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    cols_for_table = [
        ("SIL","Silhouette"),
        ("CH","CH"),
        ("DBI","DBI$\\downarrow$"),
        ("NMI_speedbins","NMI$_{\\text{speed}}$"),
        ("velvar_wmean","velvar"),
        ("temporal_consistency","Temporal"),
        ("F1_dyn@0.30mps","F1@0.3"),
        ("F1_dyn@0.50mps","F1@0.5"),
        ("F1_dyn@1.00mps","F1@1.0"),
    ]
    # build table body
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Unsupervised segmentation on \\textit{river\\_island\\_01\\_Day} (vMF, $K{=}10$). Higher is better for Silhouette, CH, F1, NMI; lower is better for DBI and velvar.}")
    header = "Run " + " & ".join([h for _,h in cols_for_table]) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c"*len(cols_for_table) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for c,_h in cols_for_table:
            if c in row and pd.notnull(row[c]):
                if c in ("CH",):  # big numbers: show with thousands sep (no decimals)
                    cells.append(f"{int(round(row[c])):,}".replace(",", "\\,"))
                else:
                    cells.append(fmt(row[c], 4 if "F1" in c else 3))
            else:
                cells.append("--")
        lines.append(f"{row['run']} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:results_river_island}")
    lines.append("\\end{table}")

    (outdir/"table_results.tex").write_text("\n".join(lines), encoding="utf-8")

# ====================== PLOTTING ======================
def maybe_color(run_name: str):
    if USE_CUSTOM_COLORS and run_name in RUN_COLORS:
        return dict(color=RUN_COLORS[run_name])
    return {}

def fig_intrinsic(df: pd.DataFrame, outdir: Path):
    # Build a tidy frame for plotting
    runs = list(df["run"])
    x = np.arange(len(runs), dtype=float)
    barw = 0.25

    plt.figure(figsize=(8,4))
    # CH (norm)
    if "CH_norm" in df:
        plt.bar(x - barw, df["CH_norm"].values, width=barw, label="CH (norm)")
    # Silhouette
    if "SIL" in df:
        plt.bar(x, df["SIL"].values, width=barw, label="Silhouette")
    # 1/DBI (norm)
    if "DBI_inv_norm" in df:
        plt.bar(x + barw, df["DBI_inv_norm"].values, width=barw, label="1/DBI (norm)")

    plt.xticks(x, runs, rotation=20, ha="right")
    plt.ylabel("Score (↑)")
    plt.title("Intrinsic clustering quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"fig_intrinsic.png", dpi=220)
    plt.savefig(outdir/"fig_intrinsic.pdf")
    plt.close()

def fig_motion_f1(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(8,4))
    ths = F1_THRESHOLDS
    for _, row in df.iterrows():
        y = []
        for t in ths:
            col = f"F1_dyn@{t:.2f}mps"
            y.append(row[col] if col in df.columns else np.nan)
        plt.plot(ths, y, marker="o", **maybe_color(row["run"]), label=row["run"])
    plt.xlabel("|v| threshold (m/s)")
    plt.ylabel("F1 dynamic/static (↑)")
    plt.title("Dynamic/static consistency")
    plt.xticks(ths, [str(t) for t in ths])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"fig_motion_f1.png", dpi=220)
    plt.savefig(outdir/"fig_motion_f1.pdf")
    plt.close()

def fig_motion_bars(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(8,4))
    runs = list(df["run"])
    x = np.arange(len(runs), dtype=float)
    barw = 0.35

    left = x - barw/2
    right = x + barw/2

    if "NMI_speedbins" in df:
        plt.bar(left, df["NMI_speedbins"].values, width=barw, label="NMI(|v| bins)")
    if "velvar_inv_norm" in df:
        plt.bar(right, df["velvar_inv_norm"].values, width=barw, label="1/velvar (norm)")
    plt.xticks(x, runs, rotation=20, ha="right")
    plt.ylabel("Score (↑)")
    plt.title("Motion alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"fig_motion_bars.png", dpi=220)
    plt.savefig(outdir/"fig_motion_bars.pdf")
    plt.close()

def fig_temporal(df: pd.DataFrame, outdir: Path):
    if "temporal_consistency" not in df:
        return
    plt.figure(figsize=(6,4))
    plt.bar(df["run"], df["temporal_consistency"])
    plt.ylabel("Temporal consistency (↑)")
    plt.title("Temporal stability")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outdir/"fig_temporal.png", dpi=220)
    plt.savefig(outdir/"fig_temporal.pdf")
    plt.close()

# ====================== MAIN ======================
def main():
    # Validate inputs
    for k, p in RUNS.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing CSV for '{k}': {p}")

    df = agg_runs(RUNS)
    save_csv_and_table(df, OUTDIR)

    # Make figures
    fig_intrinsic(df, OUTDIR)
    fig_motion_f1(df, OUTDIR)
    fig_motion_bars(df, OUTDIR)
    fig_temporal(df, OUTDIR)

    print(f"[OK] Wrote: {OUTDIR.resolve()}")
    print(" - results_aggregated.csv")
    print(" - table_results.tex")
    print(" - fig_intrinsic.(png|pdf)")
    print(" - fig_motion_f1.(png|pdf)")
    print(" - fig_motion_bars.(png|pdf)")
    print(" - fig_temporal.(png|pdf)")

if __name__ == "__main__":
    main()
