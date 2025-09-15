# scripts/analyze_metrices.py  — grouped plots + percent deltas (no args)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- EDIT THESE TWO PATHS ----
CSV_OFF = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0532_unsup_outputs_parking_lot_02_Day_nospd\metrics_k10.csv")
CSV_ON  = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0539_unsup_outputs_parking_lot_02_Day_spd\metrics_k10.csv")

META_COLS = {"frame","stem","seq","path","timestamp","mode","K","notes"}
# metric groups used for separate figures
GROUPS = {
    "intrinsic": ["silhouette","dbi","ch","calinski","davies"],
    "velocity":  ["nmi_speedbins","f1_dyn","velvar"],
    # add more groups if needed, e.g. "coverage","noise","stability"
}

def read_any_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        for sep in [";", "\t", "|"]:
            try:
                t = pd.read_csv(path, sep=sep, engine="python")
                if t.shape[1] > 1:
                    df = t; break
            except Exception: pass
    print(f"[read] {path.name}: rows={len(df)} cols={len(df.columns)} | cols={list(df.columns)}")
    return df

def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.lower() for c in df.columns]
    if len(df.columns) == 2 and any(c in ("value","score","metric_value") for c in cols):
        value_col = df.columns[cols.index("value")] if "value" in cols \
            else (df.columns[cols.index("metric_value")] if "metric_value" in cols else df.columns[cols.index("score")])
        name_col  = [c for c in df.columns if c != value_col][0]
        tmp = df[[name_col, value_col]].copy()
        tmp.columns = ["metric","value"]
        w = tmp.groupby("metric", as_index=True)["value"].mean().to_frame().T
        for c in w.columns: w[c] = pd.to_numeric(w[c], errors="coerce")
        return w
    return pd.DataFrame()

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in META_COLS and not c.lower().startswith("unnamed")]
    df = df[keep].copy()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.loc[:, df.notna().any(0)]

def normalize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = long_to_wide(df)
    return wide if not wide.empty else select_numeric(df)

def pick_group(df: pd.DataFrame, group_keys) -> pd.DataFrame:
    cols = [c for c in df.columns if any(k in c.lower() for k in group_keys)]
    return df[cols].copy()

def bar_mean_std(mu_off, mu_on, sd_off, sd_on, title, out):
    keys = list(mu_off.index)
    if not keys: return
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(keys)*0.9), 4.8))
    ax.bar(x - w/2, mu_off.values, width=w, label="Velocity OFF")
    ax.bar(x + w/2, mu_on.values,  width=w, label="Velocity ON")
    ax.errorbar(x - w/2, mu_off.values, yerr=sd_off.values, fmt='none', capsize=3)
    ax.errorbar(x + w/2, mu_on.values,  yerr=sd_on.values,  fmt='none', capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_title(title); ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)
    print("Saved:", out)

def bar_delta_percent(mu_off, mu_on, title, out):
    keys = list(mu_off.index)
    if not keys: return
    off = mu_off.values
    on  = mu_on.values
    # percent change; guard divide-by-zero
    pct = np.where(np.abs(off) > 1e-12, (on - off) / np.abs(off) * 100.0, 0.0)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(8, len(keys)*0.9), 4.2))
    ax.bar(x, pct)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_title(title + "  (percent change)")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)
    print("Saved:", out)

def main():
    off_raw = read_any_csv(CSV_OFF)
    on_raw  = read_any_csv(CSV_ON)

    off = normalize_metrics_table(off_raw)
    on  = normalize_metrics_table(on_raw)

    # align columns
    common = [c for c in off.columns if c in on.columns]
    off, on = off[common].dropna(axis=1, how="all"), on[common].dropna(axis=1, how="all")

    if off.empty or on.empty:
        print("[ERROR] no common numeric metrics after normalization"); return

    outdir = Path("analysis_figs"); outdir.mkdir(exist_ok=True)

    # save raw means/std + deltas
    mu_off, sd_off = off.mean(0), off.std(0)
    mu_on,  sd_on  = on.mean(0),  on.std(0)
    delta          = mu_on - mu_off
    (outdir / "mean_off.csv").write_text(mu_off.to_csv(), encoding="utf-8")
    (outdir / "mean_on.csv").write_text(mu_on.to_csv(),  encoding="utf-8")
    (outdir / "delta_on_minus_off.csv").write_text(delta.to_csv(), encoding="utf-8")

    # UTF-8 summary
    summary = (
        "# Metrics summary\n\n"
        "## Means (OFF)\n" + mu_off.to_string() + "\n\n"
        "## Means (ON)\n"  + mu_on.to_string()  + "\n\n"
        "## Delta (ON - OFF)\n" + delta.to_string() + "\n"
    )
    (outdir / "summary.md").write_text(summary, encoding="utf-8")

    # groupwise plots
    for gname, keys in GROUPS.items():
        off_g = pick_group(off, keys)
        on_g  = pick_group(on,  keys)
        common_g = [c for c in off_g.columns if c in on_g.columns]
        if not common_g: 
            print(f"[WARN] no metrics for group '{gname}'"); 
            continue
        off_g, on_g = off_g[common_g], on_g[common_g]
        mu_off_g, sd_off_g = off_g.mean(0), off_g.std(0)
        mu_on_g,  sd_on_g  = on_g.mean(0),  on_g.std(0)

        bar_mean_std(mu_off_g, mu_on_g, sd_off_g, sd_on_g,
                     f"Mean±std across frames — {gname}", outdir / f"metrics_mean_std_{gname}.png")
        bar_delta_percent(mu_off_g, mu_on_g,
                          f"Velocity impact (ON−OFF) — {gname}", outdir / f"metrics_delta_percent_{gname}.png")

if __name__ == "__main__":
    main()
