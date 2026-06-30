import glob, os, re, warnings
from datetime import date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, "soils-data")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVAL_S = 900            # 15-min logging
START_DATE = "2026-06-17"   # e.g. "2025-10-01" to trim; None = use all
MIN_FRAC_PER_DAY = 0.8      # keep daily means only when >=80% of expected readings present
DEFAULT_COEF = (0.0, 0.00016119, -0.10995650)   # FA, FB, FC used when a file has no header
STAMP = date.today().isoformat()

POS_COLOR = {"Control": "#F2B705", "North": "#1CA9C9", "South": "#A11627"}
DEPTH_STYLE = {"Shallow": "-", "Deep": "--"}
DEPTH_LW = {"Shallow": 1.9, "Deep": 1.4}
POS_ORDER = ["Control", "North", "South"]; DEPTH_ORDER = ["Shallow", "Deep"]


def classify(name):
    pos = "Control" if "control" in name.lower() else "North" if "north" in name.lower() else \
          "South" if "south" in name.lower() else "?"
    depth = "Deep" if "deep" in name.lower() else "Shallow" if "shallow" in name.lower() else "?"
    return pos, depth


def parse_tms(path):
    text = open(path, encoding="utf-8", errors="replace").read().splitlines()
    coef = DEFAULT_COEF
    if text and text[0].startswith("FA="):
        m = dict(re.findall(r"(F[ABC])\s*=\s*(-?\d+\.\d+)", text[0]))
        coef = (float(m["FA"]), float(m["FB"]), float(m["FC"]))
        text = text[1:]
    serial, recs = None, []
    for line in text:
        if not line.strip():
            continue
        p = line.split(";")
        if len(p) >= 9 and re.fullmatch(r"\d{6,}", p[1]):   # layout (a): serial in col 1
            serial = p[1]; dt, t1, t2, t3, mo = p[2], p[4], p[5], p[6], p[7]
        else:                                               # layout (b)
            dt, t1, t2, t3, mo = p[1], p[3], p[4], p[5], p[6]
        recs.append((dt, t1, t2, t3, mo))
    df = pd.DataFrame(recs, columns=["dt", "T1", "T2", "T3", "moist"])
    df["dt"] = pd.to_datetime(df["dt"], format="%Y.%m.%d %H:%M", errors="coerce")
    for c in ["T1", "T2", "T3", "moist"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt")
    fa, fb, fc = coef
    df["VWC"] = np.clip(fa * df["moist"] ** 2 + fb * df["moist"] + fc, 0, None)
    if START_DATE:
        df = df[df["dt"] >= pd.Timestamp(START_DATE)]
    return df, serial, coef

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No .csv files in {os.path.abspath(INPUT_DIR)!r}. Set INPUT_DIR.")

L = {}   # key -> dict(df, pos, depth, serial)
for f in files:
    pos, depth = classify(os.path.basename(f))
    df, serial, coef = parse_tms(f)
    L[(pos, depth)] = {"df": df, "pos": pos, "depth": depth, "serial": serial,
                       "name": os.path.basename(f)}

keys = [(p, d) for p in POS_ORDER for d in DEPTH_ORDER if (p, d) in L]
def style(k): return dict(color=POS_COLOR[k[0]], ls=DEPTH_STYLE[k[1]], lw=DEPTH_LW[k[1]])
def lab(k): return f"{k[0]} {k[1]}"

exp_per_day = 86400 / INTERVAL_S


def daily(df, col):
    g = df.set_index("dt")[col].resample("D")
    out = g.mean()
    return out[g.count() >= MIN_FRAC_PER_DAY * exp_per_day]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.edgecolor": "#444",
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 200, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def legend_outside(fig, x=0.86, top=0.92):
    h = [plt.Line2D([], [], **style(k)) for k in keys]
    leg = fig.legend(h, [lab(k) for k in keys], loc="upper left",
                     bbox_to_anchor=(x, top), fontsize=9, title="Position · depth",
                     title_fontsize=9, frameon=False)
    leg._legend_box.align = "left"


# Fig 1: Temperatures
fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, (col, title) in zip(axs, [("T1", "Soil temperature (T1, ≈ -6 cm)"),
                                   ("T2", "Surface temperature (T2, 0 cm)"),
                                   ("T3", "Air temperature (T3, ≈ +12 cm)")]):
    for k in keys:
        s = daily(L[k]["df"], col)
        ax.plot(s.index, s.values, **style(k))
    ax.set_title(title); ax.set_ylabel("°C"); month_axis(ax)
fig.suptitle("Daily Mean Temperatures", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.85, 0.96]); legend_outside(fig)
fig.savefig(f"{OUTPUT_DIR}/Soils-Temp-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# Fig 2: Soil Moisture
fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
ax = axs[0]
for k in keys:
    s = daily(L[k]["df"], "moist")
    ax.plot(s.index, s.values, **style(k))
ax.set_title("Daily Mean Raw Moisture Count"); ax.set_ylabel("TMS count")
fa, fb, fc = DEFAULT_COEF
if fb:
    thr = -fc / fb
    ax.axhline(thr, color="#888", ls=":", lw=1)
    ax.text(0.005, thr, f" VWC > 0 above ~{thr:.0f} counts", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color="#555")
month_axis(ax)
ax = axs[1]
for k in keys:
    s = daily(L[k]["df"], "VWC")
    ax.plot(s.index, s.values, **style(k))
ax.set_title("Daily Mean Volumetric Water Content (after calibration + clipping)")
ax.set_ylabel("VWC (m³ m⁻³)"); month_axis(ax)
fig.suptitle("Soil Moisture", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.85, 0.96]); legend_outside(fig, top=0.92)
fig.savefig(f"{OUTPUT_DIR}/Soils-Moisture-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# Fig 3: Diurnal
def diurnal(df, col):
    g = df.copy(); g["h"] = g["dt"].dt.hour
    return g.groupby("h")[col].mean()

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (col, title) in zip(axs, [("T3", "Air (T3)"), ("T1", "Soil (T1)")]):
    for k in keys:
        m = diurnal(L[k]["df"], col)
        ax.plot(m.index, m.values, **style(k))
    ax.set_title("Mean Diurnal Cycle"); ax.set_xlabel("Hour of day")
    ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3))
axs[0].set_ylabel("°C")
fig.suptitle("Diurnal Temperature: Air swings, soil is buffered", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.86, 0.95]); legend_outside(fig, x=0.875, top=0.86)
fig.savefig(f"{OUTPUT_DIR}/Soils-Diurnal-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

print("\nSaved figures")