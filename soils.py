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
LOCAL_TZ = "America/New_York"   # diurnal cycle is shown in this zone (handles EDT/EST)
MIN_FRAC_PER_DAY = 0.8      # keep daily means only when >=80% of expected readings present
DEFAULT_COEF = (0.0, 0.00016119, -0.10995650)   # FA, FB, FC used when a file has no header
STAMP = date.today().isoformat()

DEPTH_STYLE = {"Shallow": "-", "Deep": "--"}
DEPTH_LW = {"Shallow": 1.9, "Deep": 1.4}
DEPTH_ORDER = ["Shallow", "Deep"]

SITES = {
    "Pasture": {
        "order": ["Control", "North", "South"],
        "color": {"Control": "#F2B705", "North": "#1CA9C9", "South": "#A11627"},
        "install": "2026-06-17",
    },
    "ELG": {
        "order": ["SU", "NU",
                  "SC", "SE"],
        "color": {"SU": "#1CA9C9",
                  "NU": "#1B7837",
                  "SC":     "#A11627",
                  "SE":       "#F2B705"},
        "install": "2026-06-27",
    },
}

# ELG files start with a two-letter position code (SU/NU/SC/SE).
ELG_CODES = {"su": "SU", "nu": "NU",
             "sc": "SC",     "se": "SE"}

def classify(name):
    """Return (site, position, depth) from a file name."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    depth = "Deep" if "deep" in stem else "Shallow" if ("shallow" in stem or "sh" in stem) else "?"
    for code, pos in ELG_CODES.items():          # ELG codes are unambiguous prefixes
        if stem.startswith(code):
            return "ELG", pos, depth
    pos = ("Control" if "control" in stem else
           "North"   if "north"   in stem else
           "South"   if "south"   in stem else "?")
    return "Pasture", pos, depth

def parse_tms(path, start=None):
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
    if start:
        df = df[df["dt"] >= pd.Timestamp(start)]
    return df, serial, coef

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No .csv files in {os.path.abspath(INPUT_DIR)!r}. Set INPUT_DIR.")

data = {s: {} for s in SITES}            # site -> {(pos, depth): dict(df, ...)}
for f in files:
    site, pos, depth = classify(f)
    if site not in SITES or pos == "?" or depth == "?":
        print(f"  ! skipping unclassified file {os.path.basename(f)} -> ({site}, {pos}, {depth})")
        continue
    df, serial, coef = parse_tms(f, start=SITES[site]["install"])
    data[site][(pos, depth)] = {"df": df, "pos": pos, "depth": depth,
                                "serial": serial, "name": os.path.basename(f)}

def cadence_per_day(df):
    """Expected readings/day inferred from the logger's own median interval.
    Loggers here mix cadences (e.g. some drop to hourly at deployment), so a
    global 15-min assumption would wrongly discard the coarser records."""
    if len(df) < 3:
        return 86400 / INTERVAL_S
    med = df["dt"].diff().dt.total_seconds().median()
    return 86400 / med if med and med > 0 else 86400 / INTERVAL_S

def daily(df, col):
    exp = cadence_per_day(df)
    g = df.set_index("dt")[col].resample("D")
    out = g.mean()
    return out[g.count() >= MIN_FRAC_PER_DAY * exp]

def diurnal(df, col):
    g = df.copy()
    t = g["dt"]
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    g["h"] = t.dt.tz_convert(LOCAL_TZ).dt.hour   # UTC timestamps -> local hour
    return g.groupby("h")[col].mean()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.edgecolor": "#444",
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 200, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})

def date_axis(ax, span_days):
    """Pick a sensible x-axis density for the actual length of the record."""
    if span_days <= 21:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    elif span_days <= 120:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

def make_site_figures(site):
    cfg, L = SITES[site], data[site]
    keys = [(p, d) for p in cfg["order"] for d in DEPTH_ORDER if (p, d) in L]
    if not keys:
        print(f"  ! {site}: no matching files, skipping")
        return
    span = max((L[k]["df"]["dt"].max() - L[k]["df"]["dt"].min()).days for k in keys)

    def style(k): return dict(color=cfg["color"][k[0]], ls=DEPTH_STYLE[k[1]], lw=DEPTH_LW[k[1]])
    def lab(k):   return f"{k[0]} {k[1]}"
    def legend_outside(fig, x=0.86, top=0.92):
        h = [plt.Line2D([], [], **style(k)) for k in keys]
        leg = fig.legend(h, [lab(k) for k in keys], loc="upper left",
                         bbox_to_anchor=(x, top), fontsize=9, title="Position · depth",
                         title_fontsize=9, frameon=False)
        leg._legend_box.align = "left"

    # --- Temperatures -------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, (col, title) in zip(axs, [("T1", "Soil temperature (T1, ≈ -6 cm)"),
                                       ("T2", "Surface temperature (T2, 0 cm)"),
                                       ("T3", "Air temperature (T3, ≈ +12 cm)")]):
        for k in keys:
            s = daily(L[k]["df"], col)
            ax.plot(s.index, s.values, **style(k))
        ax.set_title(title); ax.set_ylabel("°C"); date_axis(ax, span)
    fig.suptitle(f"{site} — Daily Mean Temperatures", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96]); legend_outside(fig)
    fig.savefig(f"{OUTPUT_DIR}/{site}-Temp-{STAMP}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Moisture -----------------------------------------------------------
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
    date_axis(ax, span)
    ax = axs[1]
    for k in keys:
        s = daily(L[k]["df"], "VWC")
        ax.plot(s.index, s.values, **style(k))
    ax.set_title("Daily Mean Volumetric Water Content")
    ax.set_ylabel("VWC (m³ m⁻³)"); date_axis(ax, span)
    fig.suptitle(f"{site} — Soil Moisture", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96]); legend_outside(fig, top=0.92)
    fig.savefig(f"{OUTPUT_DIR}/{site}-Moisture-{STAMP}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Diurnal ------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (col, title) in zip(axs, [("T3", "Air temperature (T3, ≈ +12 cm)"),
                                      ("T1", "Soil temperature (T1, ≈ -6 cm)")]):
        for k in keys:
            m = diurnal(L[k]["df"], col)
            ax.plot(m.index, m.values, **style(k))
        ax.set_title(title); ax.set_xlabel("Hour of day (Eastern)")
        ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3))
    axs[0].set_ylabel("°C")
    fig.suptitle(f"{site} — Diurnal Temperature",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95]); legend_outside(fig, x=0.875, top=0.86)
    fig.savefig(f"{OUTPUT_DIR}/{site}-Diurnal-{STAMP}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

for site in SITES:
    n = len(data[site])
    print(f"{site}: {n} logger file(s)")
    make_site_figures(site)

print("\nSaved figures")