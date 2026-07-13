import glob, os, re, warnings
from datetime import date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, "hobo-data")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INTERVAL_S = 600; MIN_READINGS_PER_DAY = 140; START_DATE = "2026-06-27"; STAMP = date.today().isoformat()

TCOL="Temperature   (°C)"; RHCOL="RH   (%)"
PARCOL="Photosynthetically Active Radiation   (μmol/m²/s)"
VPDCOL="Vapor Pressure Deficit   (kPa)"; DTCOL="Date-Time (EDT)"

SITES = {
    "ELG": ["22396965","22396959","22385220","22385208"],          # SU, SE, SC, NU
    "Pasture Food Forest": ["22411232","22385226","22411231"],     # Control, N of tree, S of tree
}
LABELS = {
    "22396965":"SU", "22396959":"SE",
    "22385220":"SC",     "22385208":"NU",
    "22411232":"Control", "22385226":"North", "22411231":"South",
}
SITE_OF = {sn:site for site,ids in SITES.items() for sn in ids}
ORDER = SITES["ELG"] + SITES["Pasture Food Forest"]
COLORSEQ = {
    "ELG": ["#7DC9F0", "#1CA9C9", "#1F5FBF", "#0B1F6B"],
    "Pasture Food Forest": ["#F2B705", "#EA5A0B", "#A11627"],
}
colors = {}
for site, ids in SITES.items():
    for c, sn in zip(COLORSEQ[site], ids):
        colors[sn] = c

def read_serial(path):
    try:
        d = pd.read_excel(path, sheet_name="Details", header=None)
        for _, row in d.iterrows():
            cells = [str(x) for x in row.tolist()]
            for i, c in enumerate(cells):
                if c.strip() == "Serial Number" and i + 1 < len(cells) and cells[i + 1] != "nan":
                    return cells[i + 1].strip()
    except Exception:
        pass
    m = re.search(r"\d{8}", os.path.basename(path))
    return m.group(0) if m else None

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")))
if not files:
    raise SystemExit(f"No .xlsx files found in {os.path.abspath(INPUT_DIR)!r}.\n"
                     "Set INPUT_DIR to the folder that holds the HOBO exports.")

loggers = {}
for f in files:
    sn = read_serial(f)
    if sn is None:
        print(f"  ! could not determine a serial for {os.path.basename(f)} — skipping")
        continue
    df = pd.read_excel(f, sheet_name="Data")
    df[DTCOL] = pd.to_datetime(df[DTCOL], errors="coerce")
    df = df.dropna(subset=[DTCOL]).sort_values(DTCOL)
    if START_DATE is not None:
        df = df[df[DTCOL] >= pd.Timestamp(START_DATE)]
    loggers[sn] = {"df": df, "full": TCOL in df.columns}

missing = [s for s in ORDER if s not in loggers]
if missing:
    raise SystemExit(
        "Loaded serials: " + (", ".join(sorted(loggers)) or "(none)") + "\n"
        "Expected but not found: " + ", ".join(missing) + "\n"
        "Check INPUT_DIR, or update the serials in the SITES dict to match your files.")

full_ids=[s for s in ORDER if loggers[s]["full"]]
all_ids=[s for s in ORDER]
def lab(sn): return LABELS.get(sn,sn)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","axes.labelsize":11,"axes.edgecolor":"#444","axes.linewidth":0.8,
    "figure.dpi":150,"savefig.dpi":200,"legend.frameon":False,"axes.grid":True,
    "grid.alpha":0.25,"grid.linewidth":0.6})

def fmt_time(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for t in ax.get_xticklabels(): t.set_rotation(0); t.set_ha("center")

def site_legend(fig, ids, x=0.855, top=0.90, fs=9):
    y = top
    for site in SITES:
        sids = [s for s in ids if SITE_OF[s] == site]
        if not sids:
            continue
        h = [plt.Line2D([], [], color=colors[s], lw=2.5) for s in sids]
        leg = fig.legend(h, [lab(s) for s in sids], title=site, loc="upper left",
                         bbox_to_anchor=(x, y), fontsize=fs, title_fontsize=fs, frameon=False)
        leg._legend_box.align = "left"
        y -= 0.075 * (len(sids) + 1.4)

fig,axs=plt.subplots(2,2,figsize=(14,8.5),sharex=True)
for ax,(title,unit,col,ids) in zip(axs.flat,
    [("Air Temperature","°C",TCOL,full_ids),
     ("Relative Humidity","%",RHCOL,full_ids),
     ("Vapor Pressure Deficit","kPa",VPDCOL,full_ids),
     ("Photosynthetically Active Radiation","μmol m⁻² s⁻¹",PARCOL,all_ids)]):
    lw=0.7 if col==PARCOL else 0.9
    for sn in ids:
        d=loggers[sn]["df"]; ax.plot(d[DTCOL],d[col],color=colors[sn],lw=lw,alpha=0.85)
    ax.set_title(title); ax.set_ylabel(unit); fmt_time(ax)
axs[0,1].set_ylim(top=102)
fig.suptitle("10-min Time Series (ELG vs Pasture Food Forest)", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 0.84, 0.96])
site_legend(fig, all_ids)
fig.savefig(f"{OUTPUT_DIR}/Microclimate-TimeSeries-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

def diurnal(df,col):
    g=df.copy(); g["h"]=g[DTCOL].dt.hour
    return g.groupby("h")[col].mean()
fig,axs=plt.subplots(2,2,figsize=(13,8))
for ax,(title,unit,col,ids) in zip(axs.flat,
    [("Air Temperature","°C",TCOL,full_ids),
     ("Relative Humidity","%",RHCOL,full_ids),
     ("Vapor Pressure Deficit","kPa",VPDCOL,full_ids),
     ("PAR","μmol m⁻² s⁻¹",PARCOL,all_ids)]):
    for sn in ids:
        m=diurnal(loggers[sn]["df"],col)
        ax.plot(m.index,m.values,color=colors[sn],lw=1.8,marker="o",ms=2.5)
    ax.set_title(f"{title}"); ax.set_ylabel(unit)
    ax.set_xlabel("Hour of day (EDT)"); ax.set_xlim(0,23); ax.set_xticks(range(0,24,3))
fig.suptitle("Mean Diurnal Cycles", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 0.84, 0.96])
site_legend(fig, all_ids)
fig.savefig(f"{OUTPUT_DIR}/Microclimate-Diurnal-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

rows=[]
for sn in all_ids:
    d=loggers[sn]["df"][[DTCOL,PARCOL]].dropna().copy(); d["date"]=d[DTCOL].dt.date
    cnt=d.groupby("date")[PARCOL].count(); dli=d.groupby("date")[PARCOL].sum()*INTERVAL_S/1e6
    for dt in cnt[cnt>=MIN_READINGS_PER_DAY].index: rows.append({"sn":sn,"date":dt,"dli":dli[dt]})
dli_df=pd.DataFrame(rows)
mean_dli=dli_df.groupby("sn")["dli"].agg(["mean","std"]).reindex(all_ids)

fig,axs=plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={"width_ratios":[1,1.3]})
ax=axs[0]; x=0; xticks=[]; xlabs=[]
for si,(site,ids) in enumerate(SITES.items()):
    for sn in ids:
        ax.bar(x,mean_dli["mean"][sn],yerr=mean_dli["std"][sn],color=colors[sn],
               edgecolor="#222",linewidth=0.7,capsize=4,alpha=0.95)
        ax.text(x,mean_dli["mean"][sn]+(mean_dli["std"][sn] or 0)+0.5,f'{mean_dli["mean"][sn]:.1f}',ha="center",fontsize=8)
        xticks.append(x); xlabs.append(lab(sn)); x+=1
    start=xticks[-len(ids)]; end=xticks[-1]
    ax.text((start+end)/2,-0.30,site,ha="center",va="top",fontsize=10,fontweight="bold",
            transform=ax.get_xaxis_transform())
    x+=0.8
ax.set_xticks(xticks); ax.set_xticklabels(xlabs,rotation=45,ha="right",fontsize=8.5)
ax.set_ylabel("Daily Light Integral (mol m⁻² day⁻¹)"); ax.set_title("Mean DLI by logger (± SD)")
ax.margins(x=0.02)

ax=axs[1]
for sn in all_ids:
    sub=dli_df[dli_df.sn==sn]
    ax.plot(pd.to_datetime(sub["date"]),sub["dli"],color=colors[sn],lw=1.6,marker="o",ms=4)
ax.set_ylabel("DLI (mol m⁻² day⁻¹)"); ax.set_title("Daily Light Integral over time")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.get_xticklabels(),rotation=0)
fig.suptitle("Daily Light Integral (complete days only)", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 0.87, 0.95])
site_legend(fig, all_ids, x=0.88, top=0.86)
fig.savefig(f"{OUTPUT_DIR}/Microclimate-Light-{STAMP}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

for site,ids in SITES.items():
    print(f"\n--{site}--")
    for sn in ids:
        d=loggers[sn]["df"]; extra=""
        if loggers[sn]["full"]:
            extra=f' | meanT={d[TCOL].mean():.1f} maxT={d[TCOL].max():.1f} meanRH={d[RHCOL].mean():.0f} meanVPD={d[VPDCOL].mean():.2f}'
        print(f'  {lab(sn):16} DLI={mean_dli["mean"][sn]:5.1f}{extra}')
print("\nSaved figures.")