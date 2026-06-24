#!/usr/bin/env python3
"""
================================================================================
 Ground Station RF Cascade Analyzer  —  1534.5 MHz L-band FUSE downlink
================================================================================
 Tuned specifically to the File-Downlink profile (satellite telemetry @
 1534.5 MHz, FUSE -> PTU3). This is NOT the C-band model; the chain is
 different in two important ways:

   1. NO DOWN-CONVERSION.  1534.5 MHz already falls inside the Work Microwave
      modem's L-band input (950-2150 MHz), so the signal feeds the modem
      directly. The MX6000 mixer, the IF bandpass, and the IF amp are NOT in
      this path. (If your routing actually converts L-band to the 1000 MHz IF,
      re-insert a mixer stage: gain ~ -8 dB, NF ~ 8 dB, oip3 ~ +5 dBm.)

   2. BETTER LNA NUMBERS.  The ZX60-83LN12+ at 1.5 GHz is far better than its
      8 GHz worst case. From the datasheet typical-performance table @ 1500 MHz:
      Gain 22.46 dB, NF 1.39 dB, OIP3 +36.2 dBm, P1dB +20.9 dBm.

 L-band band-limiting is the 6LP8-1680 lowpass chain (cutoff 1.68 GHz, passes
 the 1525-1559 MHz FUSE downlink). Everything is referred to the feed-LNA input.

 CAVEATS for this band:
   * 1534.5 MHz sits squarely in the 1525-1559 MHz band the source flags for
     interference -> the REAL floor here is above the thermal value below.
   * For a GIVEN dish, gain at 1.5 GHz is ~10 dB lower than at 4.9 GHz
     (G ~ f^2), so G/T at L-band is much lower unless a larger/dedicated dish
     is used. G_RX_DBI below is a placeholder — set it to your L-band dish.

 HOW TO USE: edit STAGES + CONFIG, then `python cascade_1534MHz.py`
 Outputs: console report + cascade_1534MHz_dashboard.png + cascade_1534MHz.csv
 Requires: numpy, matplotlib
================================================================================
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============================================================================
# CONFIG  --  edit for your station / this band
# =============================================================================
FREQ_MHZ    = 1534.5    # band of interest (FUSE downlink)
T0_K        = 290.0     # IEEE reference temperature for NF (K)
T_ANT_K     = 80.0      # L-band sky/antenna temp (K) [PLACEHOLDER - higher than
                        #   C-band: galactic background is stronger at 1.5 GHz]
G_RX_DBI    = 25.0      # L-band receive dish gain (dBi) [PLACEHOLDER - a given
                        #   dish has ~10 dB less gain at 1.5 GHz than at 4.9 GHz]

RS_SYM      = 5e6       # telemetry symbol rate (sym/s) [PLACEHOLDER - set yours]
ROLLOFF     = 0.20      # RRC roll-off (occupied-BW reporting only)
IMPL_MARGIN = 1.0       # demod implementation margin (dB)

PIN_DBM     = -100.0    # example input signal level for the level diagram (dBm)
BW_LEVEL    = 5e6       # bandwidth for the level-diagram noise (Hz)
BW_SFDR     = 1e6       # bandwidth for the SFDR figure in the IP3 panel (Hz)

KTB0        = -174.0    # kT0 in dBm/Hz at 290 K
THEME       = "#1f3a5f"  # navy
ACCENT      = "#e07b39"  # orange
RED         = "#b23a48"

# =============================================================================
# RECEIVE CHAIN  --  1534.5 MHz L-band, direct-to-modem (no mixer)
#   NF of a passive stage == its loss. OIP3=36.2 dBm for the LNAs is the
#   datasheet value AT 1.5 GHz; passive OIP3 values are placeholders.
# =============================================================================
@dataclass
class Stage:
    name:     str
    short:    str
    gain_db:  float
    nf_db:    float
    oip3_dbm: float = 45.0       # placeholder default for passive parts
    oip3_known: bool = False     # True only where taken from a datasheet

STAGES = [
    Stage("Feed LNA (ZX60-83LN12+ @1.5 GHz)",       "Feed LNA", +22.5, 1.39, 36.2, True),
    Stage("Bias tee (ZX85-12G-S+)",                 "Bias T",    -0.4, 0.4,  45.0, False),
    Stage("Coax feed -> RF Box",                    "Coax",      -1.7, 1.7,  45.0, False),
    Stage("RF Box input switch (SW264K)",           "Sw in",     -1.0, 1.0,  45.0, False),
    Stage("Filter-bank LNA (ZX60-83LN12+ @1.5 GHz)","FB LNA",   +22.5, 1.39, 36.2, True),
    Stage("6LP8-1680 lowpass x2 (1.68 GHz cutoff)", "LPF x2",    -1.2, 1.2,  45.0, False),
    Stage("Band pad / match (1+2 dB)",              "Pad",       -2.0, 2.0,  45.0, False),
    Stage("Filter-bank output amp (ZX60-83LN12+)",  "Out amp",  +22.5, 1.39, 36.2, True),
    Stage("Output route (SW6004 + pad) -> modem",   "Out sw",    -2.5, 2.5,  45.0, False),
]
# NOTE: no mixer / IF stages — 1534.5 MHz goes straight to the modem L-band input.

# DVB-S2X required Es/N0 (dB) for QEF, ideal AWGN (representative subset)
MODCODS = {
    "QPSK 1/4": -2.35, "QPSK 1/3": -1.24, "QPSK 1/2": 1.00, "QPSK 2/3": 3.10,
    "QPSK 3/4": 4.03,  "QPSK 5/6": 5.18,  "8PSK 2/3": 5.50, "8PSK 3/4": 6.62,
    "8PSK 5/6": 7.91,  "16APSK 2/3": 8.97, "16APSK 3/4": 10.21, "16APSK 5/6": 11.61,
}

# ---- unit helpers ------------------------------------------------------------
db    = lambda x:  10.0 * np.log10(x)
lin   = lambda d:  10.0 ** (d / 10.0)
dbm2w = lambda d:  10.0 ** ((d - 30.0) / 10.0)
w2dbm = lambda w:  30.0 + 10.0 * np.log10(w)


# =============================================================================
# CORE CASCADE ANALYSIS
# =============================================================================
def analyze(stages, t0=T0_K):
    g_db = np.array([s.gain_db  for s in stages], float)
    nf_db = np.array([s.nf_db   for s in stages], float)
    oip3 = np.array([s.oip3_dbm for s in stages], float)

    g = lin(g_db)
    f = lin(nf_db)

    cum_g     = np.cumprod(g)
    cum_g_bef = np.concatenate(([1.0], cum_g[:-1]))

    contrib = np.empty_like(f)
    contrib[0]  = f[0]
    contrib[1:] = (f[1:] - 1.0) / cum_g_bef[1:]
    cum_f = np.cumsum(contrib)

    f_total    = cum_f[-1]
    nf_total   = db(f_total)
    te         = t0 * (f_total - 1.0)
    tsys       = T_ANT_K + te
    gain_total = db(cum_g[-1])
    gt         = G_RX_DBI - db(tsys)

    iip3_dbm = oip3 - g_db
    iip3_w   = dbm2w(iip3_dbm)
    iip3_casc_w   = 1.0 / np.sum(cum_g_bef / iip3_w)
    iip3_casc_dbm = w2dbm(iip3_casc_w)
    oip3_casc_dbm = iip3_casc_dbm + gain_total

    return dict(
        g_db=g_db, nf_db=nf_db, oip3=oip3,
        cum_g_db=db(cum_g), cum_nf_db=db(cum_f),
        contrib=contrib, contrib_pct=100.0 * contrib / f_total,
        iip3_dbm=iip3_dbm,
        f_total=f_total, nf_total=nf_total, te=te, tsys=tsys,
        gain_total=gain_total, gt=gt,
        iip3_casc_dbm=iip3_casc_dbm, oip3_casc_dbm=oip3_casc_dbm,
    )


def noise_floor(nf_db, bw_hz):
    return KTB0 + nf_db + 10.0 * np.log10(bw_hz)


def mds(nf_db, esn0_db, rs=RS_SYM, margin=IMPL_MARGIN):
    return KTB0 + nf_db + 10.0 * np.log10(rs) + esn0_db + margin


# =============================================================================
# TEXT REPORT
# =============================================================================
def print_report(stages, R):
    line = "-" * 92
    print("\n" + "=" * 92)
    print(f" GROUND STATION RF CASCADE  —  {FREQ_MHZ:.1f} MHz L-band FUSE downlink")
    print(" Direct-to-modem path (no down-conversion)")
    print("=" * 92)
    print(f"{'#':>2}  {'Stage':<32}{'Gain':>7}{'NF':>6}{'cumGain':>9}"
          f"{'cumNF':>7}{'NF share':>10}{'OIP3':>8}{'IIP3':>8}")
    print(f"{'':>2}  {'':<32}{'(dB)':>7}{'(dB)':>6}{'(dB)':>9}"
          f"{'(dB)':>7}{'(%)':>10}{'(dBm)':>8}{'(dBm)':>8}")
    print(line)
    for i, s in enumerate(stages):
        flag = "" if s.oip3_known else "*"
        print(f"{i+1:>2}  {s.short:<32}{R['g_db'][i]:>7.1f}{R['nf_db'][i]:>6.2f}"
              f"{R['cum_g_db'][i]:>9.1f}{R['cum_nf_db'][i]:>7.2f}"
              f"{R['contrib_pct'][i]:>10.2f}"
              f"{R['oip3'][i]:>7.1f}{flag}{R['iip3_dbm'][i]:>8.1f}")
    print(line)
    print("  * OIP3 is a placeholder (passive part, no datasheet value).\n")

    occ_bw = RS_SYM * (1 + ROLLOFF)
    print(" SYSTEM NOISE")
    print(f"   System noise figure ........ {R['nf_total']:6.2f} dB")
    print(f"   Noise factor (linear) ...... {R['f_total']:6.3f}")
    print(f"   Equivalent noise temp Te ... {R['te']:6.1f} K")
    print(f"   System noise temp Tsys ..... {R['tsys']:6.1f} K  "
          f"(T_ant {T_ANT_K:.0f} K + Te)")
    print(f"   Total chain gain ........... {R['gain_total']:6.1f} dB\n")

    print(" GAIN / TEMPERATURE")
    print(f"   L-band dish gain ........... {G_RX_DBI:6.1f} dBi  [placeholder]")
    print(f"   G/T ........................ {R['gt']:6.2f} dB/K\n")

    print(" NOISE FLOOR (referred to LNA input)")
    for b, lbl in [(1, "1 Hz"), (1e3, "1 kHz"), (1e6, "1 MHz"),
                   (RS_SYM, f"{RS_SYM/1e6:.0f} MHz (Rs)"), (10e6, "10 MHz")]:
        print(f"   {lbl:>14} : {noise_floor(R['nf_total'], b):8.1f} dBm")
    print()

    print(" LINEARITY (cascaded)")
    print(f"   Input  IP3 (IIP3) .......... {R['iip3_casc_dbm']:6.1f} dBm")
    print(f"   Output IP3 (OIP3) .......... {R['oip3_casc_dbm']:6.1f} dBm")
    nfloor_in = noise_floor(R['nf_total'], BW_SFDR)
    sfdr = (2.0 / 3.0) * (R['iip3_casc_dbm'] - nfloor_in)
    print(f"   SFDR (in {BW_SFDR/1e6:.0f} MHz) ........... {sfdr:6.1f} dB\n")

    print(f" DVB-S2X SENSITIVITY  (Rs={RS_SYM/1e6:.1f} Msym/s, occ BW "
          f"~{occ_bw/1e6:.1f} MHz, margin {IMPL_MARGIN:.1f} dB)")
    print(f"   {'MODCOD':<12}{'Es/N0 (dB)':>12}{'MDS (dBm)':>12}")
    for mc, esn0 in MODCODS.items():
        print(f"   {mc:<12}{esn0:>12.2f}{mds(R['nf_total'], esn0):>12.1f}")
    print("=" * 92)
    print(" NOTE: 1534.5 MHz lies in the 1525-1559 MHz band flagged for")
    print(" interference. The floor above is THERMAL (best case); the real")
    print(" in-band floor is higher. Capture cold-sky vs on-band IQ to separate")
    print(" intrinsic receiver noise from environmental interference.")
    print("=" * 92 + "\n")


def export_csv(stages, R, path="cascade_1534MHz.csv"):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# {FREQ_MHZ} MHz L-band FUSE downlink, direct-to-modem (no mixer)"])
        w.writerow(["#", "stage", "gain_dB", "nf_dB", "cum_gain_dB", "cum_nf_dB",
                    "nf_share_pct", "oip3_dBm", "oip3_is_placeholder", "iip3_dBm"])
        for i, s in enumerate(stages):
            w.writerow([i + 1, s.name, f"{R['g_db'][i]:.2f}", f"{R['nf_db'][i]:.2f}",
                        f"{R['cum_g_db'][i]:.2f}", f"{R['cum_nf_db'][i]:.3f}",
                        f"{R['contrib_pct'][i]:.3f}", f"{R['oip3'][i]:.1f}",
                        (not s.oip3_known), f"{R['iip3_dbm'][i]:.2f}"])
        w.writerow([])
        w.writerow(["system_NF_dB", f"{R['nf_total']:.3f}"])
        w.writerow(["Te_K", f"{R['te']:.1f}"])
        w.writerow(["Tsys_K", f"{R['tsys']:.1f}"])
        w.writerow(["total_gain_dB", f"{R['gain_total']:.2f}"])
        w.writerow(["GT_dB_per_K", f"{R['gt']:.2f}"])
        w.writerow(["IIP3_casc_dBm", f"{R['iip3_casc_dbm']:.2f}"])
        w.writerow(["OIP3_casc_dBm", f"{R['oip3_casc_dbm']:.2f}"])
    print(f"[saved] {path}")


# =============================================================================
# PLOT DASHBOARD
# =============================================================================
def make_dashboard(stages, R, path="cascade_1534MHz_dashboard.png"):
    n = len(stages)
    shorts = [s.short for s in stages]
    x = np.arange(n)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "figure.dpi": 100})
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(f"Ground Station Cascade — {FREQ_MHZ:.1f} MHz L-band FUSE "
                 f"downlink (direct to modem, no mixer)",
                 fontsize=14, fontweight="bold", color=THEME)

    # --- A: cumulative gain & NF profile -------------------------------------
    a = ax[0, 0]
    a.plot(x, R["cum_g_db"], "-o", color=THEME, lw=2)
    a.set_ylabel("Cumulative gain (dB)", color=THEME)
    a.tick_params(axis="y", labelcolor=THEME)
    a2 = a.twinx()
    a2.plot(x, R["cum_nf_db"], "-s", color=ACCENT, lw=2)
    a2.set_ylabel("Cumulative NF (dB)", color=ACCENT)
    a2.tick_params(axis="y", labelcolor=ACCENT)
    a.set_xticks(x); a.set_xticklabels(shorts, rotation=40, ha="right")
    a.set_title("A. Gain & noise-figure build-up")
    a.grid(alpha=0.3); a.axhline(0, color="gray", lw=0.6)
    a2.text(0.97, 0.5, f"system NF\n{R['nf_total']:.2f} dB", transform=a2.transAxes,
            ha="right", va="center", fontsize=9, color=ACCENT,
            bbox=dict(boxstyle="round", fc="#fff3e9", ec=ACCENT))

    # --- B: per-stage noise contribution -------------------------------------
    b = ax[0, 1]
    colors = [ACCENT if p == R["contrib_pct"].max() else THEME
              for p in R["contrib_pct"]]
    bars = b.bar(x, R["contrib_pct"], color=colors)
    b.set_yscale("log")
    b.set_ylabel("Share of total noise factor (%)")
    b.set_xticks(x); b.set_xticklabels(shorts, rotation=40, ha="right")
    b.set_title("B. Noise budget — where the NF comes from")
    b.grid(alpha=0.3, axis="y")
    for rect, p in zip(bars, R["contrib_pct"]):
        txt = f"{p:.1f}%" if p >= 0.1 else f"{p:.2f}%"
        b.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.15,
               txt, ha="center", va="bottom", fontsize=7.5)

    # --- C: signal-level diagram ---------------------------------------------
    c = ax[1, 0]
    xn = np.arange(n + 1)
    labels = ["In"] + shorts
    sig = np.concatenate(([PIN_DBM], PIN_DBM + R["cum_g_db"]))
    n_in = KTB0 + 10 * np.log10(BW_LEVEL)
    noise = np.concatenate(([n_in], n_in + R["cum_nf_db"] + R["cum_g_db"]))
    c.plot(xn, sig, "-o", color=THEME, lw=2, label="Signal")
    c.plot(xn, noise, "-s", color=RED, lw=2, label="Noise floor")
    c.fill_between(xn, noise, sig, color=THEME, alpha=0.10)
    c.set_xticks(xn); c.set_xticklabels(labels, rotation=40, ha="right")
    c.set_ylabel("Power (dBm)")
    c.set_title(f"C. Signal-level diagram  (Pin={PIN_DBM:.0f} dBm, "
                f"B={BW_LEVEL/1e6:.0f} MHz)")
    c.grid(alpha=0.3); c.legend(loc="upper left", fontsize=8)
    snr_out = sig[-1] - noise[-1]
    c.annotate(f"output SNR ≈ {snr_out:.1f} dB", xy=(xn[-1], (sig[-1]+noise[-1])/2),
               xytext=(xn[-1]-3.2, (sig[-1]+noise[-1])/2), fontsize=8, color=THEME)

    # --- D: noise floor vs bandwidth -----------------------------------------
    d = ax[1, 1]
    bw = np.logspace(0, 9, 300)
    d.semilogx(bw, noise_floor(R["nf_total"], bw), color=THEME, lw=2)
    for bmark, lbl in [(1e6, "1 MHz"), (RS_SYM, f"{RS_SYM/1e6:.0f} MHz (Rs)"),
                       (10e6, "10 MHz")]:
        y = noise_floor(R["nf_total"], bmark)
        d.plot(bmark, y, "o", color=ACCENT)
        d.annotate(f"{lbl}\n{y:.0f} dBm", (bmark, y), textcoords="offset points",
                   xytext=(6, -20), fontsize=8, color=ACCENT)
    d.set_xlabel("Bandwidth (Hz)"); d.set_ylabel("Noise floor (dBm)")
    d.set_title(f"D. Noise floor vs bandwidth  (NF={R['nf_total']:.2f} dB)")
    d.grid(alpha=0.3, which="both")

    # --- E: DVB-S2X sensitivity ----------------------------------------------
    e = ax[2, 0]
    mcs = list(MODCODS.keys())
    vals = [mds(R["nf_total"], MODCODS[m]) for m in mcs]
    yb = np.arange(len(mcs))
    e.barh(yb, vals, color=THEME)
    e.set_yticks(yb); e.set_yticklabels(mcs, fontsize=8)
    e.invert_yaxis()
    floor_rs = KTB0 + R["nf_total"] + 10 * np.log10(RS_SYM)
    e.axvline(floor_rs, color=RED, ls="--", lw=1.5,
              label=f"noise floor in Rs\n{floor_rs:.1f} dBm")
    for yi, v in zip(yb, vals):
        e.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=7.5)
    e.set_xlabel("Required Rx power / MDS (dBm)")
    e.set_title(f"E. DVB-S2X sensitivity  (Rs={RS_SYM/1e6:.0f} Msps, "
                f"+{IMPL_MARGIN:.0f} dB margin)")
    e.grid(alpha=0.3, axis="x"); e.legend(loc="lower right", fontsize=7.5)

    # --- F: cascaded linearity / SFDR ----------------------------------------
    f_ax = ax[2, 1]
    iip3, oip3c, g_tot = R["iip3_casc_dbm"], R["oip3_casc_dbm"], R["gain_total"]
    pin = np.linspace(-80, iip3 + 8, 120)
    fund = pin + g_tot
    im3 = 3 * pin + (oip3c - 3 * iip3)
    f_ax.plot(pin, fund, color=THEME, lw=2, label="Fundamental (1:1)")
    f_ax.plot(pin, im3, color=ACCENT, lw=2, ls="--", label="IM3 product (3:1)")
    f_ax.plot(iip3, oip3c, "k*", ms=13)
    f_ax.annotate(f"IIP3={iip3:.1f} dBm\nOIP3={oip3c:.1f} dBm", (iip3, oip3c),
                  textcoords="offset points", xytext=(-115, -4), fontsize=8)
    nfloor_out = KTB0 + R["nf_total"] + 10 * np.log10(BW_SFDR) + g_tot
    f_ax.axhline(nfloor_out, color=RED, ls=":", lw=1.5,
                 label=f"out noise floor ({BW_SFDR/1e6:.0f} MHz)")
    sfdr = (2.0 / 3.0) * (oip3c - nfloor_out)
    f_ax.set_xlabel("Input power per tone (dBm)")
    f_ax.set_ylabel("Output power (dBm)")
    f_ax.set_title(f"F. Cascaded linearity — SFDR ≈ {sfdr:.1f} dB")
    f_ax.grid(alpha=0.3); f_ax.legend(loc="lower right", fontsize=7.5)
    f_ax.set_ylim(nfloor_out - 15, oip3c + 15)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    R = analyze(STAGES)
    print_report(STAGES, R)
    make_dashboard(STAGES, R)
    export_csv(STAGES, R)


if __name__ == "__main__":
    main()
