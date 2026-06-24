#!/usr/bin/env python3
"""
Ground Station Noise-Figure Calculator  —  1534.5 MHz L-band downlink
Simple, step-by-step cascade through the full receive chain.

Antenna data (from the Receive Antenna page):
  * 72" wideband mesh dish, designed/built at MITRE
  * Feed: A-Info LX-1080_LHCPSPO spiral, 1-8 GHz, left-hand circular
  * Measured reflector+feed gain ~17 dBiC at 1.5 GHz up to ~30 dBiC at 6 GHz
  * LNA mounted at the feed via a SHORT coax; a 10-20 ft cable then runs to the RF Box
  * 1534.5 MHz is inside the modem L-band input (950-2150 MHz) -> NO down-conversion

Outputs: a step-by-step console report + cascade_1534MHz_simple.png (3 plots)
Requires: numpy, matplotlib       (pip install numpy matplotlib)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- CONFIG (edit me) -----------------------------
FREQ_MHZ = 1534.5
T0       = 290.0     # reference temperature for noise figure (K)
T_ANT    = 80.0      # antenna noise temp (K)   [PLACEHOLDER - refine for mesh dish/elevation]
G_DISH   = 16.5      # dish+feed gain at 1.5 GHz (dBiC), read off the measured curve (~16-17)
PIN_DBM  = -100.0    # example signal level into the feed jumper (for the level diagram), dBm
BW_HZ    = 5e6       # bandwidth used for the level diagram (Hz)

KTB0     = -174.0    # kT0 thermal noise density (dBm/Hz @ 290 K)

# --------------------------- RECEIVE CHAIN ----------------------------------
# Signal path from the feed to the modem.  (name, gain_dB, NF_dB)
#   gain_dB : + for gain, - for loss      NF_dB : for a passive part, NF = its loss
CHAIN = [
    ("Feed jumper (short coax, feed->LNA)",      -0.3, 0.3),   # PRE-LNA: adds ~1:1 to NF
    ("Feed LNA (ZX60-83LN12+ @1.5 GHz)",        +22.5, 1.39),  # sets the noise floor
    ("Cable run 10-20 ft (feed->RF Box)",        -1.0, 1.0),   # placeholder for ~15 ft @1.5 GHz
    ("Bias tee (ZX85-12G-S+)",                   -0.4, 0.4),
    ("RF Box input switch (SW264K)",             -1.0, 1.0),
    ("Filter-bank LNA (ZX60-83LN12+)",          +22.5, 1.39),
    ("6LP8-1680 lowpass x2 (passes 1534)",       -1.2, 1.2),
    ("Band pad / match",                         -2.0, 2.0),
    ("Filter-bank output amp (ZX60-83LN12+)",   +22.5, 1.39),
    ("Output route (SW6004 + pad) -> modem",     -2.5, 2.5),
]
SHORT = ["Feed jmp", "Feed LNA", "Cable", "Bias T", "Sw in",
         "FB LNA", "LPF x2", "Pad", "Out amp", "Out sw"]

db  = lambda x: 10 * np.log10(x)
lin = lambda d: 10 ** (d / 10)

# --------------------- STEP-BY-STEP FRIIS CASCADE ---------------------------
def cascade(chain):
    """Walk the chain one component at a time, printing the running NF & gain."""
    print("=" * 100)
    print(f" NOISE-FIGURE CASCADE  —  {FREQ_MHZ} MHz   (Friis, referred to the feed-jumper input)")
    print("=" * 100)
    print(f"{'#':>2} {'Component':<41}{'Gain':>7}{'NF':>7}{'adds to F':>12}"
          f"{'NF so far':>11}{'Gain so far':>13}")
    print(f"{'':>2} {'':<41}{'(dB)':>7}{'(dB)':>7}{'(linear)':>12}{'(dB)':>11}{'(dB)':>13}")
    print("-" * 100)

    F = 0.0          # running noise factor (linear)
    Glin = 1.0       # running gain (linear) AHEAD of the next stage
    cum_nf, cum_gain = [], []

    for i, (name, gd, nfd) in enumerate(chain):
        fi, gi = lin(nfd), lin(gd)
        if i == 0:
            add = fi                 # first stage contributes its full noise factor
            F = fi
        else:
            add = (fi - 1) / Glin    # later stages are divided by the gain ahead of them
            F += add
        Glin *= gi                   # update cumulative gain through this stage
        cum_nf.append(db(F))
        cum_gain.append(db(Glin))
        print(f"{i+1:>2} {name:<41}{gd:>7.1f}{nfd:>7.2f}{add:>12.4f}"
              f"{db(F):>11.3f}{db(Glin):>13.1f}")

    print("-" * 100)
    return np.array(cum_nf), np.array(cum_gain), F, Glin


def main():
    cum_nf, cum_gain, F, Glin = cascade(CHAIN)
    NF   = db(F)
    Te   = T0 * (F - 1)
    Tsys = T_ANT + Te
    Gtot = db(Glin)
    GT   = G_DISH - db(Tsys)

    print(f"\n SYSTEM RESULT")
    print(f"   System noise figure  NF = {NF:6.2f} dB     (noise factor F = {F:.3f})")
    print(f"   Equivalent noise temp Te = {Te:6.1f} K")
    print(f"   Total receive gain       = {Gtot:6.1f} dB")

    print(f"\n ANTENNA / G-OVER-T")
    print(f"   Dish+feed gain @1.5 GHz  = {G_DISH:6.1f} dBiC   (72\" mesh dish + A-Info spiral feed)")
    print(f"   Antenna temp T_ant       = {T_ANT:6.0f} K     [placeholder]")
    print(f"   System noise temp Tsys   = {Tsys:6.1f} K")
    print(f"   G/T                      = {GT:6.2f} dB/K")

    print(f"\n NOISE FLOOR (referred to the feed-jumper input)")
    for b, lbl in [(1, "1 Hz"), (1e3, "1 kHz"), (1e6, "1 MHz"),
                   (BW_HZ, f"{BW_HZ/1e6:.0f} MHz"), (10e6, "10 MHz")]:
        print(f"   {lbl:>8} : {KTB0 + NF + 10*np.log10(b):8.1f} dBm")
    print("\n NOTE: this is the THERMAL (best-case) floor. 1534.5 MHz sits in the")
    print(" 1525-1559 MHz band flagged for interference, so the real in-band floor")
    print(" will be higher. The feed jumper before the LNA is the highest-leverage")
    print(" loss in the chain - keep it as short as possible.\n")

    make_plots(cum_nf, cum_gain, NF)


# ------------------------------- PLOTS --------------------------------------
def make_plots(cum_nf, cum_gain, NF, path="cascade_1534MHz_simple.png"):
    NAVY, ORANGE, RED = "#1f3a5f", "#e07b39", "#b23a48"
    n = len(SHORT)
    x = np.arange(n)
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(3, 1, figsize=(11, 14))
    fig.suptitle(f"{FREQ_MHZ} MHz Receive Chain — Noise Figure, Gain, "
                 f"Signal Level & Noise Floor", fontsize=14, fontweight="bold",
                 color=NAVY, y=0.997)

    # 1) Noise-figure build-up and gain ---------------------------------------
    a = ax[0]
    a.plot(x, cum_gain, "-o", color=NAVY, lw=2, label="Cumulative gain")
    a.set_ylabel("Cumulative gain (dB)", color=NAVY)
    a.tick_params(axis="y", labelcolor=NAVY)
    a.axhline(0, color="gray", lw=0.6)
    a2 = a.twinx()
    a2.plot(x, cum_nf, "-s", color=ORANGE, lw=2, label="Cumulative NF")
    a2.set_ylabel("Cumulative noise figure (dB)", color=ORANGE)
    a2.tick_params(axis="y", labelcolor=ORANGE)
    a2.text(0.97, 0.45, f"system NF = {NF:.2f} dB", transform=a2.transAxes,
            ha="right", fontsize=10, color=ORANGE,
            bbox=dict(boxstyle="round", fc="#fff3e9", ec=ORANGE))
    a.set_xticks(x); a.set_xticklabels(SHORT, rotation=35, ha="right")
    a.set_title("1.  Noise-figure build-up & gain", fontweight="bold")
    a.grid(alpha=0.3)

    # 2) Signal-level diagram -------------------------------------------------
    c = ax[1]
    xn = np.arange(n + 1)
    sig = np.concatenate(([PIN_DBM], PIN_DBM + cum_gain))
    n_in = KTB0 + 10 * np.log10(BW_HZ)
    noise = np.concatenate(([n_in], n_in + cum_nf + cum_gain))
    c.plot(xn, sig, "-o", color=NAVY, lw=2, label="Signal")
    c.plot(xn, noise, "-s", color=RED, lw=2, label="Noise floor")
    c.fill_between(xn, noise, sig, color=NAVY, alpha=0.10)
    c.set_xticks(xn); c.set_xticklabels(["In"] + SHORT, rotation=35, ha="right")
    c.set_ylabel("Power (dBm)")
    c.set_title(f"2.  Signal-level diagram  (Pin = {PIN_DBM:.0f} dBm, "
                f"B = {BW_HZ/1e6:.0f} MHz)", fontweight="bold")
    c.grid(alpha=0.3); c.legend(loc="upper left")
    c.annotate(f"output SNR ≈ {sig[-1]-noise[-1]:.1f} dB",
               xy=(xn[-1], (sig[-1]+noise[-1])/2),
               xytext=(xn[-1]-3.4, (sig[-1]+noise[-1])/2 + 4), fontsize=9, color=NAVY)

    # 3) Noise floor vs bandwidth ---------------------------------------------
    d = ax[2]
    bw = np.logspace(0, 9, 300)
    d.semilogx(bw, KTB0 + NF + 10*np.log10(bw), color=NAVY, lw=2)
    for bmark, lbl in [(1e6, "1 MHz"), (BW_HZ, f"{BW_HZ/1e6:.0f} MHz"), (10e6, "10 MHz")]:
        y = KTB0 + NF + 10*np.log10(bmark)
        d.plot(bmark, y, "o", color=ORANGE)
        d.annotate(f"{lbl}: {y:.0f} dBm", (bmark, y), textcoords="offset points",
                   xytext=(8, -4), fontsize=9, color=ORANGE)
    d.set_xlabel("Bandwidth (Hz)"); d.set_ylabel("Noise floor (dBm)")
    d.set_title(f"3.  Noise floor vs bandwidth  (NF = {NF:.2f} dB)", fontweight="bold")
    d.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
