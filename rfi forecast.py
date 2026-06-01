#!/usr/bin/env python3
"""
rfi_forecast.py  —  Satellite RFI forecasting for a radio-astronomy site.

WHAT IT DOES
------------
For each catalogued satellite and a fixed ground site it:
  1. Propagates the orbit with SGP4 (via Skyfield) and finds every above-horizon
     pass in a forward time window  ->  WHEN does a pass happen.
  2. Samples each pass to produce elevation, azimuth (sky-path), slant range,
     Doppler shift and Doppler rate ("deviation").
  3. Runs a physics-based link budget to estimate the interference-to-noise
     ratio (INR) at a monitor SDR (e.g. Deepwave Air-T 7310)  ->  WILL it
     interfere, and HOW BAD.
  4. Propagates the *uncertainty* (stale TLE -> orbit error, transmit duty
     cycle, EIRP spread) with a Monte-Carlo so every pass gets a probability of
     harmful interference, and the whole catalogue gets an ROC curve that turns
     a warn-threshold into a (prob. of missed interference, prob. of false
     alarm) pair  ->  the "not an ideal world" question.

DESIGN CHOICE: few, physical parameters. Everything tunable lives in SITE /
RX / GEO_ERR / the per-satellite dict below, each with a defensible default and
a one-line meaning. Nothing is fit; nothing is magic.

Outputs results.json (consumed by the dashboard) and prints a summary table.
"""

from __future__ import annotations
import json, math, argparse, sys, datetime as dt
import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs

C = 299_792_458.0            # m/s
K_DBW = -228.6               # 10*log10(Boltzmann) dBW / (Hz*K)

# ----------------------------------------------------------------------------
# SITE  — MITRE Bedford, MA (the radio-astronomy / monitoring site)
# ----------------------------------------------------------------------------
SITE = dict(name="Bedford, MA (MITRE)", lat=42.50518, lon=-71.23530, alt_m=68.0)

# RECEIVER / monitor SDR.  Defaults describe an omni-ish RFI monitor (discone /
# biconical) feeding an Air-T-class front end. Override with your real numbers.
RX = dict(
    T_sys_K=150.0,    # system noise temperature (LNA + sky + ground)
    G_rx_dBi=0.0,     # monitor antenna gain toward the satellite (omni ~ 0 dBi)
    el_mask_deg=0.0,  # horizon mask used for pass detection
)

# ORBIT-ERROR model. TLE position error grows with epoch age. We inject it in
# the RTN frame (Along-track / Cross-track / Radial) as sigma = base + rate*age.
# Defaults are conservative rule-of-thumb values for LEO public elements.
GEO_ERR = dict(
    A_base_km=0.5, A_rate_km_per_day=1.5,   # along-track (largest)
    C_base_km=0.3, C_rate_km_per_day=0.3,   # cross-track (drives max elevation)
    R_base_km=0.2, R_rate_km_per_day=0.2,   # radial      (drives altitude/range)
)

# Interference thresholds (INR, dB).
T_FLOOR_DB = 0.0     # signal == receiver noise floor  -> recording corruption
T_RA769_DB = -10.0   # ITU-R RA.769-style detrimental level for radio astronomy
T_HARM_DB  = T_FLOOR_DB   # "truth" definition for the probability model

# ----------------------------------------------------------------------------
# SATELLITE CATALOGUE.  TLEs fetched 2026-06-01 (METOP-C ~7 d old, CORTEZ ~26 d).
# Swap in fresh elements from Space-Track for operations (see --tle override).
# ----------------------------------------------------------------------------
SATS = [
    dict(
        name="METOP-C", norad=43689,
        l1="1 43689U 18087A   26145.54402736  .00000096  00000-0  63709-4 0  9997",
        l2="2 43689  98.6655 205.5600 0000799 150.8284 209.2938 14.21505074391614",
        downlink_label="AHRPT", f_hz=1701.3e6,  # L-band high-rate broadcast
        eirp_dbw=18.0, sigma_eirp_db=2.0,        # strong, continuous downlink
        occ_bw_hz=3.5e6, p_tx=1.00,              # broadcasts globally & always-on
        atm_zenith_db=0.04,
        note="EUMETSAT polar weather sat. AHRPT @1701.3 MHz is a classic L-band RFI source.",
    ),
    dict(
        name="M-SEL/CORTEZ", norad=63231,
        l1="1 63231U 25052X   26126.42601873  .00007710  00000-0  30702-3 0  9997",
        l2="2 63231  97.3960  20.4301 0003284 313.3836  46.7129 15.25511285 63558",
        downlink_label="UHF TT&C", f_hz=400.5e6,  # GFSK 38.4 kbaud
        eirp_dbw=0.0, sigma_eirp_db=3.0,          # ~1 W into near-omni
        occ_bw_hz=50e3, p_tx=0.35,                # scheduled contacts, not always on
        atm_zenith_db=0.02,
        note="MITRE / Astro Digital CORVUS-MICRO smallsat. UHF @400.5 MHz, scheduled TT&C.",
    ),
]

# ----------------------------------------------------------------------------
# Geometry helpers (all in ECEF/ITRF, validated against Skyfield altaz)
# ----------------------------------------------------------------------------
def enu_basis(lat_deg, lon_deg):
    phi, lam = math.radians(lat_deg), math.radians(lon_deg)
    up    = np.array([math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi)])
    east  = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north = np.array([-math.sin(phi)*math.cos(lam), -math.sin(phi)*math.sin(lam), math.cos(phi)])
    return up, east, north

def topo_from_ecef(r_sat_km, r_site_km, up, east, north):
    topo = r_sat_km - r_site_km
    rng = np.linalg.norm(topo, axis=-1)
    u = topo / rng[..., None]
    el = np.degrees(np.arcsin(np.clip(u @ up, -1, 1)))
    az = np.degrees(np.arctan2(u @ east, u @ north)) % 360.0
    return el, az, rng, topo

def link_budget(eirp_dbw, f_hz, rng_km, el_deg, sat, rx):
    """Return INR(dB), received power(dBW), PFD(dBW/m^2)."""
    d_m = rng_km * 1e3
    fspl = 20*np.log10(d_m) + 20*np.log10(f_hz) - 147.5522124       # 20log10(4pi/c)
    el_eff = np.clip(el_deg, 3.0, 90.0)
    l_atm = sat["atm_zenith_db"] / np.sin(np.radians(el_eff))
    p_rx = eirp_dbw - fspl - l_atm + rx["G_rx_dBi"]
    noise = K_DBW + 10*np.log10(rx["T_sys_K"]) + 10*np.log10(sat["occ_bw_hz"])
    inr = p_rx - noise
    pfd = eirp_dbw - 20*np.log10(d_m) - 10.99206                    # 10log10(4pi)
    return inr, p_rx, pfd

# ----------------------------------------------------------------------------
# Pass finding + sampling
# ----------------------------------------------------------------------------
def find_passes(sat_sf, site_sf, ts, t0, t1, el_mask):
    t, ev = sat_sf.find_events(site_sf, t0, t1, altitude_degrees=el_mask)
    passes, cur = [], {}
    for ti, e in zip(t, ev):
        if e == 0: cur = {"aos": ti}
        elif e == 1: cur["tca"] = ti
        elif e == 2:
            if "aos" in cur and "tca" in cur:
                cur["los"] = ti; passes.append(cur)
            cur = {}
    return passes

def sample_pass(sat_sf, p, ts, up, east, north, site_ecef, n=140):
    aos, los = p["aos"], p["los"]
    jd = np.linspace(aos.tt, los.tt, n)
    t = ts.tt_jd(jd)
    rsat, vsat = sat_sf.at(t).frame_xyz_and_velocity(itrs)
    rs = rsat.km.T            # (n,3)
    vs = vsat.km_per_s.T      # (n,3)
    el, az, rng, topo = topo_from_ecef(rs, site_ecef, up, east, north)
    rr = np.einsum("ij,ij->i", topo, vs) / rng     # km/s, site fixed in ECEF
    secs = (jd - jd[0]) * 86400.0
    return dict(t=t, secs=secs, rs=rs, vs=vs, el=el, az=az, rng=rng, rr=rr)

# ----------------------------------------------------------------------------
# Monte-Carlo interference risk for one pass
# ----------------------------------------------------------------------------
def rtn_offsets(rs, vs, A, Cc, R, rng_count, rngn):
    """Build per-realization RTN displacement vectors (km) at the TCA geometry."""
    # RTN basis at closest approach (representative for the short pass)
    r = rs; v = vs
    rhat = r / np.linalg.norm(r)
    chat = np.cross(r, v); chat /= np.linalg.norm(chat)
    ahat = np.cross(chat, rhat)
    a = rngn.normal(0, A, rng_count)
    c = rngn.normal(0, Cc, rng_count)
    rr = rngn.normal(0, R, rng_count)
    return (a[:, None]*ahat + c[:, None]*chat + rr[:, None]*rhat)  # (K,3)

def montecarlo_pass(samp, sat, rx, age_days, geo, T_harm, K=2000, seed=0):
    rngn = np.random.default_rng(seed)
    up, east, north = samp["_basis"]
    site_ecef = samp["_site"]
    # nominal peak INR (predictor's score): transmit on, no orbit error
    inr_nom_curve, _, _ = link_budget(sat["eirp_dbw"], sat["f_hz"], samp["rng"], samp["el"], sat, rx)
    inr_nom_peak = float(np.max(inr_nom_curve))

    # orbit-error sigmas (km) from TLE age
    A  = geo["A_base_km"] + geo["A_rate_km_per_day"]*age_days
    Cc = geo["C_base_km"] + geo["C_rate_km_per_day"]*age_days
    R  = geo["R_base_km"] + geo["R_rate_km_per_day"]*age_days

    # TCA index for RTN basis
    itca = int(np.argmax(samp["el"]))
    offs = rtn_offsets(samp["rs"][itca], samp["vs"][itca], A, Cc, R, K, rngn)  # (K,3)

    eirp = rngn.normal(sat["eirp_dbw"], sat["sigma_eirp_db"], K)
    tx_on = rngn.random(K) < sat["p_tx"]

    # peak INR per realization: shift whole track by RTN offset (constant over pass)
    rs = samp["rs"][None, :, :] + offs[:, None, :]          # (K,n,3)
    topo = rs - site_ecef
    rng = np.linalg.norm(topo, axis=2)                      # (K,n)
    u = topo / rng[:, :, None]
    el = np.degrees(np.arcsin(np.clip(u @ up, -1, 1)))      # (K,n)
    d_m = rng*1e3
    fspl = 20*np.log10(d_m) + 20*np.log10(sat["f_hz"]) - 147.5522124
    el_eff = np.clip(el, 3.0, 90.0)
    l_atm = sat["atm_zenith_db"]/np.sin(np.radians(el_eff))
    noise = K_DBW + 10*np.log10(rx["T_sys_K"]) + 10*np.log10(sat["occ_bw_hz"])
    inr = (eirp[:, None] - fspl - l_atm + rx["G_rx_dBi"]) - noise   # (K,n)
    # only count samples that are actually above the horizon for that realization
    inr = np.where(el > 0.0, inr, -300.0)
    peak = np.max(inr, axis=1)                              # (K,)
    peak = np.where(tx_on, peak, -300.0)                    # not transmitting -> no RFI
    truth = peak > T_harm
    p_int = float(np.mean(truth))
    cond_peak = float(np.mean(peak[tx_on])) if tx_on.any() else float("nan")
    return dict(inr_nom_peak=inr_nom_peak, p_int=p_int,
                peak_samples=peak, truth=truth, warn_score=p_int,
                sigma_along_km=A, sigma_cross_km=Cc, sigma_radial_km=R,
                cond_peak_inr=cond_peak)

# ----------------------------------------------------------------------------
# ROC across all passes (warn-threshold sweep on nominal INR; truth from MC)
# ----------------------------------------------------------------------------
def build_roc(all_mc, thresholds):
    roc = []
    truth_all = np.concatenate([m["truth"] for m in all_mc])
    score_all = np.concatenate([np.full(m["truth"].shape, m["warn_score"]) for m in all_mc])
    pos = truth_all.sum(); neg = (~truth_all).sum()
    for T in thresholds:
        warn = score_all > T
        pd = float((warn & truth_all).sum() / pos) if pos else 0.0
        pfa = float((warn & ~truth_all).sum() / neg) if neg else 0.0
        roc.append(dict(T=float(T), pd=pd, pfa=pfa, pmd=1.0-pd))
    return roc, int(pos), int(neg)

# ----------------------------------------------------------------------------
def risk_level(p_int, inr_peak):
    if p_int >= 0.5 or inr_peak >= 10: return "HIGH"
    if p_int >= 0.15 or inr_peak >= 0: return "MEDIUM"
    if p_int >= 0.02 or inr_peak >= T_RA769_DB: return "LOW"
    return "MINIMAL"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--start", default="2026-06-01T19:00:00Z",
                    help="UTC window start (default ~now)")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--mc", type=int, default=3000)
    args = ap.parse_args()

    ts = load.timescale(builtin=True)
    site_sf = wgs84.latlon(SITE["lat"], SITE["lon"], elevation_m=SITE["alt_m"])
    up, east, north = enu_basis(SITE["lat"], SITE["lon"])
    t_start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    t0 = ts.from_datetime(t_start)
    t1 = ts.from_datetime(t_start + dt.timedelta(hours=args.hours))
    site_ecef = site_sf.at(t0).frame_xyz(itrs).km

    out = dict(site=SITE, rx=RX, geo_err=GEO_ERR,
               thresholds=dict(floor=T_FLOOR_DB, ra769=T_RA769_DB, harm=T_HARM_DB),
               generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               window=dict(start=args.start, hours=args.hours),
               satellites=[], passes=[])

    all_mc = []
    print(f"\nSite: {SITE['name']}  ({SITE['lat']}, {SITE['lon']}, {SITE['alt_m']} m)")
    print(f"Window: {args.start}  +{args.hours:.0f} h\n")

    for si, sat in enumerate(SATS):
        sat_sf = EarthSatellite(sat["l1"], sat["l2"], sat["name"], ts)
        age_days = float(t0 - sat_sf.epoch)
        out["satellites"].append(dict(
            name=sat["name"], norad=sat["norad"], downlink_label=sat["downlink_label"],
            f_mhz=sat["f_hz"]/1e6, eirp_dbw=sat["eirp_dbw"], occ_bw_khz=sat["occ_bw_hz"]/1e3,
            p_tx=sat["p_tx"], tle_age_days=round(age_days, 2),
            tle_epoch=sat_sf.epoch.utc_iso(), note=sat["note"],
            l1=sat["l1"], l2=sat["l2"]))
        passes = find_passes(sat_sf, site_sf, ts, t0, t1, RX["el_mask_deg"])
        print(f"=== {sat['name']}  (NORAD {sat['norad']}, {sat['downlink_label']} "
              f"{sat['f_hz']/1e6:.1f} MHz, TLE age {age_days:.1f} d) — {len(passes)} passes ===")
        for p in passes:
            samp = sample_pass(sat_sf, p, ts, up, east, north, site_ecef)
            samp["_basis"] = (up, east, north); samp["_site"] = site_ecef
            inr_curve, prx_curve, pfd_curve = link_budget(
                sat["eirp_dbw"], sat["f_hz"], samp["rng"], samp["el"], sat, RX)
            f = sat["f_hz"]
            doppler = -f * (samp["rr"]*1e3) / C            # Hz
            drate = np.gradient(doppler, samp["secs"])     # Hz/s ("deviation")
            mc = montecarlo_pass(samp, sat, RX, age_days, GEO_ERR, T_HARM_DB,
                                 K=args.mc, seed=1000*si + len(out["passes"]))
            all_mc.append(mc)
            imax = int(np.argmax(samp["el"]))
            rec = dict(
                sat=sat["name"], norad=sat["norad"], f_mhz=f/1e6,
                aos=p["aos"].utc_iso(), tca=p["tca"].utc_iso(), los=p["los"].utc_iso(),
                aos_unix=p["aos"].utc_datetime().timestamp(),
                tca_unix=p["tca"].utc_datetime().timestamp(),
                los_unix=p["los"].utc_datetime().timestamp(),
                dur_min=round(float(samp["secs"][-1])/60.0, 2),
                max_el=round(float(samp["el"][imax]), 2),
                tca_az=round(float(samp["az"][imax]), 1),
                min_range_km=round(float(np.min(samp["rng"])), 1),
                doppler_max_khz=round(float(np.max(np.abs(doppler)))/1e3, 3),
                drate_max_hz_s=round(float(np.max(np.abs(drate))), 1),
                inr_peak_db=round(mc["inr_nom_peak"], 2),
                pfd_peak_dbw_m2=round(float(np.max(pfd_curve)), 1),
                p_interference=round(mc["p_int"], 3),
                cond_peak_inr_db=round(mc["cond_peak_inr"], 2),
                sigma_along_km=round(mc["sigma_along_km"], 1),
                sigma_cross_km=round(mc["sigma_cross_km"], 1),
                risk=risk_level(mc["p_int"], mc["inr_nom_peak"]),
                # downsampled curves for the dashboard
                secs=[round(float(x), 1) for x in samp["secs"][::3]],
                el=[round(float(x), 2) for x in samp["el"][::3]],
                az=[round(float(x), 2) for x in samp["az"][::3]],
                rng=[round(float(x), 1) for x in samp["rng"][::3]],
                doppler_khz=[round(float(x)/1e3, 3) for x in doppler[::3]],
                drate_hz_s=[round(float(x), 1) for x in drate[::3]],
                inr_db=[round(float(x), 2) for x in inr_curve[::3]],
            )
            out["passes"].append(rec)

    out["passes"].sort(key=lambda r: r["aos_unix"])

    # ROC over the whole catalogue (detector score = predicted P(interference))
    thr = np.linspace(0.0, 1.0, 101)
    roc, npos, nneg = build_roc(all_mc, thr)
    out["roc"] = roc
    out["roc_counts"] = dict(harmful=npos, benign=nneg)

    # "now" snapshot look-angles for the live header
    out["now"] = []
    tnow = ts.from_datetime(dt.datetime.now(dt.timezone.utc))
    for sat in SATS:
        sat_sf = EarthSatellite(sat["l1"], sat["l2"], sat["name"], ts)
        alt, az, dist = (sat_sf - site_sf).at(tnow).altaz()
        out["now"].append(dict(name=sat["name"], el=round(alt.degrees, 1),
                               az=round(az.degrees, 1), range_km=round(dist.km, 0)))

    with open(args.out, "w") as fp:
        json.dump(out, fp)

    # ---- console summary -----------------------------------------------------
    print("\n----------------------------------------------------------------------------------")
    print(f"{'Satellite':<14}{'AOS (UTC)':<20}{'maxEl':>6}{'dur':>6}{'|Dopp|':>9}"
          f"{'INRpk':>7}{'P(int)':>8}  Risk")
    print("----------------------------------------------------------------------------------")
    for r in out["passes"]:
        print(f"{r['sat']:<14}{r['aos'][5:19].replace('T',' '):<20}{r['max_el']:>5.0f}°"
              f"{r['dur_min']:>5.0f}m{r['doppler_max_khz']:>7.1f}k{r['inr_peak_db']:>7.1f}"
              f"{r['p_interference']:>8.2f}  {r['risk']}")
    print("----------------------------------------------------------------------------------")
    print(f"Monte-Carlo realizations: {args.mc}/pass | harmful={npos} benign={nneg}")
    # report a couple of ROC operating points
    for target in (0.05, 0.10):
        best = min((x for x in roc if x["pfa"] <= target), key=lambda x: x["pmd"], default=None)
        if best:
            print(f"  @ Pfa<= {target:.2f}: warn if P(int) > {best['T']:.2f} "
                  f"-> Pd={best['pd']:.2f}  Pmd(missed)={best['pmd']:.2f}")
    print(f"\nWrote {args.out}  ({len(out['passes'])} passes)\n")

if __name__ == "__main__":
    main()
