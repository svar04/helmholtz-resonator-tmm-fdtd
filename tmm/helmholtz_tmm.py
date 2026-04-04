import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import subprocess
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# USER PARAMETERS — change these to re-run with different parameters

ar_start  = 1.0     # lowest aspect ratio to sweep
ar_stop   = 5.0     # highest aspect ratio to sweep
ar_step   = 0.05    # step between aspect ratios

num_offsets   = 201   # how many neck positions per AR (0 = centred, max = at wall)
freq_limit    = 10001 # upper frequency limit in Hz — model won't calculate, or account for anything above this
min_peak_db   = 15    # ignore peaks shorter than this (dB) — filters noise
min_peak_gap  = 200   # ignore peaks closer than this (Hz) — prevents double-counting

# PHYSICAL CONSTANTS  - some values are chosen from the ICA 2016 paper

rho = 1.225   # air density, kg/m³
c   = 343.0   # speed of sound, m/s

neck_radius    = 0.006                           # neck radius, m  (12 mm diameter)
neck_length    = 0.002                           # physical neck length, m
neck_eff       = neck_length + 1.7 * neck_radius # effective length with end corrections
neck_area      = np.pi * neck_radius**2          # cross-sectional area of the neck
neck_impedance = rho * c / neck_area             # characteristic impedance of the neck

duct_side      = 0.03                            # side length of the square main duct, m
duct_area      = duct_side**2
duct_impedance = rho * c / duct_area             # characteristic impedance of the duct

cavity_volume  = 1.68e-5   # cavity volume, m³ — held constant across all ARs

# Set up frequency and wave number arrays

freqs  = np.arange(1, freq_limit, 1, dtype=float)  # 1 Hz → freq_limit-1 Hz
k_vals = 2.0 * np.pi * freqs / c                   # wavenumber at each frequency


# ASPECT RATIO LIST

num_ars = int(round((ar_stop - ar_start) / ar_step)) + 1
ar_list = np.linspace(ar_start, ar_stop, num_ars)

print(f"Aspect ratios : {num_ars}  ({ar_start} to {ar_stop}, step {ar_step})")
print(f"Offsets per AR: {num_offsets}")
print(f"Total configs : {num_ars * num_offsets:,}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULT STORAGE
#
# Only store peak data — not the full TL arrays — to keep RAM reasonable
# Shape: (num_ars, num_offsets, 6)
# Columns: [ar, offset_fraction, p1_freq, p1_db, p2_freq, p2_db]
#
# p2 is the genuine offset-induced resonance, not just any second peak.
# We find it by searching in a physics-informed window:
#   [1.5 × baseline_p1,  4.0 × baseline_p1]
# This is tight enough to exclude high cavity harmonics but wide enough to
# catch the second resonance as it moves with changing geometry.
# ─────────────────────────────────────────────────────────────────────────────

results = np.zeros((num_ars, num_offsets, 6))
baseline_p1_freq = np.zeros(num_ars)   # first peak at zero offset, one per AR


# PHYSICS FUNCTIONS


def cavity_impedance(k, l1, l2, z_neck, z_cav):
    """
    Complex impedance at the neck using Equation 5 from Etaix et al. 2016.

    The neck splits the rectangular cavity into two sections of length l1 and l2.
    Each section is treated as a closed pipe. The impedances of both sections
    are combined with the neck impedance to give the total resonator impedance.

    k      : wavenumber array  (1/m)
    l1, l2 : cavity section lengths either side of the neck  (m)
    z_neck : characteristic impedance of the neck  (Pa·s/m³)
    z_cav  : characteristic impedance of the cavity cross-section
    """
    A = z_neck * np.sin(k * (l1 + l2)) * np.sin(k * neck_eff)
    B = z_cav  * np.cos(k * neck_eff)  * np.cos(k * l1) * np.cos(k * l2)
    C = z_neck * np.sin(k * (l1 + l2)) * np.cos(k * neck_eff)
    D = z_cav  * np.sin(k * neck_eff)  * np.cos(k * l1) * np.cos(k * l2)

    # Guard against division by zero at exact cavity resonance frequencies
    denom = C + D
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    return 1j * z_neck * (A - B) / denom


def compute_TL(Z):
    """
    Transmission loss in dB using the side-branch silencer formula (Eq. 6, Etaix 2016).
    TL = 20 log10 |1 + Z_duct / (2Z)|
    """
    return 20.0 * np.log10(np.abs(1.0 + duct_impedance / (2.0 * Z)))


def get_TL_for(ar, offset):
    """Recompute TL for a single (ar, offset) pair — used for plotting on demand."""
    cav_width  = np.cbrt(cavity_volume / ar)
    cav_length = cav_width * ar
    z_cav      = rho * c / cav_width**2
    l1 = cav_length / 2.0 - offset
    l2 = cav_length / 2.0 + offset
    Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
    return compute_TL(Z)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CALCULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

for i, ar in enumerate(ar_list):

    # Cavity geometry — volume is fixed, shape changes with AR
    cav_width  = np.cbrt(cavity_volume / ar)   # cross-section side length
    cav_length = cav_width * ar                # long axis length
    z_cav      = rho * c / cav_width**2        # cavity cross-section impedance

    # 101 offsets from 0 (neck centred) to L/2 (neck at end wall)
    offsets      = np.linspace(0.0, (cav_length / 2.0) - neck_radius, num_offsets)
    offset_fracs = offsets / ((cav_length / 2.0) - neck_radius)   # normalised 0 → 1

    for j, offset in enumerate(offsets):

        # Split the cavity into two sections either side of the neck
        l1 = cav_length / 2.0 - offset   # left section length
        l2 = cav_length / 2.0 + offset   # right section length

        Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
        TL = compute_TL(Z)

        # Detect peaks above the noise floor
        peak_idx, _ = find_peaks(TL, height=min_peak_db, distance=int(min_peak_gap))
        peak_freqs  = freqs[peak_idx]
        peak_dbs    = TL[peak_idx]

        # First peak — main Helmholtz resonance
        p1_freq = peak_freqs[0] if len(peak_freqs) >= 1 else 0.0
        p1_db   = peak_dbs[0]   if len(peak_freqs) >= 1 else 0.0

        # Store the zero-offset first peak as the baseline for this AR
        if j == 0:
            baseline_p1_freq[i] = p1_freq

        # Second peak — the offset-induced resonance we care about.
        # Only look between 1.5× and 4× the zero-offset first peak frequency.
        # This window excludes high-order cavity harmonics while still catching
        # the genuine second resonance as it shifts with geometry.
        p2_freq, p2_db = 0.0, 0.0
        if baseline_p1_freq[i] > 0 and len(peak_freqs) >= 2:
            window_lo = baseline_p1_freq[i] * 1.5
            window_hi = baseline_p1_freq[i] * 4.0
            in_window = (peak_freqs > window_lo) & (peak_freqs < window_hi)
            if in_window.any():
                idx       = np.argmax(in_window)   # lowest-frequency peak in window
                p2_freq   = peak_freqs[idx]
                p2_db     = peak_dbs[idx]

        results[i, j] = [ar, offset_fracs[j], p1_freq, p1_db, p2_freq, p2_db]

print("Calculations done.")

# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP GRIDS
# Shape (num_ars, num_offsets) — rows = AR, columns = offset fraction.
# Zeros replaced with NaN so imshow leaves those cells white.
# ─────────────────────────────────────────────────────────────────────────────

p2_freq_grid = results[:, :, 4].copy()
p2_db_grid   = results[:, :, 5].copy()
p2_freq_grid[p2_freq_grid == 0] = np.nan
p2_db_grid[p2_db_grid == 0]     = np.nan


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — HEATMAPS
#
# Left:  where does the second peak sit in frequency?
# Right: how strong is the second peak?
#
# X-axis = aspect ratio.  Y-axis = fractional neck offset.
# White = no second peak detected at that combination.
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmaps():
    fig, (ax_freq, ax_db) = plt.subplots(1, 2, figsize=(16, 7))

    fig.suptitle(
        "Parametric Sweep — Asymmetric Helmholtz Resonator\n"
        f"Cavity V = {cavity_volume:.2e} m³  |  "
        f"Neck r = {neck_radius*1000:.0f} mm,  l = {neck_length*1000:.0f} mm  |  "
        f"Duct {int(duct_side*1000)} mm square",
        fontsize=11
    )

    extent = [ar_start, ar_stop, 0.0, 1.0]  # [x_left, x_right, y_bottom, y_top]

    # Left plot — second peak frequency
    im1 = ax_freq.imshow(
        p2_freq_grid.T,        # transpose so Y = offset fraction, X = AR
        origin='lower',        # low offset at bottom
        aspect='auto',
        extent=extent,
        cmap='plasma',         # dark purple = low freq, yellow = high freq
        interpolation='nearest',
        vmin=np.nanpercentile(p2_freq_grid, 2),
        vmax=np.nanpercentile(p2_freq_grid, 98)
    )
    cb1 = fig.colorbar(im1, ax=ax_freq, fraction=0.046, pad=0.04)
    cb1.set_label("Second Peak Frequency (Hz)", fontsize=10)
    ax_freq.set_title("Where does the second peak sit?", fontweight='bold')
    ax_freq.set_xlabel("Cavity Aspect Ratio  L / W", fontsize=10)
    ax_freq.set_ylabel("Neck Offset Fraction  (0 = centre,  1 = end wall)", fontsize=10)
    ax_freq.text(0.02, 0.98, "White = no second peak detected",
                 transform=ax_freq.transAxes, fontsize=8, va='top', color='grey')

    # Right plot — second peak attenuation
    im2 = ax_db.imshow(
        p2_db_grid.T,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='viridis',        # dark purple = weak attenuation, yellow = strong
        interpolation='nearest',
        vmin=np.nanpercentile(p2_db_grid, 2),
        vmax=np.nanpercentile(p2_db_grid, 98)
    )
    cb2 = fig.colorbar(im2, ax=ax_db, fraction=0.046, pad=0.04)
    cb2.set_label("Second Peak Attenuation (dB)", fontsize=10)
    ax_db.set_title("How strong is the second peak?", fontweight='bold')
    ax_db.set_xlabel("Cavity Aspect Ratio  L / W", fontsize=10)
    ax_db.set_ylabel("Neck Offset Fraction  (0 = centre,  1 = end wall)", fontsize=10)
    ax_db.text(0.02, 0.98, "White = no second peak detected",
               transform=ax_db.transAxes, fontsize=8, va='top', color='grey')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — SUMMARY TRENDS
#
# A: First peak frequency vs AR
#    Because volume is fixed the first peak should stay roughly constant.
#    Any drift reveals how cavity wave propagation affects the result beyond
#    the lumped-element approximation.
#
# B: Second peak frequency vs AR at five discrete offset fractions
#    This is the core result — longer cavity pushes the second peak lower.
#
# C: Fraction of offsets that produce a detectable second peak vs AR
#    High values mean the second peak appears reliably for that cavity shape.
# ─────────────────────────────────────────────────────────────────────────────

def make_summary():
    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Sweep Summary — Key Trends", fontsize=13, fontweight='bold')

    # A — first peak vs AR
    p1_zero_offset = results[:, 0, 2]
    ax_a.plot(ar_list, p1_zero_offset, color='steelblue', lw=2,
              marker='o', markersize=3, label='First peak (centred neck)')
    ax_a.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_a.set_title(
        "A  —  First peak frequency vs aspect ratio\n"
        "Volume fixed → drift from constant shows cavity wave effects on the TMM",
        fontsize=10
    )
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=9)

    # B — second peak vs AR at several offsets
    sample_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_b = plt.get_cmap('cool')
    for n, frac in enumerate(sample_fracs):
        j       = int(round(frac * (num_offsets - 1)))
        p2_vals = results[:, j, 4]
        p2_clean = np.where(p2_vals > 0, p2_vals, np.nan)
        colour  = cmap_b(n / (len(sample_fracs) - 1))
        ax_b.plot(ar_list, p2_clean, color=colour, lw=1.8,
                  label=f'{frac:.0%} of max offset')
    ax_b.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_b.set_title(
        "B  —  Second peak frequency vs aspect ratio\n"
        "Higher offset and longer cavity both push the second peak to lower frequencies",
        fontsize=10
    )
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=9, ncol=2)

    # C — detection rate
    detection_pct = np.mean(results[:, :, 4] > 0, axis=1) * 100
    ax_c.fill_between(ar_list, detection_pct, alpha=0.35, color='coral')
    ax_c.plot(ar_list, detection_pct, color='firebrick', lw=2)
    ax_c.set_ylabel("% of neck positions with P2 detected", fontsize=10)
    ax_c.set_xlabel("Cavity Aspect Ratio  L / W", fontsize=10)
    ax_c.set_ylim(0, 105)
    ax_c.set_title(
        "C  —  How reliably does the second peak appear?\n"
        "100% = second peak shows up at every neck position for that cavity shape",
        fontsize=10
    )
    ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — PER-AR TL PLOTS
#
# Each subplot shows all 101 offset curves for one aspect ratio.
# Color goes cyan (centred neck) → magenta (neck at wall).
# TL is recomputed on the fly rather than stored — keeps RAM down.
# ─────────────────────────────────────────────────────────────────────────────

def make_ar_plots(which_ars=None):
    """
    which_ars : list of integer indices into ar_list.
                Pass None to plot everything.
    """
    if which_ars is None:
        which_ars = list(range(num_ars))

    n = len(which_ars)
    fig, axes = plt.subplots(n, 1, figsize=(14, 7 * n), squeeze=False)

    offset_cmap = plt.get_cmap('cool')
    offset_norm = Normalize(vmin=0, vmax=1)

    for row, i in enumerate(which_ars):
        ax = axes[row, 0]
        ar = ar_list[i]

        cav_width  = np.cbrt(cavity_volume / ar)
        cav_length = cav_width * ar
        offsets    = np.linspace(0.0, cav_length / 2.0, num_offsets)

        for j, offset in enumerate(offsets):
            frac   = j / (num_offsets - 1)   # 0 = centred neck, 1 = at wall
            colour = offset_cmap(frac)
            TL = get_TL_for(ar, offset)
            ax.plot(freqs, TL, color=colour, linewidth=0.7, alpha=0.75)

        # Colourbar explaining the cyan → magenta gradient
        sm = ScalarMappable(cmap=offset_cmap, norm=offset_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
        cbar.set_label("Neck offset\n(0 = centre, 1 = wall)", fontsize=8)

        ax.set_title(
            f"AR = {ar:.2f}  |  L = {cav_length*1000:.1f} mm  |  W = {cav_width*1000:.1f} mm",
            fontweight='bold', loc='left', fontsize=10
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Transmission Loss (dB)", fontsize=9)
        ax.set_xlim(1, freq_limit - 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linestyle='--')

    plt.subplots_adjust(hspace=0.45, left=0.07, right=0.92, top=0.97, bottom=0.03)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION
# Page 1 : heatmaps
# Page 2 : summary trends
# Page 3+: per-AR TL plots, 5 per page so each subplot is readable
# ─────────────────────────────────────────────────────────────────────────────

pdf_path = "Helmholtz_TMM_Final_Report.pdf"
per_page = 5   # number of AR subplots per page

print("\nGenerating PDF...")

with PdfPages(pdf_path) as pdf:

    print("  Page 1: heatmaps")
    fig = make_heatmaps()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    print("  Page 2: summary trends")
    fig = make_summary()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    print("  Pages 3+: per-AR TL plots")
    for batch_start in range(0, num_ars, per_page):
        batch = list(range(batch_start, min(batch_start + per_page, num_ars)))
        fig = make_ar_plots(which_ars=batch)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    AR {ar_list[batch[0]]:.2f} – {ar_list[batch[-1]]:.2f}")

    info = pdf.infodict()
    info['Title']   = 'Helmholtz Resonator TMM Parametric Sweep'
    info['Author']  = 'Svar Joshi'
    info['Subject'] = 'Asymmetric cavity — neck offset x aspect ratio'

print(f"\nSaved: {pdf_path}")

if platform.system() == 'Darwin':
    subprocess.call(('open', pdf_path))
elif platform.system() == 'Windows':
    os.startfile(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("INTERACTIVE MODE")
print("─" * 50)
print("  heatmap        — both heatmaps")
print("  heatmap freq   — frequency heatmap only")
print("  heatmap db     — attenuation heatmap only")
print("  summary        — trend summary")
print("  [number]       — TL plots for that AR  e.g. '3.5'")
print("  exit           — quit")
print("─" * 50)

while True:
    cmd = input("\n> ").strip().lower()

    if cmd == 'exit':
        break

    elif 'heatmap' in cmd:
        fig = make_heatmaps()
        if cmd == 'heatmap freq':
            fig.axes[1].set_visible(False)
            fig.axes[3].set_visible(False)
        elif cmd == 'heatmap db':
            fig.axes[0].set_visible(False)
            fig.axes[2].set_visible(False)
        plt.tight_layout()
        plt.show()

    elif cmd == 'summary':
        fig = make_summary()
        plt.show()

    else:
        try:
            target_ar = float(cmd)
            closest   = int(np.abs(ar_list - target_ar).argmin())
            print(f"Plotting AR = {ar_list[closest]:.2f}")
            fig = make_ar_plots(which_ars=[closest])
            plt.show()
        except ValueError:
            print("Didn't recognise that — try a number like '2.5', or 'heatmap', or 'exit'.")