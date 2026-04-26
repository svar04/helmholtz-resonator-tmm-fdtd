import matplotlib
matplotlib.use('TkAgg')   # force a real detached window — stops PyCharm SciView
                          # from intercepting plt.show() and rendering inline.
                          # If you still get inline plots, go to:
                          #   PyCharm → Settings → Tools → Python Scientific
                          #   and turn off "Show plots in tool window"
                          # If TkAgg isn't installed, try 'Qt5Agg' or 'wxAgg'.

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

ar_start  = 1.0
ar_stop   = 6.0   # capped at 6 — beyond this the cavity is physically unrealistic
                  # (13mm wide at AR=7) and mode detection above 8kHz becomes unreliable
ar_step   = 0.05

num_offsets  = 201   # 201 gives 0.5% offset resolution — fine enough, going higher
                     # costs compute time with no meaningful gain in extracted data
freq_limit   = 10001

min_peak_db  = 15    # from Etaix 2016 — below 15dB attenuation is acoustically
                     # marginal and wouldn't be used in a real silencer design
min_peak_gap = 200   # Hz — wide enough to avoid double-counting the skirt around
                     # a sharp peak, narrow enough to separate genuine adjacent modes

max_modes = 6        # at AR=6 high offset you see at most 5–6 peaks in 0–10kHz

# fixed values used when one IV is held constant across all sub-question plots.
# FIXED_AR = 4.0 puts us firmly in the regime where modes 1–4 are all detectable
# without being at either extreme of the parameter space.
# FIXED_OFFSET = 0.5 is the natural midpoint — not so small that offset-induced
# modes are absent, not so large that we're only looking at the extreme edge case.
# Using the centred case (0.0) would be useless here because most offset-induced
# modes don't exist at zero offset — that's the symmetric configuration.
FIXED_AR     = 4.0
FIXED_OFFSET = 0.5


#region Constants and Array Setup

rho = 1.225
c   = 343.0

neck_radius    = 0.006
neck_length    = 0.002
neck_eff       = neck_length + 1.7 * neck_radius
neck_area      = np.pi * neck_radius**2
neck_impedance = rho * c / neck_area

duct_side      = 0.03
duct_area      = duct_side**2
duct_impedance = rho * c / duct_area

cavity_volume  = 1.68e-5   # m³ — held constant across all ARs

freqs  = np.arange(1, freq_limit, 1, dtype=float)
k_vals = 2.0 * np.pi * freqs / c

num_ars = int(round((ar_stop - ar_start) / ar_step)) + 1
ar_list = np.linspace(ar_start, ar_stop, num_ars)

print(f"Aspect ratios : {num_ars}  ({ar_start} to {ar_stop}, step {ar_step})")
print(f"Offsets per AR: {num_offsets}")
print(f"Total configs : {num_ars * num_offsets:,}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULT STORAGE
#
# Two 3D arrays — shape (num_ars, num_offsets, max_modes)
# all_freqs[i, j, m] = frequency of mode m+1 at AR i, offset j  (0 if not present)
# all_dbs  [i, j, m] = TL depth   of mode m+1 at AR i, offset j  (0 if not present)
#
# Mode indexing is 0-based internally: m=0 is mode 1, m=1 is mode 2, etc.
# Zeros mean that mode was not detected for that configuration.
# ─────────────────────────────────────────────────────────────────────────────

all_freqs        = np.zeros((num_ars, num_offsets, max_modes))
all_dbs          = np.zeros((num_ars, num_offsets, max_modes))
offset_fracs_all = np.zeros((num_ars, num_offsets))   # stored so plots don't
                                                       # need to recompute per-AR geometry

#endregion

#region PHYSICS FUNCTIONS

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

    denom = C + D
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    return 1j * z_neck * (A - B) / denom


def compute_TL(Z):
    """
    Transmission loss in dB — Equation 6, Etaix 2016.
    TL = 20 log10 |1 + Z_duct / (2Z)|
    """
    return 20.0 * np.log10(np.abs(1.0 + duct_impedance / (2.0 * Z)))


def get_TL_for(ar, offset):
    """Recompute TL for a single (ar, offset) pair — used for TL overlay plots."""
    cav_width  = np.cbrt(cavity_volume / ar)
    cav_length = cav_width * ar
    z_cav      = rho * c / cav_width**2
    l1 = (cav_length / 2.0) - offset
    l2 = (cav_length / 2.0) + offset
    Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
    return compute_TL(Z)

#endregion

#region MAIN CALCULATION LOOP

for i, ar in enumerate(ar_list):

    cav_width  = np.cbrt(cavity_volume / ar)
    cav_length = cav_width * ar
    z_cav      = rho * c / cav_width**2

    # max offset: neck centre must stay at least one neck radius from the
    # end wall so the neck doesn't overhang the cavity edge
    max_offset   = (cav_length / 2.0) - neck_radius
    offsets      = np.linspace(0.0, max_offset, num_offsets)
    offset_fracs = offsets / max_offset   # normalised 0 → 1

    offset_fracs_all[i, :] = offset_fracs

    for j, offset in enumerate(offsets):

        l1 = (cav_length / 2.0) - offset
        l2 = (cav_length / 2.0) + offset

        Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
        TL = compute_TL(Z)

        peak_idx, _ = find_peaks(TL, height=min_peak_db, distance=int(min_peak_gap))
        peak_freqs  = freqs[peak_idx]
        peak_dbs    = TL[peak_idx]

        # sort by frequency — find_peaks is usually sorted but being explicit
        if len(peak_freqs) > 0:
            order      = np.argsort(peak_freqs)
            peak_freqs = peak_freqs[order]
            peak_dbs   = peak_dbs[order]

        # store up to max_modes peaks — zero padding handles the rest
        n = min(len(peak_freqs), max_modes)
        all_freqs[i, j, :n] = peak_freqs[:n]
        all_dbs[i, j, :n]   = peak_dbs[:n]

print("Calculations done.")

# how many modes actually appear anywhere in the sweep — drives how many
# pages we produce so we never generate empty panels
n_modes_present = int(np.max(np.sum(all_freqs > 0, axis=2)))
print(f"Max modes detected in any single configuration: {n_modes_present}")

#endregion

#region Plotting

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING CONSTANTS
#
# fixed_ar_idx  — index into ar_list closest to FIXED_AR
# fixed_off_idx — index into the offset array closest to FIXED_OFFSET fraction
#
# Same fixed slice used across all sub-question graphs so results are
# directly comparable between SQ1, SQ2, and SQ3.
# ─────────────────────────────────────────────────────────────────────────────

fixed_ar_idx  = int(np.abs(ar_list - FIXED_AR).argmin())
fixed_off_idx = int(round(FIXED_OFFSET * (num_offsets - 1)))

print(f"Fixed AR for offset sweeps  : {ar_list[fixed_ar_idx]:.2f}")
print(f"Fixed offset for AR sweeps  : {FIXED_OFFSET:.0%}  (index {fixed_off_idx})")

mode_colours = plt.get_cmap('tab10')
extent_2d    = [ar_start, ar_stop, 0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# SUB-QUESTION 1 — MODE PRESENCE
#
# 3n graphs total where n = n_modes_present:
#   A: is mode m present vs AR            (offset fixed at FIXED_OFFSET)
#   B: is mode m present vs offset        (AR fixed at FIXED_AR)
#   C: 2D presence heatmap across (AR, offset) space
#
# One page per mode — n pages total for SQ1.
# ─────────────────────────────────────────────────────────────────────────────

def make_sq1_presence_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sub-Question 1 — Mode {m+1} Presence\n"
        f"Fixed AR = {ar_list[fixed_ar_idx]:.2f}  |  Fixed offset = {FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    # Panel A — presence vs AR
    presence_vs_ar = (all_freqs[:, fixed_off_idx, m] > 0).astype(float)
    axes[0].step(ar_list, presence_vs_ar, where='mid', color=mode_colours(m), lw=2)
    axes[0].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[0].set_ylabel("Mode present  (1 = yes,  0 = no)", fontsize=9)
    axes[0].set_title(f"A — vs AR  (offset fixed = {FIXED_OFFSET:.0%})", fontsize=9)
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].grid(True, alpha=0.3)

    # Panel B — presence vs offset fraction
    presence_vs_offset = (all_freqs[fixed_ar_idx, :, m] > 0).astype(float)
    axes[1].step(offset_fracs_all[fixed_ar_idx, :], presence_vs_offset,
                 where='mid', color=mode_colours(m), lw=2)
    axes[1].set_xlabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[1].set_ylabel("Mode present  (1 = yes,  0 = no)", fontsize=9)
    axes[1].set_title(f"B — vs offset  (AR fixed = {ar_list[fixed_ar_idx]:.2f})", fontsize=9)
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].grid(True, alpha=0.3)

    # Panel C — 2D presence heatmap
    presence_grid = (all_freqs[:, :, m] > 0).astype(float)
    im = axes[2].imshow(
        presence_grid.T, origin='lower', aspect='auto',
        extent=extent_2d, cmap='RdYlGn',
        interpolation='nearest', vmin=0, vmax=1
    )
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                 ticks=[0, 1]).set_ticklabels(['absent', 'present'])
    axes[2].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[2].set_ylabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[2].set_title("C — 2D presence map", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SUB-QUESTION 2 — MODE FREQUENCY
#
# 3n graphs total:
#   A: frequency vs AR        (offset fixed at FIXED_OFFSET)
#   B: frequency vs offset    (AR fixed at FIXED_AR)
#   C: 2D frequency heatmap   (one per mode)
#
# One page per mode — n pages total for SQ2.
# ─────────────────────────────────────────────────────────────────────────────

def make_sq2_frequency_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sub-Question 2 — Mode {m+1} Frequency\n"
        f"Fixed AR = {ar_list[fixed_ar_idx]:.2f}  |  Fixed offset = {FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    # Panel A — frequency vs AR
    freq_vs_ar = all_freqs[:, fixed_off_idx, m].copy()
    freq_vs_ar = np.where(freq_vs_ar > 0, freq_vs_ar, np.nan)
    axes[0].plot(ar_list, freq_vs_ar, color=mode_colours(m), lw=2,
                 marker='o', markersize=2.5)
    axes[0].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[0].set_ylabel("Frequency (Hz)", fontsize=9)
    axes[0].set_title(f"A — vs AR  (offset fixed = {FIXED_OFFSET:.0%})", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel B — frequency vs offset
    freq_vs_off = all_freqs[fixed_ar_idx, :, m].copy()
    freq_vs_off = np.where(freq_vs_off > 0, freq_vs_off, np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], freq_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[1].set_xlabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=9)
    axes[1].set_title(f"B — vs offset  (AR fixed = {ar_list[fixed_ar_idx]:.2f})", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel C — 2D frequency heatmap
    freq_grid = all_freqs[:, :, m].copy()
    freq_grid[freq_grid == 0] = np.nan
    im = axes[2].imshow(
        freq_grid.T, origin='lower', aspect='auto',
        extent=extent_2d, cmap='plasma',
        interpolation='nearest',
        vmin=np.nanpercentile(freq_grid, 2),
        vmax=np.nanpercentile(freq_grid, 98)
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("Frequency (Hz)", fontsize=9)
    axes[2].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[2].set_ylabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[2].set_title("C — 2D frequency map  (white = not detected)", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SUB-QUESTION 3 — MODE TL DEPTH
#
# Exact same structure as SQ2 but plotting TL depth in dB instead of frequency.
# 3n graphs — one page per mode.
# ─────────────────────────────────────────────────────────────────────────────

def make_sq3_depth_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Sub-Question 3 — Mode {m+1} TL Depth\n"
        f"Fixed AR = {ar_list[fixed_ar_idx]:.2f}  |  Fixed offset = {FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    # Panel A — TL depth vs AR
    db_vs_ar = all_dbs[:, fixed_off_idx, m].copy()
    db_vs_ar = np.where(db_vs_ar > 0, db_vs_ar, np.nan)
    axes[0].plot(ar_list, db_vs_ar, color=mode_colours(m), lw=2,
                 marker='o', markersize=2.5)
    axes[0].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[0].set_ylabel("TL Depth (dB)", fontsize=9)
    axes[0].set_title(f"A — vs AR  (offset fixed = {FIXED_OFFSET:.0%})", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel B — TL depth vs offset
    db_vs_off = all_dbs[fixed_ar_idx, :, m].copy()
    db_vs_off = np.where(db_vs_off > 0, db_vs_off, np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], db_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[1].set_xlabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[1].set_ylabel("TL Depth (dB)", fontsize=9)
    axes[1].set_title(f"B — vs offset  (AR fixed = {ar_list[fixed_ar_idx]:.2f})", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel C — 2D TL depth heatmap
    db_grid = all_dbs[:, :, m].copy()
    db_grid[db_grid == 0] = np.nan
    im = axes[2].imshow(
        db_grid.T, origin='lower', aspect='auto',
        extent=extent_2d, cmap='viridis',
        interpolation='nearest',
        vmin=np.nanpercentile(db_grid, 2),
        vmax=np.nanpercentile(db_grid, 98)
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("TL Depth (dB)", fontsize=9)
    axes[2].set_xlabel("Aspect Ratio  L / W", fontsize=9)
    axes[2].set_ylabel("Offset Fraction  (0 = centre,  1 = wall)", fontsize=9)
    axes[2].set_title("C — 2D TL depth map  (white = not detected)", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RAW TL OVERLAY PLOTS — per AR, all offsets overlaid
#
# Raw data verification — 201 curves per subplot, one per offset position.
# Colour goes cyan (centred neck, offset=0) → magenta (neck at wall, offset=1).
# At low AR curves overlap visually because the physical offset range is small
# and moving the neck barely changes the cavity sections — this is real physics,
# not a bug. The max offset shown in the title confirms how wide the sweep is.
# ─────────────────────────────────────────────────────────────────────────────

def make_ar_plots(which_ars=None):
    if which_ars is None:
        which_ars = list(range(num_ars))

    n   = len(which_ars)
    fig, axes = plt.subplots(n, 1, figsize=(14, 7 * n), squeeze=False)

    offset_cmap = plt.get_cmap('cool')
    offset_norm = Normalize(vmin=0, vmax=1)

    for row, i in enumerate(which_ars):
        ax = axes[row, 0]
        ar = ar_list[i]

        cav_width  = np.cbrt(cavity_volume / ar)
        cav_length = cav_width * ar
        max_offset = (cav_length / 2.0) - neck_radius
        offsets    = np.linspace(0.0, max_offset, num_offsets)

        for j, offset in enumerate(offsets):
            frac   = j / (num_offsets - 1)
            colour = offset_cmap(frac)
            TL     = get_TL_for(ar, offset)
            ax.plot(freqs, TL, color=colour, linewidth=0.7, alpha=0.75)

        sm = ScalarMappable(cmap=offset_cmap, norm=offset_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
        cbar.set_label("Neck offset\n(0 = centre, 1 = wall)", fontsize=8)

        ax.set_title(
            f"AR = {ar:.2f}  |  L = {cav_length*1000:.1f} mm  |  "
            f"W = {cav_width*1000:.1f} mm  |  Max offset = {max_offset*1000:.1f} mm",
            fontweight='bold', loc='left', fontsize=9
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Transmission Loss (dB)", fontsize=9)
        ax.set_xlim(1, freq_limit - 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linestyle='--')

    plt.subplots_adjust(hspace=0.45, left=0.07, right=0.92, top=0.97, bottom=0.03)
    return fig

#endregion

#region Generation and Interactivity

# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION
#
# SQ1 pages  (n_modes_present pages)
# SQ2 pages  (n_modes_present pages)
# SQ3 pages  (n_modes_present pages)
# Raw TL overlay plots  (3 ARs per page so each subplot is tall enough to read)
# ─────────────────────────────────────────────────────────────────────────────

pdf_path = "Helmholtz_TMM_Final_Report.pdf"
per_page = 3

print("\nGenerating PDF...")

with PdfPages(pdf_path) as pdf:

    print("  Sub-question 1 — mode presence")
    for m in range(n_modes_present):
        fig = make_sq1_presence_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1} done")

    print("  Sub-question 2 — mode frequency")
    for m in range(n_modes_present):
        fig = make_sq2_frequency_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1} done")

    print("  Sub-question 3 — mode TL depth")
    for m in range(n_modes_present):
        fig = make_sq3_depth_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1} done")

    print("  Raw TL overlay plots")
    for batch_start in range(0, num_ars, per_page):
        batch = list(range(batch_start, min(batch_start + per_page, num_ars)))
        fig   = make_ar_plots(which_ars=batch)
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
#
# All plt.show() calls open a real detached matplotlib window because of the
# TkAgg backend set at the top — you can zoom indefinitely without losing
# quality because it renders as a vector figure, not a raster screenshot.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("INTERACTIVE MODE")
print("─" * 50)
print("  sq1 [mode]   — mode presence plots  e.g. 'sq1 2'")
print("  sq2 [mode]   — mode frequency plots")
print("  sq3 [mode]   — mode TL depth plots")
print("  [number]     — raw TL overlay for that AR  e.g. '3.5'")
print("  exit         — quit")
print("─" * 50)

while True:
    cmd = input("\n> ").strip().lower()

    if cmd == 'exit':
        break

    elif cmd.startswith('sq1'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq1_presence_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq1 [mode number 1–{n_modes_present}]")

    elif cmd.startswith('sq2'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq2_frequency_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq2 [mode number 1–{n_modes_present}]")

    elif cmd.startswith('sq3'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq3_depth_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq3 [mode number 1–{n_modes_present}]")

    else:
        try:
            target_ar = float(cmd)
            closest   = int(np.abs(ar_list - target_ar).argmin())
            print(f"Plotting AR = {ar_list[closest]:.2f}")
            fig = make_ar_plots(which_ars=[closest])
            plt.show()
        except ValueError:
            print("Didn't recognise that — try 'sq1 2', a number like '3.5', or 'exit'.")

#endregion