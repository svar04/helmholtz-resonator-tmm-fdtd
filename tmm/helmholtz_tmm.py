import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import subprocess
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
matplotlib.use('TkAgg')

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# Must match FDTD exactly.
# ═══════════════════════════════════════════════════════════════════════════════

rho = 1.21     # [kg/m³] — standard dry air at 20°C
c   = 343.0    # [m/s]


# ═══════════════════════════════════════════════════════════════════════════════
# USER SWEEP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

ar_start  = 1.0
ar_stop   = 6.0
ar_step   = 0.05

num_offsets  = 201    # 0.5% offset resolution
freq_limit   = 10001  # [Hz] — sweep 1 to 10000 Hz

min_peak_db  = 15     # [dB] — Etaix threshold: below this is acoustically marginal
min_peak_gap = 200    # [Hz] — minimum separation between detected peaks

max_modes = 16         # max peaks stored per configuration

# Fixed values for 1D sweep plots (one IV swept, other held constant).
# AR=4.0: well-resolved cavity, shows 3-4 modes, not at either extreme.
# offset=0.5: clearly asymmetric, offset-induced modes present, not degenerate.
FIXED_AR     = 4.0
FIXED_OFFSET = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# NECK GEOMETRY
#
# Cylindrical neck — 3D.  Must match FDTD equivalence.
# FDTD uses neck_width_2D = neck_area / duct_height to preserve acoustic mass.
# The TMM neck_impedance = rho*c / neck_area is the reference.
# The FDTD neck_impedance = rho*c / (neck_width_2D * duct_height)
#                         = rho*c / neck_area  ← identical, by construction.
# ═══════════════════════════════════════════════════════════════════════════════

neck_width   = 0.050   # [m]
neck_length   = 0.050  # [m]

neck_radius = neck_width / 2.0
neck_eff     = neck_length + 1.7 * (neck_width / 2.0)  # [m] — effective length with end corrections
neck_area    = neck_width ** 2   # [m²]
neck_impedance = rho * c / neck_area  # [Pa·s/m³]


# ═══════════════════════════════════════════════════════════════════════════════
# DUCT GEOMETRY
#
# duct_height = 0.0113m — chosen so neck_width_2D = neck_area/duct_height
# lands on exactly 2 cells at dx=5mm.  Must match FDTD.
#
# duct_area = duct_height^2  — square cross-section, same assumption as
# z_cav = rho*c/cav_width^2 in the cavity impedance formula.
# duct_impedance drives the TL formula (Eq. 6 Etaix).
# ═══════════════════════════════════════════════════════════════════════════════


duct_height    = 0.300  # [m]
duct_area      = duct_height ** 2   # [m²] — square cross-section
duct_impedance = rho * c / duct_area   # [Pa·s/m³]


# ═══════════════════════════════════════════════════════════════════════════════
# CAVITY GEOMETRY
#
# Volume held constant.  AR = cav_length / cav_width.
# cav_width = cbrt(volume / AR) from volume = cav_width^3 * AR.
# z_cav = rho*c / cav_width^2 — impedance of square cavity cross-section.
# ═══════════════════════════════════════════════════════════════════════════════

cavity_volume = 0.008

# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY ARRAY AND WAVENUMBERS
# ═══════════════════════════════════════════════════════════════════════════════

freqs  = np.arange(1, freq_limit, 1, dtype=float)   # 1 Hz resolution
k_vals = 2.0 * np.pi * freqs / c


# ═══════════════════════════════════════════════════════════════════════════════
# AR LIST AND RESULT ARRAYS
# ═══════════════════════════════════════════════════════════════════════════════

num_ars = int(round((ar_stop - ar_start) / ar_step)) + 1
ar_list = np.linspace(ar_start, ar_stop, num_ars)

# Validate AR range against FDTD resolution limits
dx = 0.005  # must match FDTD dx for the validation prints below
for ar_check in [ar_start, ar_stop]:
    cw = np.cbrt(cavity_volume / ar_check)
    cl = cw * ar_check
    print(f"AR={ar_check:.1f}: cav {cl*1000:.1f}mm x {cw*1000:.1f}mm  "
          f"→ {int(round(cl/dx))} x {int(round(cw/dx))} cells at dx={dx*1000:.0f}mm")

print(f"\nAspect ratios : {num_ars}  ({ar_start} to {ar_stop}, step {ar_step})")
print(f"Offsets per AR: {num_offsets}")
print(f"Total configs : {num_ars * num_offsets:,}")
print(f"Neck eff len:   {neck_eff*1000:.3f}mm")
print(f"Neck impedance: {neck_impedance:.2f} Pa·s/m³")
print(f"Duct impedance: {duct_impedance:.2f} Pa·s/m³")
print(f"Neck/duct Z ratio: {neck_impedance/duct_impedance:.1f}")

# 3D result arrays — zeros mean mode not detected
all_freqs        = np.zeros((num_ars, num_offsets, max_modes))
all_dbs          = np.zeros((num_ars, num_offsets, max_modes))
offset_fracs_all = np.zeros((num_ars, num_offsets))


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cavity_impedance(k, l1, l2, z_neck, z_cav):
    """
    Complex impedance at the neck — Equation 5, Etaix et al. 2016.

    The asymmetric neck position splits the cavity into two closed-pipe
    sections of length l1 (shorter) and l2 (longer).  Their impedances
    combine with the neck inertance to give the total resonator impedance.

    k      : wavenumber array [1/m]
    l1, l2 : cavity section lengths [m]  (l1 + l2 = cav_length)
    z_neck : neck characteristic impedance [Pa·s/m³]
    z_cav  : cavity cross-section impedance [Pa·s/m³]
    """
    A = z_neck * np.sin(k * (l1 + l2)) * np.sin(k * neck_eff)
    B = z_cav  * np.cos(k * neck_eff)  * np.cos(k * l1) * np.cos(k * l2)
    C = z_neck * np.sin(k * (l1 + l2)) * np.cos(k * neck_eff)
    D = z_cav  * np.sin(k * neck_eff)  * np.cos(k * l1) * np.cos(k * l2)

    denom = C + D
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)   # avoid divide-by-zero at anti-resonance

    return 1j * z_neck * (A - B) / denom


def compute_TL(Z):
    """
    Transmission loss [dB] — Equation 6, Etaix 2016.
    TL = 20 log10 |1 + Z_duct / (2Z)|
    Higher TL = more attenuation at that frequency.
    """
    return 20.0 * np.log10(np.abs(1.0 + duct_impedance / (2.0 * Z)))


def get_TL_for(ar, offset):
    """
    Recompute full TL spectrum for one (ar, offset) pair.
    Used for raw TL overlay plots without re-indexing the stored arrays.
    """
    cav_width  = np.cbrt(cavity_volume / ar)
    cav_length = cav_width * ar
    z_cav      = rho * c / cav_width ** 2
    l1 = (cav_length / 2.0) - offset   # shorter section
    l2 = (cav_length / 2.0) + offset   # longer section
    Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
    return compute_TL(Z)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CALCULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

for i, ar in enumerate(ar_list):

    cav_width  = np.cbrt(cavity_volume / ar)
    cav_length = cav_width * ar
    z_cav      = rho * c / cav_width ** 2

    # Max offset: neck centre must stay >= neck_radius from either end wall
    # so the neck never overhangs the cavity boundary.
    max_offset   = (cav_length / 2.0) - neck_radius
    offsets      = np.linspace(0.0, max_offset, num_offsets)
    offset_fracs = offsets / max_offset   # normalised 0→1

    offset_fracs_all[i, :] = offset_fracs

    for j, offset in enumerate(offsets):

        l1 = (cav_length / 2.0) - offset
        l2 = (cav_length / 2.0) + offset

        Z  = cavity_impedance(k_vals, l1, l2, neck_impedance, z_cav)
        TL = compute_TL(Z)

        peak_idx, _ = find_peaks(TL, height=min_peak_db, distance=int(min_peak_gap))
        peak_freqs  = freqs[peak_idx]
        peak_dbs    = TL[peak_idx]

        if len(peak_freqs) > 0:
            order      = np.argsort(peak_freqs)
            peak_freqs = peak_freqs[order]
            peak_dbs   = peak_dbs[order]

        n = min(len(peak_freqs), max_modes)
        all_freqs[i, j, :n] = peak_freqs[:n]
        all_dbs[i, j, :n]   = peak_dbs[:n]

print("\nCalculations done.")
n_modes_present = int(np.max(np.sum(all_freqs > 0, axis=2)))
print(f"Max modes detected in any single configuration: {n_modes_present}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

fixed_ar_idx  = int(np.abs(ar_list - FIXED_AR).argmin())
fixed_off_idx = int(round(FIXED_OFFSET * (num_offsets - 1)))

print(f"Fixed AR for offset sweeps  : {ar_list[fixed_ar_idx]:.2f}")
print(f"Fixed offset for AR sweeps  : {FIXED_OFFSET:.0%}  (index {fixed_off_idx})")

mode_colours = plt.get_cmap('tab10')
extent_2d    = [ar_start, ar_stop, 0.0, 1.0]


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS — SQ1 / SQ2 / SQ3
# ═══════════════════════════════════════════════════════════════════════════════

def make_sq1_presence_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"SQ1 — Mode {m+1} Presence\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    presence_vs_ar = (all_freqs[:, fixed_off_idx, m] > 0).astype(float)
    axes[0].step(ar_list, presence_vs_ar, where='mid', color=mode_colours(m), lw=2)
    axes[0].set_xlabel("Aspect Ratio  L/W")
    axes[0].set_ylabel("Mode present  (1=yes, 0=no)")
    axes[0].set_title(f"A — vs AR  (offset fixed={FIXED_OFFSET:.0%})")
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].grid(True, alpha=0.3)

    presence_vs_offset = (all_freqs[fixed_ar_idx, :, m] > 0).astype(float)
    axes[1].step(offset_fracs_all[fixed_ar_idx, :], presence_vs_offset,
                 where='mid', color=mode_colours(m), lw=2)
    axes[1].set_xlabel("Offset Fraction  (0=centre, 1=wall)")
    axes[1].set_ylabel("Mode present  (1=yes, 0=no)")
    axes[1].set_title(f"B — vs offset  (AR fixed={ar_list[fixed_ar_idx]:.2f})")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].grid(True, alpha=0.3)

    presence_grid = (all_freqs[:, :, m] > 0).astype(float)
    im = axes[2].imshow(
        presence_grid.T, origin='lower', aspect='auto',
        extent=extent_2d, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1
    )
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                 ticks=[0, 1]).set_ticklabels(['absent', 'present'])
    axes[2].set_xlabel("Aspect Ratio  L/W")
    axes[2].set_ylabel("Offset Fraction")
    axes[2].set_title("C — 2D presence map")

    plt.tight_layout()
    return fig


def make_sq2_frequency_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"SQ2 — Mode {m+1} Frequency\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    freq_vs_ar = np.where(all_freqs[:, fixed_off_idx, m] > 0,
                          all_freqs[:, fixed_off_idx, m], np.nan)
    axes[0].plot(ar_list, freq_vs_ar, color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[0].set_xlabel("Aspect Ratio  L/W")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(f"A — vs AR  (offset fixed={FIXED_OFFSET:.0%})")
    axes[0].grid(True, alpha=0.3)

    freq_vs_off = np.where(all_freqs[fixed_ar_idx, :, m] > 0,
                           all_freqs[fixed_ar_idx, :, m], np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], freq_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[1].set_xlabel("Offset Fraction  (0=centre, 1=wall)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"B — vs offset  (AR fixed={ar_list[fixed_ar_idx]:.2f})")
    axes[1].grid(True, alpha=0.3)

    freq_grid = all_freqs[:, :, m].copy()
    freq_grid[freq_grid == 0] = np.nan
    im = axes[2].imshow(
        freq_grid.T, origin='lower', aspect='auto', extent=extent_2d,
        cmap='plasma', interpolation='nearest',
        vmin=np.nanpercentile(freq_grid, 2), vmax=np.nanpercentile(freq_grid, 98)
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("Frequency (Hz)")
    axes[2].set_xlabel("Aspect Ratio  L/W")
    axes[2].set_ylabel("Offset Fraction")
    axes[2].set_title("C — 2D frequency map  (white=not detected)")

    plt.tight_layout()
    return fig


def make_sq3_depth_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"SQ3 — Mode {m+1} TL Depth\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}",
        fontsize=11, fontweight='bold'
    )

    db_vs_ar = np.where(all_dbs[:, fixed_off_idx, m] > 0,
                        all_dbs[:, fixed_off_idx, m], np.nan)
    axes[0].plot(ar_list, db_vs_ar, color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[0].set_xlabel("Aspect Ratio  L/W")
    axes[0].set_ylabel("TL Depth (dB)")
    axes[0].set_title(f"A — vs AR  (offset fixed={FIXED_OFFSET:.0%})")
    axes[0].grid(True, alpha=0.3)

    db_vs_off = np.where(all_dbs[fixed_ar_idx, :, m] > 0,
                         all_dbs[fixed_ar_idx, :, m], np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], db_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=2.5)
    axes[1].set_xlabel("Offset Fraction  (0=centre, 1=wall)")
    axes[1].set_ylabel("TL Depth (dB)")
    axes[1].set_title(f"B — vs offset  (AR fixed={ar_list[fixed_ar_idx]:.2f})")
    axes[1].grid(True, alpha=0.3)

    db_grid = all_dbs[:, :, m].copy()
    db_grid[db_grid == 0] = np.nan
    im = axes[2].imshow(
        db_grid.T, origin='lower', aspect='auto', extent=extent_2d,
        cmap='viridis', interpolation='nearest',
        vmin=np.nanpercentile(db_grid, 2), vmax=np.nanpercentile(db_grid, 98)
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("TL Depth (dB)")
    axes[2].set_xlabel("Aspect Ratio  L/W")
    axes[2].set_ylabel("Offset Fraction")
    axes[2].set_title("C — 2D TL depth map  (white=not detected)")

    plt.tight_layout()
    return fig


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
        cbar.set_label("Neck offset\n(0=centre, 1=wall)", fontsize=8)

        ax.set_title(
            f"AR={ar:.2f}  |  L={cav_length*1000:.1f}mm  |  "
            f"W={cav_width*1000:.1f}mm  |  max offset={max_offset*1000:.1f}mm",
            fontweight='bold', loc='left', fontsize=9
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Transmission Loss (dB)")
        ax.set_xlim(1, freq_limit - 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linestyle='--')

    plt.subplots_adjust(hspace=0.45, left=0.07, right=0.92, top=0.97, bottom=0.03)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

pdf_path = "Helmholtz_TMM_Final_Report.pdf"
per_page = 3

print("\nGenerating PDF...")

with PdfPages(pdf_path) as pdf:

    print("  SQ1 — mode presence")
    for m in range(n_modes_present):
        fig = make_sq1_presence_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1}")

    print("  SQ2 — mode frequency")
    for m in range(n_modes_present):
        fig = make_sq2_frequency_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1}")

    print("  SQ3 — mode TL depth")
    for m in range(n_modes_present):
        fig = make_sq3_depth_page(m)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"    Mode {m+1}")

    print("  Raw TL overlays")
    for batch_start in range(0, num_ars, per_page):
        batch = list(range(batch_start, min(batch_start + per_page, num_ars)))
        fig   = make_ar_plots(which_ars=batch)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    AR {ar_list[batch[0]]:.2f}–{ar_list[batch[-1]]:.2f}")

    info = pdf.infodict()
    info['Title']   = 'Helmholtz Resonator TMM Parametric Sweep'
    info['Author']  = 'Svar Joshi'
    info['Subject'] = 'Asymmetric cavity — neck offset x aspect ratio'

print(f"\nSaved: {pdf_path}")

if platform.system() == 'Darwin':
    subprocess.call(('open', pdf_path))
elif platform.system() == 'Windows':
    os.startfile(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("INTERACTIVE MODE")
print("─" * 50)
print("  sq1 [mode]   — mode presence plots   e.g. sq1 2")
print("  sq2 [mode]   — mode frequency plots")
print("  sq3 [mode]   — mode TL depth plots")
print("  [number]     — raw TL overlay for AR  e.g. 3.5")
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
            print(f"  Usage: sq1 [1–{n_modes_present}]")

    elif cmd.startswith('sq2'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq2_frequency_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq2 [1–{n_modes_present}]")

    elif cmd.startswith('sq3'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq3_depth_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq3 [1–{n_modes_present}]")

    else:
        try:
            target_ar = float(cmd)
            closest   = int(np.abs(ar_list - target_ar).argmin())
            print(f"Plotting AR = {ar_list[closest]:.2f}")
            fig = make_ar_plots(which_ars=[closest])
            plt.show()
        except ValueError:
            print("Unrecognised — try 'sq1 2', '3.5', or 'exit'.")