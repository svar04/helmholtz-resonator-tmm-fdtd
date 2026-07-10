import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import subprocess
import time
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
matplotlib.use('TkAgg')


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS  —  must match TMM exactly
# ═══════════════════════════════════════════════════════════════════════════════

c   = 343.0    # [m/s]
rho = 1.21     # [kg/m³]


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP RESOLUTION
#
# SWEEP_MODE = 'coarse'    →   9 ARs ×  21 offsets =   189 FDTD runs  (~16 min)
# SWEEP_MODE = 'medium'    →  17 ARs ×  41 offsets =   697 FDTD runs  (~60 min)
# SWEEP_MODE = 'research'  →  33 ARs ×  81 offsets =  2673 FDTD runs  (~4 hrs)
# ═══════════════════════════════════════════════════════════════════════════════

SWEEP_MODE = 'coarse'   # <- change to 'medium' or 'research' for final runs

if SWEEP_MODE == 'coarse':
    num_ars     = 9
    num_offsets = 21
elif SWEEP_MODE == 'medium':
    num_ars     = 17
    num_offsets = 41
elif SWEEP_MODE == 'research':
    num_ars     = 33
    num_offsets = 81
else:
    raise ValueError(f"Unknown SWEEP_MODE '{SWEEP_MODE}' — use 'coarse', 'medium', or 'research'")


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK DETECTION  —  must match TMM exactly
# ═══════════════════════════════════════════════════════════════════════════════

min_peak_db  = 15      # [dB]
min_peak_gap = 200     # [Hz]
max_modes    = 20      # higher than TMM to capture extra 2D modes
f_max        = 10000.0

FIXED_AR     = 4.0
FIXED_OFFSET = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# FDTD SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

dx        = 0.002                          # [m] — 2 mm
dt        = 0.4 * dx / (c * np.sqrt(2))   # CFL-stable timestep [s]
T_sim     = 0.10                           # [s] — 100 ms → 10 Hz freq resolution
N_steps   = int(T_sim / dt)
pml_cells = 10

pulse_width = 1.5 / f_max
pulse_delay = 6.0 * pulse_width


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY  —  must match TMM exactly
#
# SLOT neck: neck_area_3D = neck_width_2D * duct_height
# Matches TMM: neck_area = neck_width * duct_height
# Impedances are identical by construction.
# ═══════════════════════════════════════════════════════════════════════════════

duct_height   = 0.050   # [m]
duct_width    = 0.050   # [m]
duct_length   = 0.400   # [m]
cavity_volume = 30e-6   # [m³]  30 ml

neck_width_2D = 0.010   # [m]  slot opening — matches TMM neck_width
neck_length   = 0.020   # [m]
neck_radius   = neck_width_2D / 2.0
neck_eff      = neck_length + 1.7 * neck_radius
neck_area_3D  = neck_width_2D * duct_height   # SLOT — matches TMM neck_area

ar_start = 1.0
ar_stop  = 5.0
ar_list  = np.linspace(ar_start, ar_stop, num_ars)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED GRID LAYOUT
#
# Grid height fixed at widest cavity (AR = ar_start) — shape never changes.
#
# j = 0                              bottom grid wall
# j = 1 .. cav_cells_y_max           cavity band
# j = cav_cells_y_max+1 ..
#     cav_cells_y_max + neck_cells_y  neck band
# j = duct_j_start .. duct_j_end-1   duct interior  (always air)
# j = duct_j_end                     duct top wall
# j = N_y_full - 1                   top grid wall
# ═══════════════════════════════════════════════════════════════════════════════

neck_cells_x          = max(1, int(round(neck_width_2D / dx)))
neck_cells_y          = max(1, int(round(neck_length   / dx)))
duct_cells_y          = int(round(duct_height / dx))
cav_width_at_ar_start = np.cbrt(cavity_volume / ar_start)
cav_cells_y_max       = max(1, int(round(cav_width_at_ar_start / dx)))

N_y_full     = 1 + cav_cells_y_max + neck_cells_y + duct_cells_y + 2
duct_j_start = 1 + cav_cells_y_max + neck_cells_y
duct_j_end   = duct_j_start + duct_cells_y
duct_j_mid   = (duct_j_start + duct_j_end) // 2
N_x          = int(duct_length / dx)

resonator_centre_x = duct_length / 2.0
src_i = pml_cells + 5
rec_i = N_x - 1 - pml_cells - 5
src_j = duct_j_mid
rec_j = duct_j_mid


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT ARRAYS
#
# Mirrors TMM exactly:
#   all_freqs[ar_idx, offset_idx, mode_idx]  — peak frequencies [Hz]
#   all_dbs  [ar_idx, offset_idx, mode_idx]  — peak TL depths [dB]
#   offset_fracs_all[ar_idx, offset_idx]     — normalised offset 0→1
#
# Additionally stores full TL spectra to avoid re-running FDTD for plots:
#   all_TL[ar_idx, offset_idx, freq_bin]     — full TL spectrum per config
#   fdtd_freqs_axis                          — shared frequency axis [Hz]
# ═══════════════════════════════════════════════════════════════════════════════

# Frequency axis size: rfft of N_steps gives this many bins
n_fft_bins = N_steps // 2 + 1

all_freqs        = np.zeros((num_ars, num_offsets, max_modes))
all_dbs          = np.zeros((num_ars, num_offsets, max_modes))
offset_fracs_all = np.zeros((num_ars, num_offsets))
all_TL           = np.zeros((num_ars, num_offsets, n_fft_bins))  # full spectra


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

neck_imp_check = rho * c / neck_area_3D
duct_imp_check = rho * c / duct_height ** 2
mem_mb         = num_ars * num_offsets * n_fft_bins * 8 / 1e6

print("=" * 65)
print(f"FDTD PARAMETRIC SWEEP  —  mode: {SWEEP_MODE.upper()}")
print("=" * 65)
print(f"Grid:               {N_x} x {N_y_full} cells  ({N_x*dx*1000:.0f}mm x {N_y_full*dx*1000:.1f}mm)")
print(f"N_steps:            {N_steps:,}  ({T_sim*1000:.0f}ms)")
print(f"dt:                 {dt*1e6:.3f} us")
print(f"Freq resolution:    {1/T_sim:.1f} Hz/bin  ({n_fft_bins} bins)")
print(f"Cells/wl @ 10kHz:   {(c/f_max)/dx:.0f}")
print(f"Duct:               {duct_length*1000:.0f}mm x {duct_height*1000:.0f}mm")
print(f"Neck slot width:    {neck_width_2D*1000:.0f}mm  ->  {neck_cells_x} cells")
print(f"Neck length:        {neck_length*1000:.0f}mm  ->  {neck_cells_y} cells")
print(f"Neck eff length:    {neck_eff*1000:.2f}mm")
print(f"Neck area (3D):     {neck_area_3D:.2e} m2  ({neck_width_2D*1000:.0f}mm x {duct_height*1000:.0f}mm slot)")
print(f"Neck impedance:     {neck_imp_check:.2f} Pa.s/m3")
print(f"Duct impedance:     {duct_imp_check:.2f} Pa.s/m3")
print(f"Neck/duct Z ratio:  {neck_imp_check/duct_imp_check:.1f}")
print(f"Cavity volume:      {cavity_volume*1e6:.0f} ml")
print(f"AR range:           {ar_start} - {ar_stop}  ({num_ars} values)")
print(f"Offsets per AR:     {num_offsets}")
print(f"Total FDTD runs:    {num_ars * num_offsets}")
print(f"TL storage:         {mem_mb:.1f} MB  (all_TL array)")
print(f"max_modes:          {max_modes}")
print(f"j layout:")
print(f"  j=0               bottom grid wall")
print(f"  j=1..{cav_cells_y_max:<2}           cavity band ({cav_cells_y_max} rows)")
print(f"  j={1+cav_cells_y_max}..{duct_j_start-1:<2}         neck band ({neck_cells_y} rows)")
print(f"  j={duct_j_start}..{duct_j_end-1:<2}         duct interior ({duct_cells_y} cells, mid={duct_j_mid})")
print(f"  j={duct_j_end}              duct top wall")
print(f"  j={N_y_full-1}              top grid wall")
print(f"Source:             i={src_i}  j={src_j}")
print(f"Receiver:           i={rec_i}  j={rec_j}")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_pml(Nx, Ny, pml_cells, dx, c, dt):
    """PML decay arrays — x-boundaries only, computed once."""
    sigma_max = -3.0 * c * np.log(1e-6) / (2.0 * pml_cells * dx)
    sigma_x   = np.zeros((Nx, Ny))
    for i in range(pml_cells):
        depth              = (pml_cells - i) / pml_cells
        sigma_val          = sigma_max * depth ** 2
        sigma_x[i, :]           = sigma_val
        sigma_x[Nx - 1 - i, :] = sigma_val
    decay_x = np.exp(-sigma_x * dt)
    decay_y = np.ones((Nx, Ny))
    return decay_x, decay_y


def build_plain_mask(Nx, Ny, duct_j_start, duct_j_end):
    """Plain duct: duct interior is air, everything else is wall."""
    mask = np.ones((Nx, Ny), dtype=bool)
    mask[:, duct_j_start:duct_j_end] = False
    return mask


def build_resonator_mask(Nx, Ny,
                         duct_j_start, duct_j_end,
                         cav_i_start, cav_i_end,
                         cav_cells_y_max, cav_cells_y,
                         neck_i_start, neck_i_end,
                         neck_cells_y):
    """
    Resonator mask: duct + cavity + neck are air, everything else is wall.
    Cavity occupies the TOP cav_cells_y rows of the cavity band so shorter
    cavities sit flush against the neck with a solid closed end wall below.
    """
    mask = np.ones((Nx, Ny), dtype=bool)
    mask[:, duct_j_start:duct_j_end] = False
    cav_j_top    = 1 + cav_cells_y_max
    cav_j_bottom = max(1, cav_j_top - cav_cells_y)
    mask[cav_i_start:cav_i_end, cav_j_bottom:cav_j_top] = False
    neck_j_bottom = cav_j_top
    neck_j_top    = duct_j_start + 1   # +1 punches through duct floor cell
    mask[neck_i_start:neck_i_end, neck_j_bottom:neck_j_top] = False
    return mask


def run_simulation(Nx, Ny, N_steps, dt, dx, rho, c,
                   decay_x, decay_y,
                   src_i, src_j, rec_i, rec_j,
                   wall_mask, pulse_width, pulse_delay):
    """Staggered leapfrog FDTD with hard-wall BCs and PML."""
    p        = np.zeros((Nx, Ny))
    u        = np.zeros((Nx, Ny))
    v        = np.zeros((Nx, Ny))
    receiver = np.zeros(N_steps)
    coeff_p  = rho * c ** 2 * dt / dx
    coeff_u  = dt / (rho * dx)

    for n in range(N_steps):
        t = n * dt
        u[:-1, :] -= coeff_u * (p[1:, :] - p[:-1, :])
        v[:, :-1] -= coeff_u * (p[:, 1:] - p[:, :-1])
        u[wall_mask] = 0.0
        v[wall_mask] = 0.0
        p[1:, :] -= coeff_p * (u[1:, :] - u[:-1, :])
        p[:, 1:] -= coeff_p * (v[:, 1:] - v[:, :-1])
        p *= decay_x * decay_y
        u *= decay_x
        v *= decay_y
        pulse = np.exp(-((t - pulse_delay) / pulse_width) ** 2)
        p[src_i, src_j] += pulse
        receiver[n] = p[rec_i, rec_j]

    return receiver


def signal_to_db(time_signal, dt):
    """Hanning-windowed FFT -> (freqs, dB spectrum)."""
    N      = len(time_signal)
    window = np.hanning(N)
    P      = np.fft.rfft(time_signal * window)
    freqs  = np.fft.rfftfreq(N, d=dt)
    db     = 20.0 * np.log10(np.abs(P) + 1e-12)
    return freqs, db


def geometry_for_ar(ar):
    """Cavity dimensions and max offset for a given AR."""
    cav_width   = np.cbrt(cavity_volume / ar)
    cav_length  = cav_width * ar
    cav_cells_x = max(1, int(round(cav_length / dx)))
    cav_cells_y = max(1, int(round(cav_width  / dx)))
    max_offset  = (cav_length / 2.0) - neck_radius
    return cav_width, cav_length, cav_cells_x, cav_cells_y, max_offset


def neck_positions(neck_offset, cav_length, cav_cells_x):
    """Cavity and neck i-positions for a given neck offset [m]."""
    neck_centre_i     = int(round(resonator_centre_x / dx))
    neck_offset_cells = int(round(neck_offset / dx))
    neck_i_start      = neck_centre_i + neck_offset_cells - neck_cells_x // 2
    neck_i_end        = neck_i_start + neck_cells_x
    cav_i_start       = neck_centre_i - cav_cells_x // 2
    cav_i_end         = cav_i_start + cav_cells_x
    neck_i_start = max(cav_i_start, min(neck_i_start, cav_i_end - neck_cells_x))
    neck_i_end   = neck_i_start + neck_cells_x
    cav_i_start  = max(0, cav_i_start)
    cav_i_end    = min(N_x, cav_i_end)
    return cav_i_start, cav_i_end, neck_i_start, neck_i_end


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD SHARED ARRAYS
# ═══════════════════════════════════════════════════════════════════════════════

decay_x, decay_y = build_pml(N_x, N_y_full, pml_cells, dx, c, dt)
wall_plain       = build_plain_mask(N_x, N_y_full, duct_j_start, duct_j_end)


# ═══════════════════════════════════════════════════════════════════════════════
# PLAIN DUCT RUN  —  once only, reference for all TL calculations
# ═══════════════════════════════════════════════════════════════════════════════

print("\nRunning plain duct reference simulation (once)...")
t0        = time.time()
rec_plain = run_simulation(
    N_x, N_y_full, N_steps, dt, dx, rho, c,
    decay_x, decay_y,
    src_i, src_j, rec_i, rec_j,
    wall_plain, pulse_width, pulse_delay
)
fdtd_freqs_axis, db_plain = signal_to_db(rec_plain, dt)
freq_res      = fdtd_freqs_axis[1] - fdtd_freqs_axis[0]
freq_mask_10k = fdtd_freqs_axis <= f_max
min_dist_bins = max(1, int(min_peak_gap / freq_res))
print(f"Done in {time.time()-t0:.1f}s  |  freq resolution: {freq_res:.2f} Hz/bin")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PARAMETRIC SWEEP
#
# For every (AR, offset) configuration:
#   1. Build resonator mask
#   2. Run FDTD simulation
#   3. Compute TL = db_plain - db_resonator
#   4. Store full TL spectrum in all_TL[i, j, :]
#
# Peak detection happens AFTER the sweep on the stored TL arrays,
# mirroring the TMM pipeline exactly.
# ═══════════════════════════════════════════════════════════════════════════════

total_runs  = num_ars * num_offsets
run_count   = 0
sweep_start = time.time()

print(f"\nStarting sweep: {num_ars} ARs x {num_offsets} offsets = {total_runs} runs\n")

for i, ar in enumerate(ar_list):

    _, cav_length, cav_cells_x, cav_cells_y, max_offset = geometry_for_ar(ar)
    offsets      = np.linspace(0.0, max_offset, num_offsets)
    offset_fracs = offsets / max_offset if max_offset > 0 else np.zeros(num_offsets)
    offset_fracs_all[i, :] = offset_fracs

    for j, offset in enumerate(offsets):

        run_count += 1
        elapsed    = time.time() - sweep_start
        if run_count > 1:
            eta     = elapsed / (run_count - 1) * (total_runs - run_count + 1)
            eta_str = f"ETA {eta/60:.1f}min"
        else:
            eta_str = "ETA --"

        print(f"  [{run_count:>4}/{total_runs}]  AR={ar:.3f}  "
              f"offset={offset_fracs[j]:.0%}  {eta_str}", flush=True)

        cav_i_start, cav_i_end, neck_i_start, neck_i_end = neck_positions(
            offset, cav_length, cav_cells_x
        )
        wall_res = build_resonator_mask(
            N_x, N_y_full,
            duct_j_start, duct_j_end,
            cav_i_start, cav_i_end,
            cav_cells_y_max, cav_cells_y,
            neck_i_start, neck_i_end,
            neck_cells_y
        )
        rec_res = run_simulation(
            N_x, N_y_full, N_steps, dt, dx, rho, c,
            decay_x, decay_y,
            src_i, src_j, rec_i, rec_j,
            wall_res, pulse_width, pulse_delay
        )

        _, db_res      = signal_to_db(rec_res, dt)
        TL             = db_plain - db_res
        TL[~freq_mask_10k] = 0.0

        all_TL[i, j, :] = TL   # store full spectrum — no re-running needed for plots


total_time = time.time() - sweep_start
print(f"\nSweep complete — {total_runs} runs in {total_time/60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK DETECTION  —  identical method to TMM, applied to stored TL arrays
# ═══════════════════════════════════════════════════════════════════════════════

print("\nDetecting peaks across all configurations...")

for i in range(num_ars):
    for j in range(num_offsets):

        TL_slice = all_TL[i, j, freq_mask_10k]

        peak_idx, _ = find_peaks(
            TL_slice,
            height=min_peak_db,
            distance=min_dist_bins
        )
        peak_freqs = fdtd_freqs_axis[freq_mask_10k][peak_idx]
        peak_dbs   = TL_slice[peak_idx]

        if len(peak_freqs) > 0:
            order      = np.argsort(peak_freqs)
            peak_freqs = peak_freqs[order]
            peak_dbs   = peak_dbs[order]

        n = min(len(peak_freqs), max_modes)
        all_freqs[i, j, :n] = peak_freqs[:n]
        all_dbs  [i, j, :n] = peak_dbs[:n]

print("Peak detection done.")

n_modes_present = int(np.max(np.sum(all_freqs > 0, axis=2)))
print(f"Max modes detected in any single configuration: {n_modes_present}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

np.save("fdtd_all_freqs.npy",        all_freqs)
np.save("fdtd_all_dbs.npy",          all_dbs)
np.save("fdtd_all_TL.npy",           all_TL)
np.save("fdtd_freqs_axis.npy",       fdtd_freqs_axis)
np.save("fdtd_offset_fracs_all.npy", offset_fracs_all)
np.save("fdtd_ar_list.npy",          ar_list)
print("Results saved to .npy files.")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING CONSTANTS  —  mirror TMM exactly
# ═══════════════════════════════════════════════════════════════════════════════

fixed_ar_idx  = int(np.abs(ar_list - FIXED_AR).argmin())
fixed_off_idx = int(round(FIXED_OFFSET * (num_offsets - 1)))

print(f"Fixed AR for offset sweeps  : {ar_list[fixed_ar_idx]:.3f}")
print(f"Fixed offset for AR sweeps  : {FIXED_OFFSET:.0%}  (index {fixed_off_idx})")

mode_colours = plt.get_cmap('tab10')
extent_2d    = [ar_start, ar_stop, 0.0, 1.0]


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS  —  identical layout to TMM
# ═══════════════════════════════════════════════════════════════════════════════

def make_sq1_presence_page(m):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"FDTD  SQ1 — Mode {m+1} Presence\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}"
        f"  |  Sweep: {SWEEP_MODE}",
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
        f"FDTD  SQ2 — Mode {m+1} Frequency\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}"
        f"  |  Sweep: {SWEEP_MODE}",
        fontsize=11, fontweight='bold'
    )
    freq_vs_ar = np.where(all_freqs[:, fixed_off_idx, m] > 0,
                          all_freqs[:, fixed_off_idx, m], np.nan)
    axes[0].plot(ar_list, freq_vs_ar, color=mode_colours(m), lw=2, marker='o', markersize=4)
    axes[0].set_xlabel("Aspect Ratio  L/W")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(f"A — vs AR  (offset fixed={FIXED_OFFSET:.0%})")
    axes[0].grid(True, alpha=0.3)

    freq_vs_off = np.where(all_freqs[fixed_ar_idx, :, m] > 0,
                           all_freqs[fixed_ar_idx, :, m], np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], freq_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=4)
    axes[1].set_xlabel("Offset Fraction  (0=centre, 1=wall)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"B — vs offset  (AR fixed={ar_list[fixed_ar_idx]:.2f})")
    axes[1].grid(True, alpha=0.3)

    freq_grid = all_freqs[:, :, m].copy()
    freq_grid[freq_grid == 0] = np.nan
    vmin = np.nanpercentile(freq_grid, 2)  if not np.all(np.isnan(freq_grid)) else 0
    vmax = np.nanpercentile(freq_grid, 98) if not np.all(np.isnan(freq_grid)) else 1
    im = axes[2].imshow(
        freq_grid.T, origin='lower', aspect='auto', extent=extent_2d,
        cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax
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
        f"FDTD  SQ3 — Mode {m+1} TL Depth\n"
        f"Fixed AR={ar_list[fixed_ar_idx]:.2f}  |  Fixed offset={FIXED_OFFSET:.0%}"
        f"  |  Sweep: {SWEEP_MODE}",
        fontsize=11, fontweight='bold'
    )
    db_vs_ar = np.where(all_dbs[:, fixed_off_idx, m] > 0,
                        all_dbs[:, fixed_off_idx, m], np.nan)
    axes[0].plot(ar_list, db_vs_ar, color=mode_colours(m), lw=2, marker='o', markersize=4)
    axes[0].set_xlabel("Aspect Ratio  L/W")
    axes[0].set_ylabel("TL Depth (dB)")
    axes[0].set_title(f"A — vs AR  (offset fixed={FIXED_OFFSET:.0%})")
    axes[0].grid(True, alpha=0.3)

    db_vs_off = np.where(all_dbs[fixed_ar_idx, :, m] > 0,
                         all_dbs[fixed_ar_idx, :, m], np.nan)
    axes[1].plot(offset_fracs_all[fixed_ar_idx, :], db_vs_off,
                 color=mode_colours(m), lw=2, marker='o', markersize=4)
    axes[1].set_xlabel("Offset Fraction  (0=centre, 1=wall)")
    axes[1].set_ylabel("TL Depth (dB)")
    axes[1].set_title(f"B — vs offset  (AR fixed={ar_list[fixed_ar_idx]:.2f})")
    axes[1].grid(True, alpha=0.3)

    db_grid = all_dbs[:, :, m].copy()
    db_grid[db_grid == 0] = np.nan
    vmin = np.nanpercentile(db_grid, 2)  if not np.all(np.isnan(db_grid)) else 0
    vmax = np.nanpercentile(db_grid, 98) if not np.all(np.isnan(db_grid)) else 1
    im = axes[2].imshow(
        db_grid.T, origin='lower', aspect='auto', extent=extent_2d,
        cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("TL Depth (dB)")
    axes[2].set_xlabel("Aspect Ratio  L/W")
    axes[2].set_ylabel("Offset Fraction")
    axes[2].set_title("C — 2D TL depth map  (white=not detected)")
    plt.tight_layout()
    return fig


def make_ar_plots(which_ars=None):
    """
    Raw TL spectra overlaid across all offsets for selected ARs.
    3 ARs per page. Reads from stored all_TL — no FDTD re-runs.
    """
    if which_ars is None:
        which_ars = list(range(num_ars))

    n   = len(which_ars)
    fig, axes = plt.subplots(n, 1, figsize=(14, 7 * n), squeeze=False)
    offset_cmap = plt.get_cmap('cool')
    offset_norm = Normalize(vmin=0, vmax=1)

    for row, i in enumerate(which_ars):
        ax = axes[row, 0]
        ar = ar_list[i]
        cav_width_plot, cav_length, _, _, max_offset = geometry_for_ar(ar)

        for j in range(num_offsets):
            frac   = j / (num_offsets - 1) if num_offsets > 1 else 0.0
            colour = offset_cmap(frac)
            TL     = all_TL[i, j, freq_mask_10k]
            ax.plot(fdtd_freqs_axis[freq_mask_10k], TL,
                    color=colour, linewidth=0.7, alpha=0.75)

        sm = ScalarMappable(cmap=offset_cmap, norm=offset_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
        cbar.set_label("Neck offset\n(0=centre, 1=wall)", fontsize=8)

        ax.set_title(
            f"FDTD  AR={ar:.3f}  |  L={cav_length*1000:.1f}mm  |  "
            f"W={cav_width_plot*1000:.1f}mm  |  max offset={max_offset*1000:.1f}mm",
            fontweight='bold', loc='left', fontsize=9
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Transmission Loss (dB)")
        ax.set_xlim(0, f_max)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linestyle='--')

    plt.subplots_adjust(hspace=0.45, left=0.07, right=0.92, top=0.97, bottom=0.03)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

pdf_path = f"Helmholtz_FDTD_{SWEEP_MODE.capitalize()}_Report.pdf"
per_page = 3
print(f"\nGenerating PDF: {pdf_path}")

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

    print("  Raw TL overlays — all ARs (from stored data, no re-runs)")
    for batch_start in range(0, num_ars, per_page):
        batch = list(range(batch_start, min(batch_start + per_page, num_ars)))
        fig   = make_ar_plots(which_ars=batch)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    AR {ar_list[batch[0]]:.2f}-{ar_list[batch[-1]]:.2f}")

    info = pdf.infodict()
    info['Title']   = f'Helmholtz Resonator FDTD Parametric Sweep ({SWEEP_MODE})'
    info['Author']  = 'Svar Joshi'
    info['Subject'] = 'Asymmetric rectangular cavity — neck offset x aspect ratio — 2D FDTD'

print(f"\nSaved: {pdf_path}")

if platform.system() == 'Darwin':
    subprocess.call(('open', pdf_path))
elif platform.system() == 'Windows':
    os.startfile(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE  —  identical commands to TMM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 52)
print("INTERACTIVE MODE")
print("-" * 52)
print("  sq1 [mode]   — mode presence plots    e.g. sq1 2")
print("  sq2 [mode]   — mode frequency plots")
print("  sq3 [mode]   — mode TL depth plots")
print("  [number]     — raw TL overlay for AR  e.g. 3.5")
print("  exit         — quit")
print("-" * 52)
print(f"  ({n_modes_present} modes detected, {SWEEP_MODE} sweep)")
print("-" * 52)

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
            print(f"  Usage: sq1 [1-{n_modes_present}]")

    elif cmd.startswith('sq2'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq2_frequency_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq2 [1-{n_modes_present}]")

    elif cmd.startswith('sq3'):
        try:
            m = int(cmd.split()[1]) - 1
            if not (0 <= m < n_modes_present):
                print(f"  Mode out of range — enter 1 to {n_modes_present}")
                continue
            fig = make_sq3_depth_page(m)
            plt.show()
        except (IndexError, ValueError):
            print(f"  Usage: sq3 [1-{n_modes_present}]")

    else:
        try:
            target_ar = float(cmd)
            closest   = int(np.abs(ar_list - target_ar).argmin())
            print(f"Plotting AR={ar_list[closest]:.3f}  (from stored TL data)")
            fig = make_ar_plots(which_ars=[closest])
            plt.show()
        except ValueError:
            print("Unrecognised — try 'sq1 2', '3.5', or 'exit'.")