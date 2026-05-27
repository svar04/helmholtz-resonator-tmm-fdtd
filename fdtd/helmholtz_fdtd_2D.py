import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
matplotlib.use('TkAgg')


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

c   = 343.0
rho = 1.21


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

f_max       = 10000.0
dx          = 0.001
dt          = 0.4 * dx / (c * np.sqrt(2))
pml_cells   = 10

pulse_width = 1.5 / f_max
pulse_delay = 6.0 * pulse_width


# ═══════════════════════════════════════════════════════════════════════════════
# DUCT GEOMETRY
#
# duct_height must satisfy c/(2*duct_height) > f_max = 10000 Hz
# → duct_height < c/(2*f_max) = 343/20000 = 0.01715 m
# Using 0.012m gives cutoff = 343/(2*0.012) = 14292 Hz  ✓
# Neck width = 2*neck_radius = 0.012m → neck_cells_x = 12 at dx=1mm
# Neck area  = neck_width * duct_height = 0.012 * 0.012 = 1.44e-4 m²
# neck_area_3D = (2*0.006)^2 = 1.44e-4 m²  ← exact match ✓
# ═══════════════════════════════════════════════════════════════════════════════

duct_length = 0.800
duct_height = 0.300
duct_width  = 0.300

# ═══════════════════════════════════════════════════════════════════════════════
# NECK GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

neck_width_2D = 0.050
neck_length   = 0.050

neck_radius = neck_width_2D / 2
neck_eff      = neck_length + 1.7 * neck_radius
neck_area_3D  = neck_width_2D ** 2

neck_cells_x = max(1, int(round(neck_width_2D / dx)))
neck_cells_y = max(1, int(round(neck_length   / dx)))


# ═══════════════════════════════════════════════════════════════════════════════
# CAVITY GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

cavity_volume = 0.008

ar_start = 1
ar_stop = 6
ar_step = 0.05

AR          = 4
offset_frac = 0.5

cav_width  = np.cbrt(cavity_volume / AR)
cav_length = cav_width * AR

max_offset  = (cav_length / 2.0) - neck_radius
neck_offset = offset_frac * max_offset

resonator_centre_x = duct_length / 2.0

cav_cells_x = max(1, int(round(cav_length / dx)))
cav_cells_y = max(1, int(round(cav_width  / dx)))


# ═══════════════════════════════════════════════════════════════════════════════
# GRID LAYOUT
#
# j=0                         bottom grid wall  (solid)
# j=1 .. cav_cells_y_max      cavity band       (solid in plain, air in resonator
#                                                 for the top cav_cells_y rows)
# j=cav_cells_y_max+1 ..
#   cav_cells_y_max+neck_cells_y  neck band     (solid in plain, air in resonator
#                                                 at neck x-positions only)
# j=duct_j_start ..
#   duct_j_end-1              duct interior     (always air)
# j=duct_j_end                duct top wall     (solid)
# j=N_y_full-1                top grid wall     (solid)
# ═══════════════════════════════════════════════════════════════════════════════

duct_cells_y    = int(round(duct_height / dx))
cav_width_at_ar_min  = np.cbrt(cavity_volume / ar_start)   # ar_start = 1.5
cav_cells_y_max = max(1, int(round(cav_width_at_ar_min / dx)))

N_y_full     = 1 + cav_cells_y_max + neck_cells_y + duct_cells_y + 2
duct_j_start = 1 + cav_cells_y_max + neck_cells_y   # first duct air cell
duct_j_end   = duct_j_start + duct_cells_y            # one past last duct air cell
duct_j_mid   = (duct_j_start + duct_j_end) // 2

N_x     = int(duct_length / dx)
T_sim   = 0.06
N_steps = int(T_sim / dt)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE AND RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

src_i = pml_cells + 5
rec_i = N_x - 1 - pml_cells - 5
src_j = duct_j_mid
rec_j = duct_j_mid


# ═══════════════════════════════════════════════════════════════════════════════
# NECK AND CAVITY POSITIONS IN i
# ═══════════════════════════════════════════════════════════════════════════════

neck_centre_i     = int(round(resonator_centre_x / dx))
neck_offset_cells = int(round(neck_offset / dx))
neck_i_start      = neck_centre_i + neck_offset_cells - neck_cells_x // 2
neck_i_end        = neck_i_start + neck_cells_x

cav_i_start = neck_centre_i - cav_cells_x // 2
cav_i_end   = cav_i_start + cav_cells_x

# Clamp to grid bounds
neck_i_start = max(cav_i_start, min(neck_i_start, cav_i_end - neck_cells_x))
neck_i_end   = neck_i_start + neck_cells_x
cav_i_start  = max(0, cav_i_start)
cav_i_end    = min(N_x, cav_i_end)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PRINTS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("GEOMETRY SUMMARY")
print("=" * 60)
print(f"Grid:              {N_x} x {N_y_full} cells  ({N_x*dx*100:.0f}cm x {N_y_full*dx*100:.1f}cm)")
print(f"Time steps:        {N_steps}  ({T_sim*1000:.0f}ms)")
print(f"dt:                {dt*1e6:.3f} us")
print(f"Cells/wavelength:  {(c/f_max)/dx:.1f} at {f_max:.0f}Hz")
print()
print(f"Duct:              {duct_length*100:.0f}cm x {duct_height*1000:.1f}mm")
print(f"Duct cells:        {N_x} x {duct_cells_y}  (mid j={duct_j_mid})")
cutoff = c / (2 * duct_height)
status = "OK" if cutoff > f_max else "WARNING — ABOVE CUTOFF"
print(f"Cutoff freq:       {cutoff:.0f} Hz  [{status}]")
print()
print(f"Neck area (3D):    {neck_area_3D*1e6:.4f} mm²")
print(f"Neck area (2D):    {neck_width_2D*duct_height*1e6:.4f} mm²  (must equal 3D area)")
print(f"Neck width (2D):   {neck_width_2D*1000:.1f}mm  →  {neck_cells_x} cells")
print(f"Neck length:       {neck_length*1000:.1f}mm  →  {neck_cells_y} cells")
print(f"Neck impedance:    TMM={rho*c/neck_area_3D:.0f}  FDTD={rho*c/(neck_width_2D*duct_height):.0f}  match={'YES' if abs(neck_area_3D - neck_width_2D*duct_height) < 1e-10 else 'NO'}")
print()
print(f"Cavity AR:         {AR:.1f}")
print(f"Cavity:            {cav_length*1000:.1f}mm x {cav_width*1000:.1f}mm  ({cav_cells_y} x {cav_cells_x} cells)")
print(f"Neck offset:       {neck_offset*1000:.2f}mm  ({offset_frac:.0%} of max {max_offset*1000:.2f}mm)")
print()
print(f"j layout:")
print(f"  j=0                 bottom grid wall")
print(f"  j=1..{cav_cells_y_max:<3}           cavity band (max AR=6 → {cav_cells_y_max} rows)")
print(f"  j={1+cav_cells_y_max}..{duct_j_start-1:<3}          neck band ({neck_cells_y} rows)")
print(f"  j={duct_j_start}..{duct_j_end-1:<3}          duct interior ({duct_cells_y} cells)  mid={duct_j_mid}")
print(f"  j={duct_j_end}                 duct top wall")
print(f"  j={N_y_full-1}                 top grid wall")
print()
print(f"Source:            i={src_i}  j={src_j}")
print(f"Receiver:          i={rec_i}  j={rec_j}")
print(f"Left PML:          i=0..{pml_cells-1}")
print(f"Right PML:         i={N_x-pml_cells}..{N_x-1}")
print(f"Cavity i:          {cav_i_start}..{cav_i_end-1}  ({cav_cells_x} cells)")
print(f"Neck i:            {neck_i_start}..{neck_i_end-1}  ({neck_cells_x} cells)")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_pml_sigma(Nx, Ny, pml_cells, dx, c):
    sigma_max = -3 * c * np.log(1e-6) / (2 * pml_cells * dx)
    sigma_x   = np.zeros((Nx, Ny))
    sigma_y   = np.zeros((Nx, Ny))
    for i in range(pml_cells):
        depth              = (pml_cells - i) / pml_cells
        sigma_val          = sigma_max * depth ** 2
        sigma_x[i, :]           = sigma_val
        sigma_x[Nx - 1 - i, :] = sigma_val
    return sigma_x, sigma_y


def build_plain_duct_mask(Nx, Ny, duct_j_start, duct_j_end):
    mask = np.ones((Nx, Ny), dtype=bool)
    mask[:, duct_j_start:duct_j_end] = False
    return mask


def build_resonator_mask(Nx, Ny,
                         duct_j_start, duct_j_end,
                         cav_i_start, cav_i_end,
                         cav_cells_y_max, cav_cells_y,
                         neck_i_start, neck_i_end,
                         neck_cells_y):
    mask = np.ones((Nx, Ny), dtype=bool)

    # Duct interior
    mask[:, duct_j_start:duct_j_end] = False

    # Cavity — occupies the TOP cav_cells_y rows of the cavity band.
    # The cavity band runs from j=1 to j=cav_cells_y_max (inclusive).
    # Shorter cavities (lower AR) leave solid wall below them — correct physics.
    cav_j_top    = 1 + cav_cells_y_max      # Python slice end (exclusive) = first neck row
    cav_j_bottom = cav_j_top - cav_cells_y  # first cavity air row (inclusive)
    cav_j_bottom = max(1, cav_j_bottom)     # j=0 stays as solid floor
    mask[cav_i_start:cav_i_end, cav_j_bottom:cav_j_top] = False

    # Neck — runs from cav_j_top up to and including duct_j_start.
    # neck_j_top = duct_j_start + 1 so the slice mask[..., neck_j_bottom:neck_j_top]
    # includes duct_j_start, punching through the duct bottom wall.
    neck_j_bottom = cav_j_top
    neck_j_top    = duct_j_start + 1
    mask[neck_i_start:neck_i_end, neck_j_bottom:neck_j_top] = False

    return mask


def run_simulation(Nx, Ny, N_steps, dt, dx, rho, c,
                   decay_x, decay_y,
                   src_i, src_j, rec_i, rec_j,
                   wall_mask, pulse_width, pulse_delay):
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


def pressure_to_db(time_signal, dt):
    N      = len(time_signal)
    window = np.hanning(N)
    P      = np.fft.rfft(time_signal * window)
    freqs  = np.fft.rfftfreq(N, d=dt)
    db     = 20.0 * np.log10(np.abs(P) + 1e-12)
    return freqs, db


def visualise_masks(wall_resonator, src_i, rec_i, pml_cells, N_x, N_y_full):
    cell_scale = 0.025
    fig_w = max(16, N_x * cell_scale + 1)
    fig_h = max(2, N_y_full * cell_scale + 1)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(wall_resonator.T, origin='lower', cmap='gray_r', aspect='equal',
              extent=[-0.5, N_x - 0.5, -0.5, N_y_full - 0.5])

    left_pml = mpatches.Rectangle(
        (-0.5, -0.5), pml_cells, N_y_full,
        linewidth=0, facecolor='#ff69b4', alpha=0.25, zorder=0
    )
    right_pml = mpatches.Rectangle(
        (N_x - pml_cells - 0.5, -0.5), pml_cells, N_y_full,
        linewidth=0, facecolor='#ff69b4', alpha=0.25, zorder=0
    )
    ax.add_patch(left_pml)
    ax.add_patch(right_pml)

    ax.axvline(src_i, color='lime', lw=1.2, label=f'source i={src_i}', zorder=2)
    ax.axvline(rec_i, color='red',  lw=1.2, label=f'receiver i={rec_i}', zorder=2)

    ax.set_xlim(-0.5, N_x - 0.5)
    ax.set_ylim(-0.5, N_y_full - 0.5)
    ax.set_title(f"Resonator  AR={AR:.1f}  offset={offset_frac:.0%}  (black=wall, white=air, pink=PML)")
    ax.set_xlabel("i  (x →)")
    ax.set_ylabel("j  (y ↑)")
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD MASKS AND SHARED ARRAYS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"DEBUG cav_i_start={cav_i_start} cav_i_end={cav_i_end} width={cav_i_end-cav_i_start}")
print(f"DEBUG cav_j_bottom={1+cav_cells_y_max-cav_cells_y} cav_j_top={1+cav_cells_y_max} height={cav_cells_y}")

sigma_x, sigma_y = build_pml_sigma(N_x, N_y_full, pml_cells, dx, c)
decay_x = np.exp(-sigma_x * dt)
decay_y = np.exp(-sigma_y * dt)

wall_plain = build_plain_duct_mask(N_x, N_y_full, duct_j_start, duct_j_end)

wall_resonator = build_resonator_mask(
    N_x, N_y_full,
    duct_j_start, duct_j_end,
    cav_i_start, cav_i_end,
    cav_cells_y_max, cav_cells_y,
    neck_i_start, neck_i_end,
    neck_cells_y
)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\nWall checks (all must be False):")
print(f"  Source in plain wall:      {wall_plain[src_i, src_j]}")
print(f"  Receiver in plain wall:    {wall_plain[rec_i, rec_j]}")
print(f"  Source in resonator wall:  {wall_resonator[src_i, src_j]}")
print(f"  Receiver in resonator wall:{wall_resonator[rec_i, rec_j]}")

extra_air = int(np.sum(~wall_resonator) - np.sum(~wall_plain))
expected  = cav_cells_x * cav_cells_y + neck_cells_x * neck_cells_y
print(f"  Extra air cells:           {extra_air}  (expected ~{expected})")
print(f"  Cavity carved:             {extra_air >= cav_cells_x * cav_cells_y}")
print(f"  Neck open:                 {np.any(~wall_resonator[neck_i_start:neck_i_end, :])}")

# ── Show masks ────────────────────────────────────────────────────────────────
visualise_masks(wall_resonator, src_i, rec_i, pml_cells, N_x, N_y_full)



# ═══════════════════════════════════════════════════════════════════════════════
# RUN 1 — PLAIN DUCT
# ═══════════════════════════════════════════════════════════════════════════════

print("\nRun 1: plain duct...")
rec_plain = run_simulation(
    N_x, N_y_full, N_steps, dt, dx, rho, c,
    decay_x, decay_y,
    src_i, src_j, rec_i, rec_j,
    wall_plain, pulse_width, pulse_delay
)
print("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 2 — WITH RESONATOR
# ═══════════════════════════════════════════════════════════════════════════════

print("Run 2: with resonator...")
rec_resonator = run_simulation(
    N_x, N_y_full, N_steps, dt, dx, rho, c,
    decay_x, decay_y,
    src_i, src_j, rec_i, rec_j,
    wall_resonator, pulse_width, pulse_delay
)
print("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

freqs, db_plain     = pressure_to_db(rec_plain,     dt)
_,     db_resonator = pressure_to_db(rec_resonator, dt)

TL        = db_plain - db_resonator
freq_mask = freqs <= f_max

print(f"\nFrequency resolution: {freqs[1]-freqs[0]:.2f} Hz/bin")
print(f"TL range 0–10kHz:     {TL[freq_mask].min():.1f} to {TL[freq_mask].max():.1f} dB")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(freqs[freq_mask], db_plain[freq_mask],
             label="Plain duct", color='steelblue', lw=1.5)
axes[0].plot(freqs[freq_mask], db_resonator[freq_mask],
             label="With resonator", color='tomato', lw=1.5, alpha=0.85)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Pressure level (dB)")
axes[0].set_xlim(0, 10000)
axes[0].set_title("Raw spectra")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(freqs[freq_mask], TL[freq_mask], color='darkgreen', lw=1.5)
axes[1].axhline(0, color='gray', linestyle='--', lw=0.8)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Transmission Loss (dB)")
axes[1].set_xlim(0, 10000)
axes[1].set_ylim(0, 80)
axes[1].set_title(
    f"Transmission Loss — AR={AR:.1f}, offset={offset_frac:.0%}"
    f"  |  Cavity {cav_length*1000:.1f}mm × {cav_width*1000:.1f}mm"
    f"  |  Neck {neck_width_2D*1000:.1f}mm"
)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()