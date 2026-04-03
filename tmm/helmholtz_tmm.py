import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.colors as mcolors
import os
import platform
import subprocess
from matplotlib.backends.backend_pdf import PdfPages

# User Parameters, can be changed


arStop = 5
offNum = 50


# Variables

arStep = 0.2  # Fixed step size
arStart = 2 # Fixed start
arNum = int((arStop - arStart) / arStep) + 1
fRange = 20001

p, c = 1.225, 343
r = 0.006
l_base = 0.002
l_eff = l_base + 1.7 * r
Sn = np.pi * r ** 2
Zn = p * c / Sn
sPipe = 0.03 ** 2
zPipe = p * c / sPipe
V = 1.68e-5

f = np.arange(1, fRange, 1)
k = 2 * np.pi * f / c
aspect_ratios = np.arange(arStart, arStop+0.1, arStep)

results = {}

print(f"Number of Aspect Ratios: {arNum}")
print(f"Number of Offsets: {offNum}")
print(f"Total Configurations: {arNum * offNum}")

for ar in aspect_ratios:

    W_cav = np.cbrt(V / ar)
    L_cav = W_cav * ar
    S_cav =  W_cav ** 2
    Z_cav = p * c / S_cav

    offsets = np.linspace(0, L_cav / 2, offNum)

    for off in offsets:

        l1 = L_cav / 2 - off
        l2 = L_cav / 2 + off

        A = Zn * np.sin(k * (l1 + l2)) * np.sin(k * l_eff)
        B = Z_cav * np.cos(k * l_eff) * np.cos(k * l1) * np.cos(k * l2)
        C = Zn * np.sin(k * (l1 + l2)) * np.cos(k * l_eff)
        D = Z_cav * np.sin(k * l_eff) * np.cos(k * l1) * np.cos(k * l2)

        Z = 1j * Zn * (A - B) / (C + D)

        TL = 20 * np.log10(np.abs(1 + (zPipe / (2 * Z))))

        peaks, _ = find_peaks(TL, height=5, distance=100)

        label = f"AR:{ar:.1f}, Off:{off:.3f}"
        results[label] = {
            "f": f, "TL": TL, "peak_freqs": f[peaks], "peak_dbs": TL[peaks],
            "L": L_cav, "W": W_cav, "V": V
        }

# Terminal Output

# 1. Find the maximum number of peaks to set the header width
max_peaks = max([len(data["peak_freqs"]) for data in results.values()]) if results else 0
peak_col_width = 18  # Fixed width for each peak column

# Calculate total width for separators
base_width = 65  # CONFIG + L + W + V + # PEAKS
total_width = base_width + (max_peaks * (peak_col_width + 3))

print("\n" + "=" * total_width)

# 2. Build the dynamic header
header = f"{'CONFIGURATION':<22} | {'L (m)':<8} | {'W (m)':<8} | {'V (m^3)':<10} | {'# PEAKS':<8}"
for i in range(1, max_peaks + 1):
    header += f" | {'PEAK ' + str(i):<{peak_col_width}}"

print(header)
print("=" * total_width)

# 3. Print the rows
for label, data in results.items():
    freqs, dbs = data["peak_freqs"], data["peak_dbs"]
    num_p = len(freqs)

    # Base info string with fixed widths
    row_str = (f"{label:<22} | {data['L']:<8.3f} | {data['W']:<8.3f} | "
               f"{data['V']:<10.2e} | {num_p:<8}")

    # Append each peak with the EXACT same padding as the header
    for i in range(num_p):
        peak_str = f"{freqs[i]:.0f} ({dbs[i]:.1f})"
        row_str += f" | {peak_str:<{peak_col_width}}"

    print(row_str)

print("-" * total_width)

# Generate Colors

all_labels = list(results.keys())
unique_ars = np.unique([float(label.split(',')[0].split(':')[1]) for label in all_labels])
num_ars = len(unique_ars)

# Use 0.9 range to keep colors from looping back to red
base_hues = np.linspace(0, 0.9, num_ars)
color_map = {}

for i, ar in enumerate(unique_ars):
    ar_labels = [l for l in all_labels if l.startswith(f"AR:{ar:.1f}")]
    num_off = len(ar_labels)
    for j, label in enumerate(ar_labels):
        sat = 0.4 + (j / max(1, num_off - 1)) * 0.6  # Saturation range
        val = 0.3 + (j / max(1, num_off - 1)) * 0.7  # Brightness range
        color_map[label] = mcolors.hsv_to_rgb((base_hues[i], sat, val))

# FIGURE 1 WITH ALL LINES

fig1, ax1 = plt.subplots(figsize=(16, 10))
for label, data in results.items():
    ax1.plot(data["f"], data["TL"], color=color_map[label], linewidth=0.8, alpha=0.6, label=label)

ax1.set_title("Full Sweep Overview", fontsize=14, fontweight='bold')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("TL (dB)")
ax1.grid(True, alpha=0.2)

# FIGURE 1 LEGEND

ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
           ncol=int(np.ceil(num_ars / 10)), fontsize='4', title="Configurations", frameon=True)

plt.subplots_adjust(right=0.75, bottom=0.1)

# FIGURE 2 WITH INDIVIDUAL PLOTS FOR EACH ASPECT RATIO
# Increase the height per plot to 8 or 9 to account for the legend below
total_height = 9 * num_ars
fig2, axes2 = plt.subplots(num_ars, 1, figsize=(14, total_height))

if num_ars == 1: axes2 = [axes2]

for i, ar in enumerate(unique_ars):
    ax = axes2[i]
    ar_labels = [l for l in all_labels if l.startswith(f"AR:{ar:.1f}")]
    sweep_cmap = plt.get_cmap('cool').resampled(len(ar_labels))

    for j, label in enumerate(ar_labels):
        ax.plot(results[label]["f"], results[label]["TL"], color=sweep_cmap(j), label=label, linewidth=1.2)

    ax.set_title(f"Aspect Ratio: {ar}", fontweight='bold', loc='left', fontsize=12)
    ax.set_ylabel("TL (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)

    # Move legend BELOW the plot
    # ncol=5 spreads the 50 offsets into 5 columns so it stays compact
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=5, fontsize='6', title="Neck Offsets (m)", frameon=True)

# Use a large hspace (e.g., 1.5) to create the "gap" between packets
plt.subplots_adjust(hspace=1.8, left=0.1, right=0.95, top=0.95, bottom=0.05)

# Save to PDF and close figures to prevent them from popping up later
pdf_path = "Helmholtz_TMM_Sweep_Results.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig1, bbox_inches='tight')
    pdf.savefig(fig2, bbox_inches='tight')

# CRITICAL: Close these figures so plt.show() doesn't trigger them later
plt.close(fig1)
plt.close(fig2)

print(f"\n[SUCCESS] PDF Generated: {pdf_path}")
if platform.system() == 'Darwin':
    subprocess.call(('open', pdf_path))
elif platform.system() == 'Windows':
    os.startfile(pdf_path)
else:
    try:
        subprocess.call(('xdg-open', pdf_path))
    except:
        print("Could not open PDF automatically.")

# --- INTERACTIVE EXPLORATION MODE ---
print("\n" + "=" * 60)
print("INTERACTIVE MODE: Zoom in on specific results")
print("- Enter an Aspect Ratio (e.g., 1.6)")
print("- Type 'all' to see the Full Sweep Overview")
print("- Type 'exit' to quit")
print("=" * 60)

while True:
    user_input = input("\nSelection: ").strip().lower()

    if user_input == 'exit':
        print("Exiting...")
        break

    plt.figure(figsize=(12, 8))  # Create a fresh window for this specific request

    if user_input == 'all':
        # Re-plot the Overview (All lines in one)
        for label, data in results.items():
            plt.plot(data["f"], data["TL"], color=color_map[label], linewidth=0.8, alpha=0.6)
        plt.title("Full Sweep Overview", fontweight='bold')
        print("Opening Overview Plot...")

    else:
        try:
            target_ar = float(user_input)
            match_labels = [l for l in all_labels if l.startswith(f"AR:{target_ar:.1f}")]

            if not match_labels:
                print(f"[!] No data found for AR: {target_ar:.1f}. Try another.")
                plt.close()  # Close the empty figure we just made
                continue

            sweep_cmap = plt.get_cmap('cool').resampled(len(match_labels))
            for j, label in enumerate(match_labels):
                plt.plot(results[label]["f"], results[label]["TL"],
                         color=sweep_cmap(j), label=label, linewidth=1.5)

            plt.title(f"Interactive View - Aspect Ratio: {target_ar:.1f}", fontweight='bold')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize='8')
            print(f"Opening interactive window for AR {target_ar:.1f}...")

        except ValueError:
            print("[!] Invalid input. Enter a number, 'all', or 'exit'.")
            plt.close()
            continue

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("TL (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # Only shows the CURRENTLY active figure