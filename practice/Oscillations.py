import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

t = np.linspace(0,20,4000)

a1 = 10
a2 = 8
a3 = 5

w1 = 16.7
w2 = 18
w3 = 230

phi1 = 0.9
phi2 = 3
phi3 = 8

x = a1 * np.cos(w1*t+phi1)
y = a2 * np.cos(w2 * t + phi2)
z = a3 * np.cos(w3 * t + phi3)

# 1. Setup the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# 2. Set axis limits (important so the "camera" doesn't jump)
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

# 3. Initialize an empty line object
line, = ax.plot([], [], [], lw=2)

# 4. Define the update function
def update(frame):
    # 'frame' is the current index (0 to 999)
    # Update X and Y
    line.set_data(x[:frame], y[:frame])
    # Update Z (specific to 3D plots)
    line.set_3d_properties(z[:frame])
    return line,

# 5. Create the animation
# frames=1000 to match your array size
# interval=4 (ms) to play through 4 seconds of data roughly in real-time
# (1000 frames / 4 seconds = 250 fps, which is very fast;
#  interval=20 or 40 is more standard for smooth viewing)
ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

plt.show()