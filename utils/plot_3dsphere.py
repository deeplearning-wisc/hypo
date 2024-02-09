import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot unit sphere with sparser gridlines
phi = np.linspace(0, np.pi, 30)  # Reduced from 100 to 50 for sparser gridlines
theta = np.linspace(0, 2 * np.pi, 30)  # Reduced from 100 to 50 for sparser gridlines
phi, theta = np.meshgrid(phi, theta)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

ax.plot_surface(x, y, z, color='c', alpha=0.1, linewidth=0, antialiased=False)

# Set axis labels
ax.set_xlabel('X-axis', fontsize=12, labelpad=10)
ax.set_ylabel('Y-axis', fontsize=12, labelpad=10)
ax.set_zlabel('Z-axis', fontsize=12, labelpad=10)

# Customize grid and background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Disable the X, Y, Z axes
ax.set_axis_off()

# Set the background transparent
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Save and show the plot with transparent background
plt.savefig('3d_plot_unit_sphere_no_title_sparser_grid_transparent_background.png', dpi=300, bbox_inches='tight', transparent=True)

