import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Generate some example data
x = np.linspace(0, 10, 500)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x)*np.cos(x)

# Create the main figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y1, label='sin(x)')
ax.plot(x, y2, label='cos(x)')
ax.plot(x, y3, label='sin(x)*cos(x)')
ax.legend()
ax.grid(True)

# Define the region we want to zoom into
x1, x2 = 0, 5   # region in x to zoom
y1_lim, y2_lim = -0.2, 0.2  # region in y to zoom

# Create an inset axes with a zoom factor
zoom_factor = 2
axins = zoomed_inset_axes(ax, zoom_factor, loc='lower left', bbox_to_anchor=(0.5, 0.5), borderpad=0.5)
# You can change loc to something else like 'lower left' depending on where you want the inset

# Plot the same data onto the inset axes
axins.plot(x, y1)
axins.plot(x, y2)
axins.plot(x, y3)

# Set the limits of the inset to the zoom region
axins.set_xlim(x1, x2)
axins.set_ylim(y1_lim, y2_lim)

# Hide the tick labels in the inset if desired
plt.setp(axins.get_xticklabels(), visible=False)
#plt.setp(axins.get_yticklabels(), visible=False)

# Draw lines between the inset and the main plot to highlight the zoom region
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()