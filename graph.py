import matplotlib.pyplot as plt
import numpy as np

labels = ['M1\nLow Compute\nLow Memory', 'M2\nMedium Compute\nHigh Memory', 'M3\nLow Compute\nHigh Memory', 'M4\nHigh Compute\nLow Memory', 'M5\nHigh Compute\nHigh Memory']
men_means = np.array([1, 2, 0, 10, 6])
women_means = np.array([3, 10, 5, 0, 0])

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Program 1 (compute bound)')
rects2 = ax.bar(x + width/2, women_means, width, label='Program 2 (memory bound)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Affinity')
ax.set_title('Affinities by machine type and program')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()