from numpy import *

from matplotlib import pyplot as plt
from matplotlib import animation as animation

import pylab

import time

training_history = genfromtxt("training_history.csv", delimiter=",")
points = genfromtxt("data.csv", delimiter=",")

fig, axes = plt.subplots()

axes.set_xlim(20, 80)
axes.set_ylim(20, 130)

for point in points:
    axes.scatter(point[0], point[1], s=120, marker='.', linewidths=2)

line, = axes.plot([], [], 'r-')
iterationText = axes.text(0.025, 0.025, '', transform=axes.transAxes)

plt.show(block=False)

for item in training_history:
    iteration, b, m = item

    iterationText.set_text(f"Iteration: {iteration}")

    # Update function line
    xdata = range(90)
    ydata = [m * x + b for x in xdata]
    line.set_xdata(xdata)
    line.set_ydata(ydata)

    # Draw changes
    # axes.draw_artist(axes.patch)
    axes.draw_artist(line)

    time.sleep(1 / 600)
    fig.canvas.draw()

    print(time.clock())

pylab.show()
