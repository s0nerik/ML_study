from numpy import *

from matplotlib import pyplot as plt
from matplotlib import animation as animation

training_history = genfromtxt("training_history.csv", delimiter=",")
points = genfromtxt("data.csv", delimiter=",")

fig, axes = plt.subplots()

axes.set_xlim(20, 80)
axes.set_ylim(20, 130)

for point in points:
    axes.scatter(point[0], point[1], s=120, marker='.', linewidths=2)

line, = axes.plot([], [], 'r-')
iterationText = axes.text(0.025, 0.025, '', transform=axes.transAxes)
errorText = axes.text(0.55, 0.025, '', transform=axes.transAxes)

plt.show(block=False)


def update_data():
    for item in training_history:
        iteration, b, m, error = item

        iterationText.set_text(f"Iteration: {iteration}")
        errorText.set_text(f"Error: {error}")

        # Update function line
        xdata = range(90)
        ydata = [m * x + b for x in xdata]
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        yield None


def update_graph(_): pass

# noinspection PyTypeChecker
anim = animation.FuncAnimation(fig, update_graph, update_data, interval=16)

plt.show()
