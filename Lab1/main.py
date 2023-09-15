import random
from Dot import Dot
import numpy as np  # v 1.19.2
import matplotlib.pyplot as plt  # v 3.3.2
import pandas as pd
import math
import sklearn.metrics
from sklearn.metrics import accuracy_score

def split_lines():
    df = pd.read_csv('dots.csv')
    dx = []
    dy = []
    dz = []
    for i in df:
        for j in df[i]:
            elements = j.split(";")

            dx.append(float(elements[0].strip()))
            dy.append(float(elements[1].strip()))
            dz.append(elements[2].strip())

    return dx, dy, dz


def generate_test_dots(nuber):
    x = []
    y = []
    color = []
    for i in range(nuber):
        x.append(random.randint(1, 9) + random.random())
        y.append(random.randint(1, 9) + random.random())
        color.append("#FFF673")
    return x, y, color


# def generate_test_dots(number):
#     x = [4,8,4,5]
#     y = [1,6,9,5]
#     # color = ['#007b25', '#025167', '#A61700', '#025167']
#     color = ['#FFF673', '#FFF673', '#FFF673', '#FFF673']
#     return x,y,color



def create_dot_data(xs, ys, colors):
    dots = []
    for i in range(len(xs)):
        dots.append(Dot(xs[i], ys[i], colors[i]))
    return dots


def create_circle_by_dots(ax, dots, target_dot):
    r = 0
    for i in dots:
        new_r = math.sqrt((target_dot.x - i.x) ** 2 + (target_dot.y - i.y) ** 2)
        if new_r > r:
            r = new_r

    circle1 = plt.Circle((target_dot.x, target_dot.y), r, color=target_dot.color, fill=False)
    ax.add_patch(circle1)


def calculate_evklid(dot1, dot2):
    return math.sqrt((dot1.x - dot2.x) ** 2 + (dot1.y - dot2.y) ** 2)


def calculate_knn(dots, test_dot, clasters):
    neighbors = []
    for dot in dots:
        neighbors.append({"dot": dot, "evk": calculate_evklid(test_dot, dot)})

    neighbors = sorted(neighbors, key=lambda x: x['evk'], reverse=False)
    neighbors_nearest = []
    for i in neighbors[:clasters]:
        neighbors_nearest.append(i["dot"])
    return neighbors_nearest


def color_dot(target_dot, knn):
    colors = {}
    for i in knn:
        if i.color in colors.keys():
            colors[i.color] += 1
        else:
            colors[i.color] = 1

    dominate_color_index = 0
    dominate_color = ""
    for i in colors:
        if colors[i] >= dominate_color_index:
            dominate_color_index = colors[i]
            dominate_color = i

    target_dot.color = dominate_color
    return target_dot


# Enter x and y coordinates of points and colors
xs, ys, colors = split_lines()
dots = create_dot_data(xs, ys, colors)

test_xs, test_ys, test_colors = generate_test_dots(4)
test_dots = create_dot_data(test_xs, test_ys, test_colors)

# Select length of axes and the space between tick labels
xmin, xmax, ymin, ymax = 0, 10, 0, 10
ticks_frequency = 1

# Plot points
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(xs, ys, c=colors)
ax.scatter(test_xs, test_ys, c=test_colors)

result_colors = []
for i in range(len(test_dots)):
    neighbors = calculate_knn(dots, test_dots[i], 3)
    test_dots[i] = color_dot(test_dots[i], neighbors)
    create_circle_by_dots(ax, neighbors, test_dots[i])
    result_colors.append(test_dots[i].color)



known_colors = ['#007B27', '#025176', '#A61709', '#025168']


# Set identical scales for both axes
ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

# Set bottom and left spines as x and y axes of coordinate system
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create 'x' and 'y' labels placed at the end of the axes
ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

# Create custom major ticks to determine position of tick labels
x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])

# Create minor ticks placed at each integer to enable drawing of minor grid
# lines: note that this has no effect in this example with ticks_frequency=1
ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

# Draw major and minor grid lines
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

# Draw arrows
arrow_fmt = dict(markersize=4, color='black', clip_on=False)
ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
print(result_colors)
print(known_colors)

r = sklearn.metrics.confusion_matrix(result_colors, known_colors)

print(r)
print(accuracy_score(result_colors, known_colors))


plt.show()
