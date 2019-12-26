from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
dir = "D:/hansol/Open3D2/examples/Python/Advanced/0920"
test_file = dir + "/eofe0.ply"
x, y, z = [], [], []

with open(test_file) as f:
    for index, line in enumerate(f):
        if index <=11:
            continue
        else:
            array_line = line.split()
            x.append(array_line[0])
            y.append(array_line[1])
            z.append(array_line[2])
x = map(float,x)
y = map(float,y)
z = map(float,z)
result = ax.scatter(x,y,z, c=z, s=10, alpha=1, cmap= plt.cm.rainbow, label = None)
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')
ax.set_zlabel('z_label')
fig.colorbar(result)

plt.show()