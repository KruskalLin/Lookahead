from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from optim_lookahead import Lookahead
from mpl_toolkits.mplot3d import Axes3D

# noisy hills of the cost function
def f1(x, y):
    return -1 * torch.sin(x * x) * torch.cos(3 * y * y) * torch.exp(-(x * y) * (x * y)) - torch.exp(-(x + y) * (x + y))


# bivar gaussian hills of the cost function
def f2(x, y, x_mean, y_mean, x_sig, y_sig):
    normalizing = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (x - x_mean).pow(2)) / (2 * (x_sig ** 2))
    y_exp = (-1 * (y - y_mean).pow(2)) / (2 * (y_sig ** 2))
    return normalizing * torch.exp(x_exp + y_exp)

def cost_function(x, y):
    z = -1 * f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)
    # three steep gaussian trenches
    z -= f2(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= f2(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)

    return z


# pyplot settings
fig = plt.figure(figsize=(3, 2), dpi=300)
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
params = {'legend.fontsize': 3,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.axis('off')


# visualize cost function as a contour plot
x_val = y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
z_val_mesh_flat = cost_function(torch.DoubleTensor(x_val_mesh_flat), torch.DoubleTensor(y_val_mesh_flat))
z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
levels = np.arange(-10, 1, 0.05)
# ax.contour(x_val_mesh, y_val_mesh, z_val_mesh, levels, alpha=.7, linewidths=0.4)
# ax.plot_wireframe(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.5, linewidths=0.4, antialiased=True)
ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh.numpy(), alpha=.4, cmap=cm.coolwarm)
plt.draw()

# starting location for variables
x_i = 0.75
y_i = 1.0

# create variable pair (x, y) for each optimizer
x_var, y_var = [], []
for i in range(4):
    x_var.append(torch.nn.Parameter(torch.DoubleTensor([x_i]), requires_grad=True))
    y_var.append(torch.nn.Parameter(torch.DoubleTensor([y_i]), requires_grad=True))


ops_param = np.array([['SGD', 'b'],
                     ['Adam', 'g'],
                     ['Lookahead_SGD', 'r'],
                     ['Lookahead_Adam', 'm']])

ops = []
ops.append(optim.SGD([x_var[0], y_var[0]], lr=10))
ops.append(optim.Adam([x_var[1], y_var[1]], lr=1e-1))
ops.append(Lookahead(optim.SGD([x_var[2], y_var[2]], lr=10)))
ops.append(Lookahead(optim.Adam([x_var[3], y_var[3]], lr=1e-1)))

# 3d plot camera zoom, angle
xlm = ax.get_xlim3d()
ylm = ax.get_ylim3d()
zlm = ax.get_zlim3d()
ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
azm = ax.azim
ele = ax.elev + 40
ax.view_init(elev=ele, azim=azm)


# use last location to draw a line to the current location
last_x, last_y, last_z = [], [], []
plot_cache = [None for _ in range(len(ops))]

# loop each step of the optimization algorithm
steps = 100
for iter in range(steps):
    for i, op in enumerate(ops):
        op.zero_grad()
        loss = cost_function(x_var[i], y_var[i])
        loss.backward()
        op.step()
        # move dot to the current value
        if plot_cache[i]:
            plot_cache[i] = None
        plot_cache[i] = ax.scatter(x_var[i].item(), y_var[i].item(), loss.item(), s=1, label=ops_param[i, 0], color=ops_param[i, 1])

        # draw a line from the previous value
        if iter == 0:
            last_z.append([loss.item()])
            last_x.append([x_i])
            last_y.append([y_i])
        ax.plot([last_x[i][-1], x_var[i].item()], [last_y[i][-1], y_var[i].item()], [last_z[i][-1], loss.item()], linewidth=0.5, color=ops_param[i, 1])
        last_x[i].append(x_var[i].item())
        last_y[i].append(y_var[i].item())
        last_z[i].append(loss.item())

    if iter == 0:
        plt.legend(plot_cache, ops_param[:, 0])

    plt.savefig('figures/' + str(iter) + '.png')
    print('iteration: {}'.format(iter))

    plt.pause(0.001)

print("done")
