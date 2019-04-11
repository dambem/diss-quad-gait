import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
def deg_to_rad(deg):
    return deg*(np.pi/180)
def rad_to_deg(rad):
    return rad/(np.pi/180)
def parse_ex2(gait):
    filenames = sorted(glob.glob('ex2_1/experiment-osc'+gait+'0*'))
    # print(len(filenames))
    values = np.zeros((len(filenames), 5, 3))
    # print(np.shape(values))
    val = 0
    for f in filenames:
        with open(f, 'r') as fp:
            line = fp.readline()
            maxforce = fp.readline()
            maxforce = maxforce.split(':')
            line_c = 0
            force_val = float(maxforce[1])
            while line:
                line = fp.readline()
                data = line.split(':S')
                if (len(data) == 1):
                    continue;
                val_d = float(data[0])
                val_sd = float(data[1])
                values[val, line_c, 0] = val_d
                values[val, line_c, 1] = val_sd
                values[val, line_c, 2] = force_val
                line_c += 1
            val += 1
    return(values)
def parse_angle(gait):
    filenames = sorted(glob.glob('ex_ang/experiment-osc0*log.txt'))
    # print(len(filenames))
    values = np.zeros((len(filenames), 6, 2))
    # print(np.shape(values))
    val = 0
    for f in filenames:
        with open(f, 'r') as fp:
            line = fp.readline()
            maxforce = fp.readline()
            foot = fp.readline()
            footangle = foot.split(':')
            footangle = (rad_to_deg(float(footangle[1])))
            hip = fp.readline()
            hipangle = hip.split(':')
            hipangle = (rad_to_deg(float(hipangle[1])))
            line_c = 0
            values[val, line_c, 0] = footangle
            values[val, line_c, 1] = hipangle
            line_c += 1
            while line:
                line = fp.readline()
                data = line.split(':S')
                if (len(data) == 1):
                    continue;
                val_d = float(data[0])
                val_sd = float(data[1])
                values[val, line_c, 0] = val_d
                values[val, line_c, 1] = val_sd
                line_c += 1
            val += 1
    return(values)

def plot_angles():
    val1 = parse_angle("0")
    val2 = parse_angle("1")
    val3 = parse_angle("2")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(val1[:,0,0], val1[:,0,1], val1[:,3,0])
    ax.plot_trisurf(val2[:,0,0], val2[:,0,1], val2[:,3,0])
    ax.plot_trisurf(val3[:,0,0], val3[:,0,1], val3[:,3,0])
    plt.show()

def plot_ex2():
    val1 = parse_ex2("0")
    val2 = parse_ex2("1")
    val3 = parse_ex2("2")

    cost_per_distance = val1[:,3,0]/val1[:,2,0]
    cost_per_distance = np.clip(cost_per_distance, 0, 1000)
    cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
    time_period = val1[:,4,0]
    amount_of_strides = np.ceil(20/(time_period*2))
    print(amount_of_strides)
    distance = val1[:,2,0]

    relative_stride_length = distance/(amount_of_strides)
    print(relative_stride_length)
    # relative_stride length =
    data = np.array((val1[:,1,0]/(0.3) ,relative_stride_length))

    # data =np.sort(data)
    # print(data)

    plt.scatter(data[0], data[1])
    plt.xlabel("Froude Number")
    plt.ylabel("Distance")
    plt.show()
plot_ex2()
# plot_angles()
# plt.scatter(values[:, 0, 0], values[:, 3, 0])
