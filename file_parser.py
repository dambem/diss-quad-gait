import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def deg_to_rad(deg):
    return deg*(np.pi/180)
def rad_to_deg(rad):
    return rad/(np.pi/180)
def parse_ex2(gait):
    filenames = sorted(glob.glob('ex_2'+gait+'/experiment-osc*log.txt'))
    # print(len(filenames))
    values = np.zeros((len(filenames), 6, 2))
    # print(np.shape(values))
    val = 0
    for f in filenames:
        with open(f, 'r') as fp:
            line = fp.readline()
            maxforce = fp.readline()
            line_c = 0
            values[val, line_c, 0] = maxforce
            values[val, line_c, 1] = maxforce
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

def parse_angle(gait):
    filenames = sorted(glob.glob('ex_ang/experiment-osc'+gait+'0*log.txt'))
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

def parse_angle(gait):
    filenames = sorted(glob.glob('ex_ang/experiment-osc'+gait+'0*log.txt'))
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

# plt.scatter(values[:, 0, 0], values[:, 3, 0])
plt.show()
