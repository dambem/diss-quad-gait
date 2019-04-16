import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
def deg_to_rad(deg):
    return deg*(np.pi/180)
def rad_to_deg(rad):
    return rad/(np.pi/180)
def parse_big():
    filenames = sorted(glob.glob('big/*'))
    # print(len(filenames))
    values = np.zeros((len(filenames), 7, 3))
    # print(np.shape(values))
    val = 0
    for f in filenames:
        with open(f, 'r') as fp:
            line = fp.readline()
            line = line.split(":")
            maxforce = fp.readline()
            maxforce = maxforce.split(':')
            gait = fp.readline()
            leg_rot = fp.readline()
            hip_rot = fp.readline()
            line_c = 0
            force_val = float(maxforce[1])
            oscil_val = float(line[1])
            values[val, line_c, 0] = force_val
            line_c += 1
            values[val, line_c, 0] = oscil_val
            line_c += 1
            while line:
                line = fp.readline()
                data = line.split(':')
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
def dynamic_similarity(a, u, g, h, b, froude):
    # froude = froude_number(u, g, h)
    return (a*(froude)**b)

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
    data = np.array((val1[:,1,0]/(0.3) ,distance/10))

    # data =np.sort(data)
    # print(data)
    a = 2.4
    u = 2
    g = 9.8
    # h = 2
    b = 0.34
    h = np.linspace(0.1, 0.6, 50)
    for n in h:
        plt.scatter(n, dynamic_similarity(a,u,g,h,b, n), marker='s', c='red')
    plt.scatter(data[0], data[1])
    plt.title("Distance Against Froude Number")
    plt.xlabel("Froude Number")
    plt.ylabel("Distance")
    plt.show()

def plot_big():
    # 2 = velocity
    # 3 = froude
    # 4 = distance
    # 5 = cost
    # 6 = time period
    val1 = parse_big()

    cost_per_distance = val1[:,5,0]/val1[:,4,0]
    cost_per_distance = np.clip(cost_per_distance, 0, 1000)
    cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
    # values_between = np.where(froude_number <= 0.3)
    # print(values_between)
    froude = val1[:,3,0]/(0.3)
    valid_runs = np.where(cost_per_distance != 0, )
    average_cost = np.mean(cost_per_distance)
    print(average_cost)

    froude_ind = np.where(froude<=0.3)
    print(len(froude_ind[0]))
    # values_between = np.where(values_between[0] >= 0.1)
    # print(values_between[0])
    # print(len(values_between[0]))
    # print(len(val1[:,3,0]))
    print((len(froude_ind[0])/len(valid_runs[0]))*100)
    # # print(values_between)
    time_period = val1[:,6,0]
    amount_of_strides = np.ceil(20/(time_period))
    # print(amount_of_strides)
    distance = val1[:,4,0]

    relative_stride_length = distance/(amount_of_strides)
    # print(relative_stride_length)
    # relative_stride length =
    data = np.array((val1[:,3,0]/(0.3) , distance, val1[:,0,0], val1[:,1,0], cost_per_distance))

    # data =np.sort(data)
    # print(data)
    a = 2.4
    u = 2
    g = 9.8
    # h = 2
    b = 0.34
    h = np.linspace(0, 0.3, 100)
    for n in h:
        plt.scatter(n, dynamic_similarity(a,u,g,h,b, n), marker='s', c='red')
    plt.scatter(data[0], data[1])
    # plt.scatter()
    plt.title("Distance Against Froude Number")
    plt.xlabel("Froude Number")
    plt.ylabel("Distance")
    plt.show()
    fig, ax = plt.subplots()
    id = 0
    plt.xlabel("Froude Number")
    plt.ylabel("Distance")
    ax.scatter(data[0], data[1], s=cost_per_distance/10)
    ax.legend(val1[:,0,0])
    plt.show()
    fig = plt.figure()
    plt.title("Froude Number, Distance Travelled and Cost Of Locomotion")
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Distance")
    ax.set_ylabel("Cost Of Locomotion")
    ax.scatter(data[0],  cost_per_distance,data[1], c=data[2])
    plt.show()
def parse_big2(force, osc, l, h):
    filenames = sorted(glob.glob('big/f'+force+'o'+osc+'g0l'+l+"h"+h+"log.txt"))
    # print(len(filenames))
    values = np.zeros((len(filenames), 7, 3))
    # print(np.shape(values))
    val = 0
    for f in filenames:
        with open(f, 'r') as fp:
            line = fp.readline()
            line = line.split(":")
            maxforce = fp.readline()
            maxforce = maxforce.split(':')
            gait = fp.readline()
            leg_rot = fp.readline()
            hip_rot = fp.readline()
            line_c = 0
            force_val = float(maxforce[1])
            oscil_val = float(line[1])
            values[val, line_c, 0] = force_val
            line_c += 1
            values[val, line_c, 0] = oscil_val
            line_c += 1
            while line:
                line = fp.readline()
                data = line.split(':')
                if (len(data) == 1):
                    continue;
                val_d = float(data[0])
                val_sd = float(data[1])
                values[val, line_c, 0] = val_d
                values[val, line_c, 1] = val_sd
                values[val, line_c, 2] = force_val
                line_c += 1
            val += 1
            # values[val, line_c, 0] = hip_rot
            # values[val, line_c, 1] = leg_rot
    return(values)

def big_calc():
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
    leg = ["10", "11", "12", "13", "14", "15", "16", "17",  "18", "19", "20"]
    hip =  ["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
def plot_big2():
    # 2 = velocity
    # 3 = froude
    # 4 = distance
    # 5 = cost
    # 6 = time period
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
    leg = ["10", "11", "12", "13", "14", "15", "16", "17",  "18", "19", "20"]
    hip =  ["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    fig, ax = plt.subplots()
    for n in values:
        # for j in values:
        val = parse_big2("*", n, "*", "*")
        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, None)
        # cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
        distance = val[:,4,0]

        data = np.array((val[:,3,0]/(0.3) , distance, val[:,0,0], val[:,1,0], cost_per_distance))



        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Oscillator Time Step")
        ax.set_xlabel("Oscillator Time Steps")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/0.3, label=n, yerr=np.mean(val[:,3,1]), color='blue')
        # ax.legend()
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Oscillator Time Step")
    ax.set_xticks([0, 0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5, 1.6,1.7,1.8,1.9,2.0, 2.1, 2.2])
    ax.set_ylabel("Cost Of Locomotion Per Unit Distance")
    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13"]
    markers = ["D", "x", "o", "s", "*", "h", "v", "1", "2", "3"]
    for g in range(len(values)):
        for n in forces:
            val = parse_big2(n, values[g], "*", "*")
            cost_per_distance = val[:,5,0]/val[:,4,0]
            cost_per_distance = np.clip(cost_per_distance, 0, 1500)
            # cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
            distance = val[:,4,0]
            arrayvalues = np.where(cost_per_distance > 0)
            # for j in arrayvalues[0]:
            data = np.array((val[:,3,0]/(0.3) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
            froude = val[:,3,0]/(0.3)
            froude_ind = np.where(np.logical_and(froude<=0.3, froude>=0.1))
            # valid_run = np.where(froude  0)
            # run_indices = valid_run[0]
            average_cost = np.mean(cost_per_distance)
            # for j in run_indices:
            #     # print((len(froude_ind[0])/len(valid_runs[0]))*100)
            #     # plt.title("Froude Number, Distance Travelled and Cost Of Locomotion")
            ax.scatter(froude,  cost_per_distance, float(values[g]), label=n)
    ax.legend(forces, title="Max Force")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Max Force")
    # ax.set_xticks([0, 0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5, 1.6,1.7,1.8,1.9,2.0, 2.1, 2.2])
    ax.set_ylabel("Hip Rotation (Degrees)")
    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    markers = ["D", "x", "o", "s", "*", "h", "v", "1", "2", "3"]
    for l in leg:
        for g in range(len(values)):
            for n in forces:
                val = parse_big2(n, values[g], l, "10")
                cost_per_distance = val[:,5,0]/val[:,4,0]

                cost_per_distance = np.clip(cost_per_distance, 0, 1500)
                # cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
                distance = val[:,4,0]
                # arrayvalues = np.where(cost_per_distance > 0)
                # for j in arrayvalues[0]:
                data = np.array((val[:,3,0]/(0.3) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
                froude = val[:,3,0]/(0.3)
                froude_ind = np.where(np.logical_and(froude<=0.3, froude>=0.1))
                # valid_run = np.where(froude  0)
                # run_indices = valid_run[0]
                average_cost = np.mean(cost_per_distance)
                # for j in run_indices:
                #     # print((len(froude_ind[0])/len(valid_runs[0]))*100)
                #     # plt.title("Froude Number, Distance Travelled and Cost Of Locomotion")
                ax.plot_surface(float(l), float(n), froude)
        # ax.legend(values, title="Oscillations")
    plt.show()

    fig, ax = plt.subplots()
    force_p = []
    for n in forces:
        # for j in values:
        val = parse_big2(n, "*", "*", "*")
        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, 1000)
        # cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
        froude = val[:,3,0]/(0.3)
        froude_ind = np.where(np.logical_and(froude<=0.4, froude>=0.01))
        perc = len(froude_ind[0])/len(val[:,5,0])
        print(str(perc*100) + "%")
        force_p.append([perc, float(n)])

        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Force Applied")
        # plt.title("Laikago Performing Walking Gait In PyBullet")
        ax.set_xlabel("Max Force Applied")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/0.3, label=n, yerr=np.mean(val[:,3,1])*2, color='green')
        # ax.legend()
    plt.show()
    fig, ax = plt.subplots()
    # for n in force_p:
    #     ax.plot(n[0]*100, n[1])
    # plt.show()

    val = parse_big2("*","*", "*", "*")
    distance = val[:,4,0]
    # print(distance)
    print
    valid_dist = np.where(distance >= 0.5)
    print(len(valid_dist[0]))
    print(len(val[:,4,0]))

    print("Overall %")
    froude = val[:,3,0]/(0.3)
    # print(np.where(froude >= 0.01))
    froude_ind = np.where(np.logical_and(froude<=0.4, froude != 0))
    # froude_zeros = np.where(froude==0)
    # zero = str(len(froude_zeros)) + "%"
    # print(zero)
    perc = len(froude_ind[0])/len(val[:,5,0])
    print(str(perc*100) + "%")

    for n in values:
        val = parse_big2("*", n, "*", "*")
        froude = val[:,3,0]/(0.3)

    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    fig, ax = plt.subplots()
    for n in leg:
        # for j in values:
        val = parse_big2("*", "*", n, "*")

        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, 1000)
        cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)

        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Leg Rotation")
        ax.set_xlabel("Leg Rotation")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/0.3, label=n, yerr=np.mean(val[:,3,1]), color='red')
        # ax.legend()
    plt.show()
    fig, ax = plt.subplots()
    hip=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]

    for n in hip:
        # for j in values:
        val = parse_big2("*", "*", "*", n)

        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, 1000)
        cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)

        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Leg Rotation")
        ax.set_xlabel("Leg Rotation")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/0.3, label=n, yerr=np.mean(val[:,3,1]), color='red')
        # ax.legend()
    plt.show()
# plot_big()
# plot_big2
plot_big2()
# plot_ex2()
# plot_angles()
# plt.scatter(values[:, 0, 0], values[:, 3, 0])
