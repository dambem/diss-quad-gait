import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.stats as stats
from scipy.stats import norm
import pandas as pd
import sklearn.metrics as sk
height = 0.4

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
def dynamic_similarity(a, g, b, froude):
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
    data = np.array((val1[:,1,0]/(height) ,distance/10))

    # data =np.sort(data)

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
    # values_between = np.where(froude_number <= height)
    # print(values_between)
    froude = val1[:,3,0]/(height)
    valid_runs = np.where(cost_per_distance != 0, )
    average_cost = np.mean(cost_per_distance)
    print(average_cost)

    froude_ind = np.where(froude<=height)
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
    data = np.array((val1[:,3,0]/(height) , distance, val1[:,0,0], val1[:,1,0], cost_per_distance))

    # data =np.sort(data)
    # print(data)
    a = 2.4
    u = 2
    g = 9.8
    # h = 2
    b = 0.34
    h = np.linspace(0, height, 100)
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
def parse_t(force, osc, l, h):
    filenames = sorted(glob.glob('ttest/f'+force+'o'+osc+"g0lttesth"+l+"*"))
    # print(len(filenames))
    values = np.zeros((len(filenames), 11, 3))
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
            leg_p = fp.readline()
            hip_p = fp.readline()
            leg_t = fp.readline()
            hip_t = fp.readline()
            pvalleg = leg_p.split(':')
            pvalhip = hip_p.split(':')
            tvalleg = leg_t.split(':')
            tvalhip = hip_t.split(':')
            leg_p_val = float(pvalleg[1])
            leg_t_val = float(tvalleg[1])
            hip_p_val = float(pvalhip[1])
            hip_t_val = float(tvalhip[1])
            line_c = 0
            force_val = float(maxforce[1])
            oscil_val = float(line[1])
            values[val, line_c, 0] = force_val
            line_c += 1
            values[val, line_c, 0] = oscil_val
            line_c += 1
            values[val, line_c, 0] = leg_t_val
            line_c += 1
            values[val, line_c, 0] = leg_p_val
            line_c += 1
            values[val, line_c, 0] = hip_t_val
            line_c += 1
            values[val, line_c, 0] = hip_p_val
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

def plot_t():
    # 2 = leg t
    # 3 = leg p
    # 4 = hip t
    # 5 = hip p
    forces = ["20", "30", "40", "50", "60", "70", "80", "90", "100"]
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    fig,ax = plt.subplots()
    for n in forces:
        for j in values:
            val = parse_t(n, j, "*", "*")
            pvalue = val[:, 3, 0]
            print(val)
            ax.bar(n, np.mean(pvalue), label=n, color='blue')

    plt.show()

    val = parse_t("*", "*", "*", "*")
    fig, ax = plt.subplots()
    pvalue = (val[:,6,0])
    pvalue.sort()
    pvaluemean = np.mean(pvalue)
    pvaluestd = np.std(pvalue)
    normp = stats.norm.pdf(pvalue, pvaluemean, pvaluestd)
    ax.plot(pvalue, normp) # including h here is crucial
    plt.show()

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

        data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))



        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Oscillator Time Step")
        ax.set_xlabel("Oscillator Time Steps")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/height, label=n, yerr=np.mean(val[:,3,1]), color='blue')
        # ax.legend()
    # plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Oscillator Time Step")
    ax.set_xticks([0, 0.1,0.2,height,height, 0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5, 1.6,1.7,1.8,1.9,2.0, 2.1, 2.2])
    ax.set_ylabel("Cost Of Locomotion Per Unit Distance")
    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "15", "16", "17","18"]
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
            data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
            froude = val[:,3,0]/(height)
            froude_ind = np.where(np.logical_and(froude<=height, froude>=0.1))
            # valid_run = np.where(froude  0)
            # run_indices = valid_run[0]
            average_cost = np.mean(cost_per_distance)
            # for j in run_indices:
            #     # print((len(froude_ind[0])/len(valid_runs[0]))*100)
            #     # plt.title("Froude Number, Distance Travelled and Cost Of Locomotion")
            ax.scatter(froude,  cost_per_distance, float(values[g]), label=n)
    ax.legend(forces, title="Max Force")
    # plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Max Force")
    # ax.set_xticks([0, 0.1,0.2,height,height, 0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5, 1.6,1.7,1.8,1.9,2.0, 2.1, 2.2])
    ax.set_ylabel("Hip Rotation (Degrees)")
    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
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
                data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
                froude = val[:,3,0]/(height)
                froude_ind = np.where(np.logical_and(froude<=height, froude>=0.1))
                # valid_run = np.where(froude  0)
                # run_indices = valid_run[0]
                # for n in range()
                average_cost = np.mean(cost_per_distance)
                # for j in run_indices:
                #     # print((len(froude_ind[0])/len(valid_runs[0]))*100)
                #     # plt.title("Froude Number, Distance Travelled and Cost Of Locomotion")
                ax.scatter(float(l), float(n), froude)
        # ax.legend(values, title="Oscillations")
    # plt.show()

    fig, ax = plt.subplots()
    force_p = []
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    for n in values:
        # for j in values:
        val = parse_big2("*", n, "*", "*")
        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, 1000)
        # cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
        froude = val[:,3,0]/(height)
        # valid_gaits
        froude_ind = np.where(np.logical_and(froude<=height, froude >= 0.001))
        perc = len(froude_ind[0])/len(val[:,5,0])
        print(str(n))
        print(str(perc*100) + "%")
        force_p.append([perc, float(n)])
        # data = np.array([n,np.mean(val[:,3,0])/height])
        std = (np.mean(val[:,3,1])/height)
        mean = (np.mean(val[:,3,0])/height)
        # mu, std = norm.fit()
        # ax.set_xlabel("Froude Number")
        # ax.set_zlabel("Distance")
        # ax.set_ylabel("Cost Of Locomotion")
        # ax.scatter(val[:,3,0],  float(j), float(n))
        plt.title("Average Froude Number against Force Applied")
        # plt.title("Laikago Performing Walking Gait In PyBullet")
        ax.set_xlabel("Max Force Applied")
        ax.set_ylabel("Average Froude Number")
        # ax.boxplot(n, mean, label=n, yerr=std, color='green')

        # ax.legend()
    # plt.show()
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
    froude = val[:,3,0]/(height)
    # print(np.where(froude >= 0.01))
    froude_ind = np.where(np.logical_and(froude<=height, froude >= 0.001))
    # froude_zeros = np.where(froude==0)
    # zero = str(len(froude_zeros)) + "%"
    # print(zero)
    perc = len(froude_ind[0])/len(val[:,5,0])
    print(str(perc*100) + "%")

    for n in values:
        val = parse_big2("*", n, "*", "*")
        froude = val[:,3,0]/(height)

    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
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
        ax.bar(n, np.mean(val[:,3,0])/height, label=n, yerr=np.mean(val[:,3,1]), color='red')
        # ax.legend()

    # val = parse_big2("*", "*", "*", "*")
    # fig, ax = plt.subplots()
    # froude = (val[:,3,0]/height)
    # # froude = np.where(froude > 0, froude, 0)
    # froude.sort()
    # froudemean = np.mean(froude)
    # froudestd = np.std(froude)
    # norm = stats.norm.pdf(froude, froudemean, std)
    # plt.title("Distribution Froude Number Values , μ= " + str(round(froudemean, 3)) + " , σ²= " + str(round(froudestd**2, 3)))
    # ax.set_xlabel("Froude Number")
    # ax.set_ylabel("Probability")
    # ax.plot(froude, norm) # including h here is crucial
    #
    # val = parse_big2("*", "*", "*", "*")
    # fig, ax = plt.subplots()
    # cost = (val[:,6,0]/val[:,4,0])
    # cost = np.where(cost > 0, cost, 0)
    # cost.sort()
    # costmean = np.mean(cost)
    # froudestd = np.std(cost)
    # norm = stats.norm.pdf(cost, costmean, froudestd)
    # plt.title("Distribution Of Cost Of Locomotion Values")
    # ax.set_xlabel("Cost Of Locomotion")
    # ax.set_ylabel("Probability")
    # ax.plot(cost, norm) # including h here is crucial


    fig, ax = plt.subplots()
    froude = (val[:,3,0]/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    plt.title("Distribution Of Froude Number Values")
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Probability")
    forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
    lege = ["All", "020", "030", "040", "050", "060", "070", "080", "090", "100"]
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    lege = []
    legen = "All- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2, 3))
    lege.append(legen)
    ax.plot(froude, normf) # including h here is crucial
    for n in values:
        val = parse_big2("*", n, "*", "*")
        froude = (val[:,3,0]/height)
        froude.sort()
        froudemean = np.mean(froude)
        froudestd = np.std(froude)
        normf = stats.norm.pdf(froude, froudemean, froudestd)
        ax.plot(froude, normf)
        legen = n + "- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2,3))
        lege.append(legen)
    ax.legend(lege, title="Oscillator Values")


    fig, ax = plt.subplots()
    val = parse_big2("*", "*", "*", "*")

    froude = (val[:,3,0]/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    plt.title("Distribution Of Froude Number Values")
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Probability")
    forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]

    lege = []
    legen = "All- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2, 3))
    lege.append(legen)
    ax.plot(froude, normf) # including h here is crucial
    for n in forces:
        val = parse_big2(n, "*", "*", "*")
        froude = (val[:,3,0]/height)
        froude.sort()
        froudemean = np.mean(froude)
        froudestd = np.std(froude)
        normf = stats.norm.pdf(froude, froudemean, froudestd)
        ax.plot(froude, normf)
        legen = n + "- " + " μ: " +str(round(froudemean, 3))+ ", σ²: " + str(round(froudestd**2, 3))
        lege.append(legen)
    ax.legend(lege, title="Max Force")

    plt.show()
def plot_ttest():
    forces = ["20", "30", "40", "50", "60", "70", "80", "90", "100"]
    plt.xlabel("Max Force")
    plt.ylabel("P value")
    val = parse_t("*", "*", "*", "*")
    force = val[:,0,0]
    osc = val[:,1,0]
    pleg = val[:,5,0]
    phip = val[:,6,0]
    pearson = stats.pearsonr(osc, pleg)
    print(pearson)
    pearson_osc = 0.49
    pearson_force = 0.10
    # plt.scatter(osc, pleg)

    pearson_force = 0.499
    plt.title("P paired t-test values against force - Pearson Correlation Coefficient: " + str(pearson))
    forces = ["20", "30", "40", "50", "60", "70", "80", "90", "100"]
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    x = []
    y = []
    # for n in values:
    table = np.zeros((len(forces), len(values)))
    f_count = 0
    v_count = 0
    for n in forces:
        v_count = 0
        for j in values:
            val = parse_t(n, j, "5", "5")
            force = val[:,0,0]
            osc = val[:,1,0]
            pleg = val[:,5,0]
            phip = val[:,6,0]
            table[f_count, v_count] = pleg
            v_count +=1
        f_count+=1
            # pearson = stats.pearsonr(force, pleg)
    print(pd.DataFrame(table))
    plt.imshow(table, cmap="hot")
    locs, labels = plt.xticks()
    ax = plt.gca()
    plt.title("Succesful Gait Systematic Test (Bounding)")
    plt.xlabel("Oscillator time-steps")
    plt.ylabel("Max force applied")
    plt.xticks(np.arange(0,len(values), step=1), values)
    plt.yticks(np.arange(0,len(forces), step=1), forces)
    ax.set_xticks(np.arange(-.5, len(values), 1), minor=True);
    ax.set_yticks(np.arange(-.5, len(forces), 1), minor=True);
    ax.grid(which = "minor", color = "grey", linestyle="--", linewidth=2)
            # plt.scatter(j, phip)
            # x =(force)
            # y= (leg)
            # pearson = stats.pearsonr(x, y)
            # print(pearson)
    # plt.scatter(force, phip)
    plt.show()

def plot_froude():
    fig, ax = plt.subplots()
    # 0.010/0.002
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    plt.title("Relative Stride Length against Froude Number")
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Relative Stride Length")
    lege = []
    # a = 2.4
    # g = 9.8
    # # h = 2
    # b = 0.34
    # h = np.linspace(0, height, 500)
    # med_x = []
    # med_y = []
    # for n in h:
    #     med_x.append(n)
    #     med_y.append(dynamic_similarity(a,g,b,n))
    #     # plt.plot(n, dynamic_similarity(a,g,b,n))
    # plt.plot(med_x, med_y, linestyle='--')
    # lege.append("Froude Cursorial Walking")
    #
    # a = 2.4
    # g = 9.8
    # # h = 2
    # b = 0.24
    # h = np.linspace(0, height, 500)
    # med_x = []
    # med_y = []
    # for n in h:
    #     med_x.append(n)
    #     med_y.append(dynamic_similarity(a,g,b,n))
    #     # plt.plot(n, dynamic_similarity(a,g,b,n))
    # plt.plot(med_x, med_y, linestyle='--')
    # lege.append("Froude Cursorial Walking Upper Bound")

    a = 2.4
    g = 9.8
    # h = 2
    b = 0.44
    h = np.linspace(0, height, 500)
    med_x = []
    med_y = []
    for n in h:
        med_x.append(n)
        med_y.append(dynamic_similarity(a,g,b,n))
        # plt.plot(n, dynamic_similarity(a,g,b,n))
    plt.plot(med_x, med_y, linestyle='--')
    lege.append("Froude Cursorial Walking Lower Bound")


    val = parse_big2("*", "*", "*", "*")
    osc_t = val[:,1,0]
    froude = (np.round(val[:,3,0], 3)/height)
    time_period = (0.001/osc_t)*(np.pi)
    time_period_s = (0.001/osc_t)
    number_of_strides = 20/time_period
    distance= (val[:,4,0])
    stride_length = distance/number_of_strides
    relative_stride_length = (stride_length/height)
    pearson = stats.pearsonr(froude, number_of_strides)

    print(str(pearson) + ": Pearson - All Data")
    # values = ["10", "5"]
    values = [["0.002", 0.56], ["0.004", 0.28]]
    for n in values:
        val = parse_big2("*", n[0], "*", "10")
        osc_t = val[:,1,0]
        froude = (np.round(val[:,3,0], 3)/height)
        time_period = n[1]
        # time_period_s = (0.001/osc_t)
        number_of_strides = 20/time_period
        distance= (val[:,4,0])
        stride_length = distance/number_of_strides
        relative_stride_length = (stride_length/height)
        lege.append("Oscillator: "+ str(n[0]) + " (Time Period: " + str(n[1]) + " )")
        plt.scatter(froude, relative_stride_length, s=5)
        g = 9.8
        # h = 2
        b = 0.44
        print(np.max(froude))
        h = np.linspace(0, np.max(froude), len(froude))
        med_x = []
        med_y = []
        for n in h:
            med_x.append(n)
            med_y.append(dynamic_similarity(a,g,b,n))

        ttest =  sk.r2_score([med_x, med_y], [froude, relative_stride_length],  multioutput='variance_weighted')
        print(str(ttest) + "-TTEST")

        pearson = stats.pearsonr(froude, number_of_strides)
        print(str(pearson) + ": Pearson " + str(n))
    plt.xlim(-0.01, 0.4)
    plt.ylim(0, 2)
    plt.legend(lege)

    # a = 2.4
    # g = 9.8
    # # h = 2
    # b = 0.44
    # h = np.linspace(0, height, 50)
    # med_x = []
    # med_y = []
    # for n in h:
    #     med_x.append(n)
    #     med_y.append(dynamic_similarity(a,g,b,n))
    #     # plt.plot(n, dynamic_similarity(a,g,b,n))
    # plt.plot(med_x, med_y)
    #
    # a = 2.4
    # g = 9.8
    # # h = 2
    # b = 0.24
    # h = np.linspace(0, height, 50)
    # med_x = []
    # med_y = []
    # for n in h:
    #     med_x.append(n)
    #     med_y.append(dynamic_similarity(a,g,b,n))
    #     # plt.plot(n, dynamic_similarity(a,g,b,n))
    # plt.plot(med_x, med_y)

    plt.show()

    # fig, ax = plt.subplots()
    # hip=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    #
    # for n in hip:
    #     # for j in values:
    #     val = parse_big2("*", "*", "*", n)
    #
    #     phase_difference = ((1/500)/float(n)*2)
    #     period = val[:,6,0]
    #     period2 = period/phase_difference
    #     cost_per_distance = val[:,5,0]/val[:,4,0]
    #     cost_per_distance = np.clip(cost_per_distance, 0, 1000)
    #     cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)
    #
    #     # ax.set_xlabel("Froude Number")
    #     # ax.set_zlabel("Distance")
    #     # ax.set_ylabel("Cost Of Locomotion")
    #     # ax.scatter(val[:,3,0],  float(j), float(n))
    #     plt.title("Average Froude Number against Leg Rotation")
    #     ax.set_xlabel("Leg Rotation")
    #     ax.set_ylabel("Average Froude Number")
    #     ax.bar(n, np.mean(val[:,3,0])/height, label=n, yerr=np.mean(val[:,3,1]), color='red')
    #     # ax.legend()
    # plt.show()
# plot_big()
# plot_big2
    # print(data)

def parse_t(force, osc, l, h):
    filenames = sorted(glob.glob('ttest/f'+force+'o'+osc+"g0lttesth"+l+"*"))
    # print(len(filenames))
    values = np.zeros((len(filenames), 11, 3))
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
            leg_p = fp.readline()
            hip_p = fp.readline()
            leg_t = fp.readline()
            hip_t = fp.readline()
            pvalleg = leg_p.split(':')
            pvalhip = hip_p.split(':')
            tvalleg = leg_t.split(':')
            tvalhip = hip_t.split(':')
            leg_p_val = float(pvalleg[1])
            leg_t_val = float(tvalleg[1])
            hip_p_val = float(pvalhip[1])
            hip_t_val = float(tvalhip[1])
            line_c = 0
            force_val = float(maxforce[1])
            oscil_val = float(line[1])
            values[val, line_c, 0] = force_val
            line_c += 1
            values[val, line_c, 0] = oscil_val
            line_c += 1
            values[val, line_c, 0] = leg_t_val
            line_c += 1
            values[val, line_c, 0] = leg_p_val
            line_c += 1
            values[val, line_c, 0] = hip_t_val
            line_c += 1
            values[val, line_c, 0] = hip_p_val
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

def parse_struc(force, osc, g):
    name ="struc/f"+force+"o"+osc+"g"+g+"lstruch10log.txt"
    filenames = sorted(glob.glob(name))
    values = np.zeros((len(filenames), 2))
    val = 0
    #
    for f in filenames:
        with open(f, 'r') as fp:
            distance = fp.readline()
            froude = fp.readline()
            values[val, 0] = distance
            values[val, 1] = froude
        val += 1
    return(values)

def parse_gaits(force, osc, g):
    name ="gait_var/f"+force+"o"+osc+"g"+g+"lgait_varh10log.txt"
    filenames = sorted(glob.glob(name))
    values = np.zeros((len(filenames), 2))
    val = 0
    #
    for f in filenames:
        with open(f, 'r') as fp:
            distance = fp.readline()
            froude = fp.readline()
            values[val, 0] = distance
            values[val, 1] = froude
        val += 1
    return(values)


def plot_experiments():
    osc = ["0.010", "0.008", "0.006", "0.004", "0.002", "0.001", "0.0008", "0.0006"]
    # osc = ["0.010", "0.008", "0.006", "0.004", "0.002", "0.001", "0.0008", "0.0006", "0.0004", "0.0002"]

    force = ["010", "020", "030", "040", "050", "060", "070", "080", "090" ,"100", "110","120"]
    # force = [ "020", "030", "040", "050", "060", "070", "080", "090" ,"100"]

    gait = ["0", "1", "2"]
    # matrix = []
    # np.array()
    fig = plt.figure()
    plt.subplot(131)
    ax = plt.gca()
    xval = 0
    yval =0
    leng = len(parse_t("*", "*", "0"))
    # array =np.array((leng/2, leng/2))

    array = np.zeros((len(osc), len(force)))
    # array = []
    val = 0
    for n in osc:
        yval = 0
        for j in force:
            print(yval)
            print(xval)
            val = parse_t(j, n, "0", "*")
            if (val[0][0] < 1):
                array[xval, yval] = 0
            else:
                array[xval, yval] = 1
            yval += 1
        xval +=1
    plt.imshow(array, cmap="hot")
    locs, labels = plt.xticks()
    print(labels)
    plt.title("Succesful Gait Systematic Test (Walking)")
    plt.ylabel("Oscillator time-steps")
    plt.xlabel("Max force applied")
    plt.yticks(np.arange(0,len(osc), step=1), osc)
    plt.xticks(np.arange(0,len(force), step=1), force)
    ax.set_yticks(np.arange(-.5, len(osc), 1), minor=True);
    ax.set_xticks(np.arange(-.5, len(force), 1), minor=True);
    ax.grid(which = "minor", color = "grey", linestyle="--", linewidth=2)

    plt.subplot(132)
    ax = plt.gca()
    xval = 0
    yval =0
    leng = len(parse_t("*", "*", "1"))
    # array =np.array((leng/2, leng/2))

    array = np.zeros((len(osc), len(force)))
    # array = []
    val = 0
    for n in osc:
        yval = 0
        for j in force:
            print(yval)
            print(xval)
            val = parse_t(j, n, "1")
            if (val[0][0] < 1):
                array[xval, yval] = 0
            else:
                array[xval, yval] = 1
            yval += 1
        xval +=1
    plt.imshow(array, cmap="hot")
    locs, labels = plt.xticks()
    print(labels)
    plt.title("Succesful Gait Systematic Test (Trotting)")
    plt.ylabel("Oscillator time-steps")
    plt.xlabel("Max force applied")
    plt.yticks(np.arange(0,len(osc), step=1), osc)
    plt.xticks(np.arange(0,len(force), step=1), force)
    ax.set_yticks(np.arange(-.5, len(osc), 1), minor=True);
    ax.set_xticks(np.arange(-.5, len(force), 1), minor=True);
    ax.grid(which = "minor", color = "grey", linestyle="--", linewidth=2)


    plt.subplot(133)
    ax = plt.gca()
    xval = 0
    yval =0
    leng = len(parse_t("*", "*", "2"))
    # array =np.array((leng/2, leng/2))

    array = np.zeros((len(osc), len(force)))
    # array = []
    val = 0
    for n in osc:
        yval = 0
        for j in force:
            print(yval)
            print(xval)
            val = parse_t(j, n, "2")
            if (val[0][0] < 1):
                array[xval, yval] = 0
            else:
                array[xval, yval] = 1
            yval += 1
        xval +=1
    plt.imshow(array, cmap="hot")
    locs, labels = plt.xticks()
    print(labels)
    plt.title("Succesful Gait Systematic Test (Bounding)")
    plt.ylabel("Oscillator time-steps")
    plt.xlabel("Max force applied")
    plt.yticks(np.arange(0,len(osc), step=1), osc)
    plt.xticks(np.arange(0,len(force), step=1), force)
    ax.set_yticks(np.arange(-.5, len(osc), 1), minor=True);
    ax.set_xticks(np.arange(-.5, len(force), 1), minor=True);
    ax.grid(which = "minor", color = "grey", linestyle="--", linewidth=2)

    plt.show()
# plot_experiments()


def plot_experiments2():
    osc = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    force = ["020", "030", "040", "050", "060", "070", "080", "090" ,"100"]
    gait = ["0", "1", "2"]
    # fig = plt.figure()
    # ax = plt.gca()
    # leng = len(parse_t("*", "*", "0"))
    fig, ax = plt.subplots()
    plt.ylabel("Probability Density")
    plt.xlabel("Froude Number")
    plt.title("Distribution of Froude number on Walking, Trotting and Bounding")
    lege = []
    # for n in force:
    val = parse_gaits("*", "*",  "0")
    ind = np.where(val[:,0] > 1)
    # print(val[0])
    values = np.zeros(len(ind[0]))
    count = 0
    for n in ind[0]:
        values[count] = val[n,1]
        count +=1
    froude = (values/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    ax.plot(froude, normf)
    legen =  "Walking- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2,3))
    lege.append(legen)
    # for n in force:
    val = parse_gaits("*", "*",  "1")
    ind = np.where(val[:,0] > 1)
    # print(val[0])
    values = np.zeros(len(ind[0]))
    count = 0
    for n in ind[0]:
        values[count] = val[n,1]
        count +=1
    froude = (values/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    ax.plot(froude, normf)
    legen = "Trotting- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2,3))
    lege.append(legen)
    # for n in force:
    val = parse_gaits("*", "*",  "2")
    ind = np.where(val[:,0] > 1)
    # print(val[0])
    values = np.zeros(len(ind[0]))
    count = 0
    for n in ind[0]:
        values[count] = val[n,1]
        count +=1
    froude = (values/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    ax.plot(froude, normf)
    legen = "Bounding- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2,3))
    lege.append(legen)
    ax.legend(lege, title="Gaits")
    plt.show()
# plot_experiments2()
# plot_experiments()
plot_big2()
# plot_ttest()
plot_froude()
# plot_ttest()
# plot_ex2()
# plot_angles()
# plt.scatter(values[:, 0, 0], values[:, 3, 0])
