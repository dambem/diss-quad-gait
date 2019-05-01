import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.stats as stats
from scipy.stats import norm
import sklearn.metrics as sk
import pandas as pd

height = 0.4
values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "15", "16", "17","18", "19", "20"]
markers = ["D", "x", "o", "s", "*", "h", "v", "1", "2", "3"]

def dynamic_similarity(a, g, b, froude):
    return (a*(froude)**b)

# Parses the ttest structure, returns np.array
# Parameters can either be a specific value or * in order to get all values of that type
def parse_t(force, osc, l):
    filenames = sorted(glob.glob('ttest/f'+force+'o'+osc+"g0lttesth"+l+"*"))
    values = np.zeros((len(filenames), 11, 3))
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

#  Parses structural tests, returns numpy array of (distance, froude)
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

def parse_all(force, osc, l, h):
    # 2 = velocity
    # 3 = froude
    # 4 = distance
    # 5 = cost
    # 6 = time period
    filenames = sorted(glob.glob('all/f'+force+'o'+osc+'g0l'+l+"h"+h+"log.txt"))
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

def oscillator_vs_froude_bar():
    fig, ax = plt.subplots()
    for n in values:
        val = parse_all("*", n, "*", "*")
        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, None)
        distance = val[:,4,0]
        data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
        plt.title("Average Froude Number against Oscillator Time Step")
        ax.set_xlabel("Oscillator Time Steps")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/height, label=n, yerr=np.mean(val[:,3,1]), color='blue')
    plt.show()

def plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Oscillator Time Step")
    ax.set_xticks([0, 0.1,0.2,height,height, 0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5, 1.6,1.7,1.8,1.9,2.0, 2.1, 2.2])
    ax.set_ylabel("Cost Of Locomotion Per Unit Distance")
    for g in range(len(values)):
        for n in forces:
            val = parse_all(n, values[g], "*", "*")
            cost_per_distance = val[:,5,0]/val[:,4,0]
            cost_per_distance = np.clip(cost_per_distance, 0, 1500)
            distance = val[:,4,0]
            arrayvalues = np.where(cost_per_distance > 0)
            data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
            froude = val[:,3,0]/(height)
            froude_ind = np.where(np.logical_and(froude<=height, froude>=0.1))
            average_cost = np.mean(cost_per_distance)
            ax.scatter(froude,  cost_per_distance, float(values[g]), label=n)
    ax.legend(forces, title="Max Force")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Froude Number")
    ax.set_zlabel("Max Force")
    ax.set_ylabel("Hip Rotation (Degrees)")
    for l in leg:
        for g in range(len(values)):
            for n in forces:
                val = parse_all(n, values[g], l, "10")
                cost_per_distance = val[:,5,0]/val[:,4,0]
                cost_per_distance = np.clip(cost_per_distance, 0, 1500)
                distance = val[:,4,0]
                data = np.array((val[:,3,0]/(height) , distance, val[:,0,0], val[:,1,0], cost_per_distance))
                froude = val[:,3,0]/(height)
                froude_ind = np.where(np.logical_and(froude<=height, froude>=0.1))
                average_cost = np.mean(cost_per_distance)
                ax.scatter(float(l), float(n), froude)
    plt.show()

def print_percentages():
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    for n in values:
        val = parse_all("*", n, "*", "*")
        froude = val[:,3,0]/(height)
        froude_ind = np.where(np.logical_and(froude<=height, froude >= 0.001))
        perc = len(froude_ind[0])/len(val[:,5,0])
        print(str(n))
        print(str(perc*100) + "%")
    val = parse_all("*","*", "*", "*")
    print("Overall %")
    froude = val[:,3,0]/(height)
    froude_ind = np.where(np.logical_and(froude<=height, froude >= 0.001))
    perc = len(froude_ind[0])/len(val[:,5,0])
    print(str(perc*100) + "%")

def plot_distribution():
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
    leg=["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
    fig, ax = plt.subplots()
    for n in leg:
        # for j in values:
        val = parse_all("*", "*", n, "*")

        phase_difference = ((1/500)/float(n)*2)
        period = val[:,6,0]
        period2 = period/phase_difference
        cost_per_distance = val[:,5,0]/val[:,4,0]
        cost_per_distance = np.clip(cost_per_distance, 0, 1000)
        cost_per_distance = np.where(cost_per_distance < 1000, cost_per_distance, 0)

        plt.title("Average Froude Number against Leg Rotation")
        ax.set_xlabel("Leg Rotation")
        ax.set_ylabel("Average Froude Number")
        ax.bar(n, np.mean(val[:,3,0])/height, label=n, yerr=np.mean(val[:,3,1]), color='red')

    val = parse_all("*", "*", "*", "*")
    fig, ax = plt.subplots()
    cost = (val[:,6,0]/val[:,4,0])
    cost = np.where(cost > 0, cost, 0)
    cost.sort()
    costmean = np.mean(cost)
    froudestd = np.std(cost)
    norm = stats.norm.pdf(cost, costmean, froudestd)
    plt.title("Distribution Of Cost Of Locomotion Values")
    ax.set_xlabel("Cost Of Locomotion")
    ax.set_ylabel("Probability")
    ax.plot(cost, norm)

    fig, ax = plt.subplots()
    froude = (val[:,3,0]/height)
    froude.sort()
    froudemean = np.mean(froude)
    froudestd = np.std(froude)
    normf = stats.norm.pdf(froude, froudemean, froudestd)
    plt.title("Distribution Of Froude Number Values")
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Probability Density")
    forces = ["020", "030", "040", "050", "060", "070", "080", "090", "100"]
    lege = ["All", "020", "030", "040", "050", "060", "070", "080", "090", "100"]
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    lege = []
    legen = "All- " + " μ: " + str(round(froudemean, 3)) + ", σ²: " + str(round(froudestd**2, 3))
    lege.append(legen)
    ax.plot(froude, normf) #
    ax.legend(lege, title="Oscillator Values")

    fig, ax = plt.subplots()
    for n in values:
        val = parse_all("*", n, "*", "*")
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
    val = parse_all("*", "*", "*", "*")

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
    ax.plot(froude, normf) #
    for n in forces:
        val = parse_all(n, "*", "*", "*")
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

def plot_froude():
    fig, ax = plt.subplots()
    # 0.010/0.002
    values = ["0.010", "0.008", "0.006", "0.004", "0.002"]
    plt.title("Relative Stride Length against Froude Number")
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Relative Stride Length")
    lege = []

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
    plt.plot(med_x, med_y, linestyle='--')
    lege.append("Froude Cursorial Walking Lower Bound")

    a = 2.4
    g = 9.8
    # h = 2
    b = 0.24
    h = np.linspace(0, height, 500)
    med_x = []
    med_y = []
    for n in h:
        med_x.append(n)
        med_y.append(dynamic_similarity(a,g,b,n))
    plt.plot(med_x, med_y, linestyle='--')
    lege.append("Froude Cursorial Walking Upper Bound")

    a = 2.4
    g = 9.8
    # h = 2
    b = 0.34
    h = np.linspace(0, height, 500)
    med_x = []
    med_y = []
    for n in h:
        med_x.append(n)
        med_y.append(dynamic_similarity(a,g,b,n))
    plt.plot(med_x, med_y, linestyle='-')
    lege.append("Froude Cursorial Walking ")


    val = parse_all("*", "*", "*", "*")
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
    values = [["0.002", 0.56], ["0.004", 0.28], ["0.006", 0.19], ["0.008", 0.14], ["0.010", 0.11]]
    for n in values:
        val = parse_all("*", n[0], "*", "10")
        osc_t = val[:,1,0]
        froude = (np.round(val[:,3,0], 3)/height)
        time_period = n[1]
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
    plt.show()


print_percentages()
oscillator_vs_froude_bar()
plot_3d()
plot_distribution()
plot_froude()
