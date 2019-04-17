import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
from scipy.integrate import odeint
import scipy.signal as signal
import sys


def deg_to_rad(deg):
    return deg*(np.pi/180)

count = 0
# Initial Configuration (Can Later Be Changed Through User Parameters)
run_array = []
gravity = -9.8
# frequency_multiplier = 175
time_step = 1/500
foot_angle = deg_to_rad(float(sys.argv[5]))
hip_angle = deg_to_rad(float(sys.argv[6]))

max_force = float(sys.argv[1])
oscillator_step = float(sys.argv[2])
hip_height = 1
w = 20
van_multi = 0.1

mu = 1
p_v = 2
num_iterations = 5000
num_epochs = 1
e_b = 999
# Hip Configurations (SET, DO NOT CHANGE)start_x_foot
front_right_hip = 1
front_left_hip = 4
back_right_hip = 7
back_left_hip = 10
front_right_foot = 2
front_left_foot = 5
back_right_foot = 8
back_left_foot = 11
front_right_shoulder = 3
front_left_shoulder = 6
back_right_shoulder = 9
back_left_shoulder = 0

feet = [front_right_foot, back_right_foot, front_left_foot, back_left_foot]
hips = [front_right_hip, back_right_hip, front_left_hip, back_left_hip]
shoulders = [front_right_shoulder, back_right_shoulder, front_left_shoulder, back_left_shoulder]

end_period = 0
p.connect(p.GUI)
position_array = np.zeros((num_epochs, 3, num_iterations))
time_array = np.zeros((num_epochs, num_iterations))
# displacement_array = np.zeros(num_iterations)
force_array = np.zeros((num_epochs,num_iterations))
distance_array = np.zeros((num_epochs,num_iterations))
period_foot = np.zeros((num_epochs, num_iterations))
tilt_array = np.zeros((num_epochs, 3, num_iterations))
height_array = np.zeros((num_epochs,num_iterations))
turn_array = np.zeros((num_epochs, num_iterations))
for e in range(num_epochs):
    print(str(e/num_epochs*100)+ "%")
    # Oscillator Values, Initiated at 1
    start_y_foot = [2,2,2,2]
    start_x_foot = [0,0,0,0]
    new_y_foot =   [2,2,2,2]
    new_x_foot =   [0,0,0,0]

    start_y_hip = [2,2,2,2]
    start_x_hip = [0,0,0,0]
    new_y_hip =   [2,2,2,2]
    new_x_hip =   [0,0,0,0]

    run_simulation = 0
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, gravity)
    p.setTimeStep(time_step)
    p.setDefaultContactERP(0)

    urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

    debug = False;
    # cube = p.loadURDF("cube.urdf", [0.31,0,0.36],[0,5,0, 0], flags = urdfFlags, useFixedBase=True)
    # cube2 = p.loadURDF("cube.urdf", [-0.31,0,0.36],[0,5,0, 0], flags = urdfFlags, useFixedBase=True)
    quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)
    base_dynamics_info = p.getDynamicsInfo(quadruped, -1)
    frh_dynamics_info = p.getDynamicsInfo(quadruped, front_right_hip)
    flh_dynamics_info = p.getDynamicsInfo(quadruped, front_left_hip)

    base_mass = base_dynamics_info[0]
    total_mass = 20 + 4*(0.141+1.527+1.095);

    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

    jointIds=[]
    paramIds=[]
    jointOffsets=[]
    jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
    jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range (4):
        jointOffsets.append(0)
        jointOffsets.append(-0.5)
        jointOffsets.append(0.5)

    for j in range (p.getNumJoints(quadruped)):
            p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(quadruped,j)

    p.getCameraImage(480,320)

    joints=[]
    # maxForceId = p.addUserDebugParameter("max_force",0,100,max_force)
    # max_force = p.readUserDebugParameter(maxForceId)
    # wId = p.addUserDebugParameter("w (Legs)",0,w,w)
    # muId = p.addUserDebugParameter("mu (Legs)", 0, mu, mu)
    # pvId = p.addUserDebugParameter("p_v (Legs)", 0, p_v, p_v)

    jointOffsets[1] -= -0.7
    jointOffsets[4] -= -0.7
    jointOffsets[7] -= -0.5
    jointOffsets[10] -= -0.5

    run_simulation = 1
    mode = p.VELOCITY_CONTROL

    # Begins timer to allow for sin function to work (Will replace with vanderpol in future)
    for i, v in enumerate(hips):
        p.setJointMotorControl2(quadruped, v,mode, -jointOffsets[i], force=max_force)
        p.enableJointForceTorqueSensor(quadruped, v)

    for i, v in enumerate(feet):
        p.setJointMotorControl2(quadruped, v, mode, -jointOffsets[i], force=max_force)
        p.enableJointForceTorqueSensor(quadruped, v)


    # p.useFixedBase = True
    # time.sleep(5);




    # foot_debug = deg_to_rad(180)
    # hip_debug = deg_to_rad(60)
    timer = 0


    current_i = 0
    current_i2 = 0
    found = 0;
    start_period = 0;

    def van_der_pol_coupled_foot(x, t):
        # global chosen_x_foot
        x0 = x[1]
        x_ai =x[0]
        for j in range(4):
            x_ai += x[0]-(lamb[current_i][j]*chosen_x_foot[j])
        x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
        # + (0.5*feedback[current_i])
        res = np.array([x0, x1])
        return res

    def van_der_pol_coupled_hip(x, t):
        # global chosen_x_hip
        x0 = x[1]
        x_ai =x[0]
        for j in range(4):
            x_ai += x[0]-(lamb[current_i2][j]*chosen_x_hip[j])
        x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
        # + (0.5*feedback2[current_i2])
        res = np.array([x0, x1])
        return res
    l = 0.1

    lamb_walk = [ [0, -l, l, -l],
                  [-l, 0, -l, l],
                  [-l, l, 0, -l],
                  [l, -l, -l, 0]]
    lamb_trot = [ [0, -l, -l, l],
                  [-l, 0, l, -l],
                  [-l, l, 0, -l],
                  [l, -l, -l, 0]]
    lamb_bound = [ [0, l, -l, -l],
                   [l, 0, -l, -l],
                   [-l, -l, 0, l],
                   [-l, -l, l, 0]]
    gaits= [lamb_walk, lamb_trot, lamb_bound]
    lamb = gaits[int(sys.argv[3])]

    qKey = ord('q')
    pKey = ord('p')
    rKey = ord('r')
    run_string= max_force  + foot_angle + hip_angle


    oscillator_values = [[],[],[],[]]
    oscillator_values2 = [[],[],[],[]]
    # force_array= []
    limb_values = [[],[],[],[], [], [], [], []]
    starting_foot_value = 0
    # velocity_array = []
    total_displacement = 0
    total_distance = 0
    total_force = 0
    force_expended = 0
    # cost_of_transport= []
    sample_timer = 0
    prev_value = 0
    counter = 0
    time0 = 0
    time1 = 0
    # TODO, CHANGE SAMPLING RATE TO TIME FOR A FULL OSCILLATION
    # sampling_rate = time_step



    for n in range(num_iterations):
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
            break;
        if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
            run_simulation = not run_simulation
            p.setRealTimeSimulation(run_simulation)
        if (run_simulation):
            time_array[e,n] = timer
            x_list_foot = []
            x_list_hip = []

            count += oscillator_step
            timer += time_step
            # max_force = p.readUserDebugParameter(maxForceId)
            # w = p.readUserDebugParameter(wId)
            # mu = p.readUserDebugParameter(muId)
            # p_v = int(p.readUserDebugParameter(pvId))
            pos_ori = p.getBasePositionAndOrientation(quadruped)
            local_inertia = p.getDynamicsInfo(quadruped, -1)[2]
            height_array[e,n] = pos_ori[0][2]
            turn_array[e,n] = pos_ori[0][0]
            position_array[e,0,n] =(pos_ori[0][0])
            position_array[e,1,n] =(pos_ori[0][1])
            tilt_array[e,0,n] = pos_ori[1][0]
            tilt_array[e,1,n] = pos_ori[1][1]
            tilt_array[e,2,n]= pos_ori[1][2]
            distance = (abs((pos_ori[0][1])**2) + abs((pos_ori[0][0])**2))**(1/2)
            # try:
            #     displacement = (distance-displacement_array[-1])
            # except Exception as e:
            #     displacement = 0;
            force_expended = 0
            for j in range(12):
                for k in range(6):
                    force_expended += ((abs(p.getJointState(quadruped, j)[2][k])/100))**2
            force_expended = (force_expended)**1/2
            distance_array[e,n] = distance
            # displacement_array[n] = displacement
            # velocity_array.append(velocity)
            # froude_number = (velocity**2)/-gravity*hip_height
            # froude_number_array.append(froude_number)
            force_array[e,n] = force_expended
            # power_avg = (total_force*displacement)

            # try:
            #     cost_transport = power_avg/(total_mass*abs(gravity)*velocity)
            # except Exception as e:
            #     cost_transport = power_avg/(total_mass*abs(gravity))
            # cost_of_transport.append(cost_transport)
            # total_force = 0
            # sample_timer = 0

            # sample_timer += time_step
            # p.addUserDebugLine((pos_ori[0][0], pos_ori[0][1], pos_ori[0][2]), (pos_ori[0][0]+0.1, pos_ori[0][1], pos_ori[0][2]))


            # feedback = [fr_foot_rot, br_foot_rot, fl_foot_rot, bl_foot_rot]
            # feedback2 = [fr_hip_rot, br_hip_rot, fl_hip_rot, bl_hip_rot]
            for i in range(4):
                current_i = i
                chosen_x_foot = start_x_foot
                osc_foot= odeint(van_der_pol_coupled_foot, [start_y_foot[i], start_x_foot[i]], [count-oscillator_step, count])
                x = osc_foot[1][1]
                y = osc_foot[1][0]
                x_list_foot.append(x)
                new_y_foot[i] = y
                new_x_foot[i] = x

            for i in range(4):
                current_i2 = i
                chosen_x_hip = start_x_hip
                osc_hip= odeint(van_der_pol_coupled_hip, [start_y_hip[i], start_x_hip[i]], [count-oscillator_step, count])
                x2 = osc_hip[1][1]
                y2 = osc_hip[1][0]
                x_list_hip.append(x2)

                new_y_hip[i] = y2
                new_x_hip[i] = x2

            if (len(oscillator_values[0]) >= 3):
                nl = counter-1
                current = oscillator_values[0][nl] - oscillator_values[0][nl-1]
                previous = oscillator_values[0][nl-1] - oscillator_values[0][nl-2]
                if(current >= 0 and previous <= 0):
                    found += 1
                    if (found == 1):
                        start_period = timer
                    if (found == 2):
                        end_period = timer
                        found = 0
                        period_foot[e,n] = end_period-start_period






            start_y_foot = new_y_foot
            start_x_foot = new_x_foot
            start_y_hip = new_y_hip
            start_x_hip = new_x_hip


            for n in range(4):
                oscillator_values[n].append(x_list_foot[n])
                oscillator_values2[n].append(x_list_hip[n])
            for i, v in enumerate(feet):
                realval = p.getJointState(quadruped, v)[0]
                limb_values[i].append(realval)
                p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL,-jointOffsets[v]+(foot_angle*x_list_foot[i]*van_multi), force=max_force)
            for i, v in enumerate(hips):
                realval = p.getJointState(quadruped, v)[0]
                limb_values[i+4].append(realval)
                p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL, -jointOffsets[v]+(hip_angle*x_list_hip[i]*van_multi), force=max_force)
            for i, v in enumerate(shoulders):
                p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL,  0, force=max_force)
            p.stepSimulation()
    p.resetSimulation()


# EXPERIMENT DESIGN
plot = "osc_hip"
final_time = np.zeros(num_epochs)
distance_val = np.zeros(num_epochs)
velocity = np.zeros(num_epochs)
froude_number = np.zeros(num_epochs)
force_values = np.zeros(num_epochs)
cost_transport = np.zeros(num_epochs)
period_average = np.zeros(num_epochs)
for n in range(num_epochs):
    final_time[n] = time_array[n, -1] - time_array[n, e_b]
    distance_val[n] = distance_array[n, -1] - distance_array[n, e_b]
    velocity[n] = distance_val[n]/final_time[n]
    froude_number[n] = (velocity[n]**2)/-gravity*hip_height
    force_values[n] = np.sum(force_array[n,e_b:])
    power_avg = (force_values[n]*distance_val[n])
    cost_transport[n] = power_avg/(total_mass*abs(gravity)*velocity[n])
    period_average[n] = np.mean(period_foot[n,:])


# mean_velocity = np.mean(velocity)
# std_velocity = np.std(velocity)
# mean_froude = np.mean(froude_number)
# std_froude = np.std(froude_number)
# mean_force = np.mean(force_values)
# std_force = np.std(force_values)
# mean_time = np.mean(final_time)
# std_time = np.std(final_time)
# mean_distance =  np.mean(distance_val)
# std_distance = np.std(distance_val)
# mean_cost = np.mean(cost_transport)
# std_cost = np.std(cost_transport)
# mean_period = np.mean(period_average)
# std_period = np.std(period_average)
# run_name = sys.argv[4]+"/f"+sys.argv[1]+"o"+sys.argv[2]+"g"+sys.argv[3]+"l"+sys.argv[4]+"h"+sys.argv[5]
# run_log = open(run_name+"log.txt", "w+")
# saved_calc = [[mean_velocity, std_velocity], [mean_froude, std_froude], [mean_distance, std_distance], [mean_cost, std_cost], [mean_period, std_period]]
# string = "Oscillator Step: " + str(oscillator_step) + "\n"
# string += "Max Force: " + str(max_force) + "\n"
# string += "Gait: " + sys.argv[3] + "\n"
# string += "Leg Rotation: " + str(foot_angle) + "\n"
# string += "Hip Rotation: " + str(hip_angle) + "\n"
# string += ""
# run_log.write(string)
# for item in saved_calc:
#     string = str(item[0])+":"+str(item[1])+"\n"
#     run_log.write(string)
# run_log.close()

# plot = "physics2"
# if plot == "map":
#
#     plt.figure(figsize=(20,20))
#     plt.title("Path Of Robot Over Time")
#     plt.xlabel("X Direction")
#     plt.ylabel("Y Direction")
#     plt.plot(y_values, x_values)
#     plt.show()

if plot == "stride":
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
if plot == "osc_hip":
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
    foot_labels = ["Front Right Foot", "Back Right Foot", "Front Left Foot", "Back Left Foot"]
    hip_labels = ["Front Right Hip", "Back Right Hip", "Front Left Hip", "Back Left Hip"]
    plt.title("Foot Imaginary Coupled Oscillator")
    plt1, =plt.plot(time_array[0], oscillator_values[0])
    plt2, =plt.plot(time_array[0], oscillator_values[1])
    plt3, =plt.plot(time_array[0], oscillator_values[2])
    plt4, =plt.plot(time_array[0], oscillator_values[3])
    plt.legend([plt1, plt2, plt3, plt4], foot_labels)

    fig, ax = plt.subplots()
    im = ax.imshow(lamb, cmap='tab10')
    plt.title("Coupled Oscillator Coefficients")
    ax.set_xticks(np.arange(len(foot_labels)))
    ax.set_yticks(np.arange(len(foot_labels)))
    ax.set_xticklabels(foot_labels)
    ax.set_yticklabels(foot_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(foot_labels)):
        for j in range(len(foot_labels)):
            text = ax.text(j, i, lamb[i][j],
                           ha="center", va="center", color="w")

    plt.show()
if plot == "oscillators":
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
    foot_labels = ["Front Right Foot", "Back Right Foot", "Front Left Foot", "Back Left Foot"]
    hip_labels = ["Front Right Hip", "Back Right Hip", "Front Left Hip", "Back Left Hip"]
    plt.subplot(3,2,1)
    plt.title("Foot Imaginary Coupled Oscillator")
    # plt.title("Hip Imaginary Coupled Oscillator")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Oscillator Output")
    plt1, =plt.plot(time_array, oscillator_values[0])
    plt2, =plt.plot(time_array, oscillator_values[1])
    plt3, =plt.plot(time_array, oscillator_values[2])
    plt4, =plt.plot(time_array, oscillator_values[3])
    plt.legend([plt1, plt2, plt3, plt4], foot_labels)

    plt.subplot(3,2,2)
    plt.title("Foot Actual Limb Rotation Values")

    plt1,= plt.plot(time_array, limb_values[0])
    plt2,= plt.plot(time_array, limb_values[1])
    plt3, =plt.plot(time_array, limb_values[2])
    plt4, =plt.plot(time_array, limb_values[3])
    plt.legend([plt1, plt2, plt3, plt4], ["Front Right Foot", "Back Right Foot", "Front Left Foot", "Back Left Foot"])


    plt.subplot(3,2,3)
    plt.title("Hip Imaginary Coupled Oscillator")
    plt.xlabel("Time Step (t)")
    plt.ylabel("X Value")
    plt1,= plt.plot(time_array, oscillator_values2[0])
    plt2,= plt.plot(time_array, oscillator_values2[1])
    plt3,= plt.plot(time_array, oscillator_values2[2])
    plt4,=plt.plot(time_array, oscillator_values2[3])
    plt.legend([plt1, plt2, plt3, plt4], hip_labels)


    plt.subplot(3,2,4)
    plt.title("Hip Actual Limb Rotation Values")
    plt1,=plt.plot(time_array, limb_values[4])
    plt2,=plt.plot(time_array, limb_values[5])
    plt3,=plt.plot(time_array, limb_values[6])
    plt4,=plt.plot(time_array, limb_values[7])
    plt.legend([plt1, plt2, plt3, plt4], ["Front Right Hip", "Back Right Hip", "Front Left Hip", "Back Left Hip"])

    fig, ax = plt.subplots()
    im = ax.imshow(lamb, cmap='tab20b')
    plt.title("Coupled Oscillator Coefficients: Walking")
    ax.set_xticks(np.arange(len(foot_labels)))
    ax.set_yticks(np.arange(len(foot_labels)))
    ax.set_xticklabels(foot_labels)
    ax.set_yticklabels(foot_labels)
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, lamb[i][j],
                           ha="center", va="center", size=20, color="w")

    # cbar = ax.figure.colorbar(lamb, ax=ax)
    # cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    # plt.yticks(np.arange(0, , 1))
    # plt.xticks(np.arange(0, , 1))
    plt.show()


if plot == "physics2":
    plt.figure(figsize=(20,20))
    # plt.subplot(1,1,1)
    # plt.subplot(5,1,1)
    # plt.ylim([0,20])
    # plt.title("Cost Of Transport")
    # plt.xlabel("Time Step (t) (Measurement taken every second)")
    # plt.ylabel("Cost Of Transport ")
    # plt.plot(time_array, cost_of_transport)
    plt.subplot(3,1 ,1)

    plt.title("Forces")
    plt.ylim([0,np.max(force_array[e_b:])])
    plt.plot(time_array[e_b:], force_array[e_b:])
    # plt.subplot(5,1,3)
    # plt.title("velocity")
    # plt.ylim([0,20])
    # plt.plot(time_array, velocity_array)


    plt.subplot(3,1,3)
    plt.title("distance")
    plt.ylim([0,np.max(distance_array)])
    plt.plot(time_array[e_b:], distance_array[e_b:])
    #
    # plt.subplot(4,1,4)
    # plt.title("Displacement Array")
    # plt.scatter(displacement_array, force_array)
    plt.show()
if plot == "physics":
    plt.figure(figsize=(15,15))
    plt.subplot(3, 3, 1)

    plt.subplot(3, 3, 3)
    plt.title("Turn in X Over Time")
    plt.plot(time_array, turn_array)

    plt.ylabel("X Value")
    plt.xlabel("Time Step (t)")
    plt.subplot(3, 3, 4)
    plt.title("X Rotation Over Time")
    plt.plot(time_array, x_tilt_array)
    plt.ylabel("X Rotation")
    plt.xlabel("Time Step (t)")
    plt.subplot(3, 3, 5)
    plt.title("Y Rotation Over Time")
    plt.plot(time_array, y_tilt_array)
    plt.ylabel("Y Value")
    plt.xlabel("Time Step (t)")
    plt.subplot(3, 3, 6)

    plt.ylabel("Z Value")
    plt.xlabel("Time Step (t)")
    plt.savefig(run_name, dpi=250)
    plt.show()
