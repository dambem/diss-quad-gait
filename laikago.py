import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
from scipy.integrate import odeint
import scipy.signal as signal


# Oscillator Values, Initiated at 1
start_y_foot = [2,2,2,2]
start_x_foot = [0,0,0,0]
new_y_foot = [2,2,2,2]
new_x_foot = [0,0,0,0]

start_y_hip = [2,2,2,2]
start_x_hip = [0,0,0,0]
new_y_hip = [2,2,2,2]
new_x_hip = [0,0,0,0]

count = 0
# Initial Configuration (Can Later Be Changed Through User Parameters)
run_array = []
gravity = -9.8
time_step = 1./500
# frequency_multiplier = 175
foot_angle = 12
hip_angle = 6
max_force = 40
oscillator_step = 0.015

w = 20
van_multi = 0.1
van_multi2 = 0.1

mu = 1
p_v = 2

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



run_simulation = 0
p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, gravity)
p.setTimeStep(time_step)
p.setDefaultContactERP(0)

urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

debug = False;
# cube = p.loadURDF("cube.urdf", [0.5,0,0.35],[0,1,0, 0], flags = urdfFlags, useFixedBase=True)
# cube2 = p.loadURDF("cube.urdf", [-0.5,0,0.35],[0,1,0, 0], flags = urdfFlags, useFixedBase=True)

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
            print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]
jointOffsets=[]
jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]


def getMotorAngles(self):
    motorAngles = []
    for i in range(self.nMotors):
          jointState = p.getJointState(self.quadruped, self.motorIdList[i])
          motorAngles.append(jointState[0])
    motorAngles = np.multiply(motorAngles, self.motorDir)
    return motorAngles

for i in range (4):
    jointOffsets.append(0)
    jointOffsets.append(-0.5)
    jointOffsets.append(0.5)

for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                jointIds.append(j)


p.getCameraImage(480,320)
p.setRealTimeSimulation(run_simulation)


joints=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,max_force)
maxForce = p.readUserDebugParameter(maxForceId)
wId = p.addUserDebugParameter("w (Legs)",0,w,w)
muId = p.addUserDebugParameter("mu (Legs)", 0, mu, mu)
pvId = p.addUserDebugParameter("p_v (Legs)", 0, p_v, p_v)

jointOffsets[1] -= -0.7
jointOffsets[4] -= -0.7
jointOffsets[7] -= -0.5
jointOffsets[10] -= -0.5
# sin =  np.sin(0)
# sin2 = np.sin(np.pi*0.5)
# sin3 = np.sin(np.pi)
# sin4 = np.sin(np.pi*3/4)
run_simulation = 1
p.setRealTimeSimulation(run_simulation)
# Begins timer to allow for sin function to work (Will replace with vanderpol in future)
p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1], force=maxForce)
p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL, -jointOffsets[4], force=maxForce)
p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL, -jointOffsets[7], force=maxForce)
p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,  -jointOffsets[10], force=maxForce)
p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, -jointOffsets[2], force=maxForce)
p.setJointMotorControl2(quadruped, front_left_foot,p.POSITION_CONTROL, -jointOffsets[5], force=maxForce)
p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, -jointOffsets[8], force=maxForce)
p.setJointMotorControl2(quadruped, back_left_foot,p.POSITION_CONTROL, -jointOffsets[11], force=maxForce)
p.enableJointForceTorqueSensor(quadruped, front_right_hip)
p.enableJointForceTorqueSensor(quadruped, front_left_hip)
p.enableJointForceTorqueSensor(quadruped, back_right_hip)
p.enableJointForceTorqueSensor(quadruped, back_left_hip)
p.enableJointForceTorqueSensor(quadruped, front_right_foot)
p.enableJointForceTorqueSensor(quadruped, front_left_foot)
p.enableJointForceTorqueSensor(quadruped, back_right_foot)
p.enableJointForceTorqueSensor(quadruped, back_left_foot)
foot_angleId = p.addUserDebugParameter("Foot Angle of rotation", 0, 20, foot_angle)
hip_angleId = p.addUserDebugParameter("Hip Angle of rotation", 0, 20, hip_angle)
p.useFixedBase = True
# time.sleep(5);

def deg_to_rad(deg):
    return deg*(np.pi/180)


foot_debug = deg_to_rad(180)
hip_debug = deg_to_rad(60)
timer = 0
distance_array = []
height_array = []
turn_array = []
time_array = []
x_tilt_array = []
y_tilt_array = []
z_tilt_array = []
lamb_trot = -0.1
lamb_pace = -0.1
lamb_bound = -0.1
lamb_walk = 0.1

lambId = p.addUserDebugParameter("Lamb pace",lamb_pace, -lamb_pace, lamb_pace)
lambWalkId = p.addUserDebugParameter("Lamb walk", -lamb_walk, lamb_walk, lamb_walk)
lambBoundId = p.addUserDebugParameter("Lamb Bound",lamb_bound, -lamb_bound, lamb_bound)
lambTrotId = p.addUserDebugParameter("Lamb Trot", lamb_trot, -lamb_trot, lamb_trot)
current_i = 0
current_i2 = 0

def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p_v - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res

def van_der_pol_oscillator_deriv_pure(x, t):
    x0 = x[1]
    x1 = 1 * ((1 - (x[0] ** 2.0)) * x0) - x[0]*1
    res = np.array([x0, x1])
    return res

def van_der_pol_coupled_foot(x, t):
    # global chosen_x_foot
    x0 = x[1]
    x_ai =x[0]
    for j in range(4):
        x_ai += x[0]-(lamb[current_i][j]*chosen_x_foot[j])
    # x_ai *= time_step
    # osc_foot = odeint(van_der_pol_oscillator_deriv, [x[0], x[1]], [t-time_step, t])
    x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
    # + (0.5*feedback[current_i])
    res = np.array([x0, x1])
    return res

def van_der_pol_coupled_hip(x, t):
    # global chosen_x_hip
    x0 = x[1]
    x_ai =x[0]
    # print(current_i2)
    for j in range(4):
        x_ai += x[0]-(lamb[current_i2][j]*chosen_x_hip[j])
    # x_ai *= time_step
    x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
    # + (0.5*feedback2[current_i2])
    res = np.array([x0, x1])
    return res

qKey = ord('q')
pKey = ord('p')
rKey = ord('r')
run_string= maxForce  + foot_angle + hip_angle


oscillator_values = [[],[],[],[]]
oscillator_values2 = [[],[],[],[]]
force_values= []
limb_values = [[],[],[],[], [], [], [], []]
time_stepId = p.addUserDebugParameter("Oscillator Time Step", 0.001, 0.05, oscillator_step)
time.sleep(1)
velocity_array = []
sampling_array =[]
total_displacement = 0
total_distance = 0
total_force = 0
force_expended = 0
cost_of_transport= []
displacement_array = []
sample_timer = 0
prev_value = 0
time0 = 0
time1 = 0
# TODO, CHANGE SAMPLING RATE TO TIME FOR A FULL OSCILLATION
sampling_rate = time_step*100
while (timer <= 10):
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        break;
    if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
        run_simulation = not run_simulation
        p.setRealTimeSimulation(run_simulation)
    if (run_simulation):
        x_list_foot = []
        x_list_hip = []
        y_list_foot = []
        y_list_hip = []
        oscillator_step = p.readUserDebugParameter(time_stepId)
        count += oscillator_step
        timer += time_step
        maxForce = p.readUserDebugParameter(maxForceId)
        foot_angle = deg_to_rad(p.readUserDebugParameter(foot_angleId))
        hip_angle = deg_to_rad(p.readUserDebugParameter(hip_angleId))
        w = p.readUserDebugParameter(wId)
        mu = p.readUserDebugParameter(muId)
        p_v = int(p.readUserDebugParameter(pvId))

        pos_ori = p.getBasePositionAndOrientation(quadruped)*2


        p.addUserDebugLine((pos_ori[0][0], pos_ori[0][1], pos_ori[0][2]), (pos_ori[0][0]+0.1, pos_ori[0][1], pos_ori[0][2]))

        local_inertia = p.getDynamicsInfo(quadruped, -1)[2]



        height_array.append(pos_ori[0][2])
        turn_array.append(pos_ori[0][0])
        x_tilt_array.append(pos_ori[1][0])
        y_tilt_array.append(pos_ori[1][1])
        z_tilt_array.append(pos_ori[1][2])
        if (sample_timer >= sampling_rate):
            distance = (abs((pos_ori[0][1])**2) + abs((pos_ori[0][0])**2))**(1/2)
            try:
                displacement = (distance-distance_array[-1])
            except Exception as e:
                displacement = distance;
            distance_array.append(distance)
            displacement_array.append(displacement)

            velocity = displacement/sampling_rate

            velocity_array.append(velocity)
            sampling_array.append(timer)
            measurements_taken = (sample_timer/time_step)

            force_avg = total_force
            force_values.append(force_avg)
            power_avg = (total_force*displacement)/sampling_rate
            try:
                cost_transport = power_avg/(total_mass*abs(gravity)*velocity)
            except Exception as e:
                cost_transport = power_avg/(total_mass*abs(gravity))
            cost_of_transport.append(cost_transport)
            total_force = 0
            sample_timer = 0

        sample_timer += time_step
        force_expended =0
        for n in range(12):
            force_expended += (abs(p.getJointState(quadruped, n)[2][0]/100)*maxForce)**2
        force_expended = force_expended**(1/2)
        total_force += force_expended



        fr_foot_forces = abs(p.getJointState(quadruped, front_right_foot)[2][0]/100*maxForce)
        br_foot_forces = abs(p.getJointState(quadruped, back_right_foot)[2][0]/100*maxForce)
        fl_foot_forces = abs(p.getJointState(quadruped, front_left_foot)[2][0]/100*maxForce)
        bl_foot_forces = abs(p.getJointState(quadruped, back_left_foot)[2][0]/100*maxForce)
        fr_hip_forces = abs(p.getJointState(quadruped, front_right_hip)[2][0]/100*maxForce)
        br_hip_forces = abs(p.getJointState(quadruped, back_right_hip)[2][0]/100*maxForce)
        fl_hip_forces = abs(p.getJointState(quadruped, front_left_hip)[2][0]/100*maxForce)
        bl_hip_forces = abs(p.getJointState(quadruped, back_left_hip)[2][0]/100*maxForce)

        # feedback = [fr_foot_rot, br_foot_rot, fl_foot_rot, bl_foot_rot]
        # feedback2 = [fr_hip_rot, br_hip_rot, fl_hip_rot, bl_hip_rot]

        lamb_walk2 = [ [0, -0.1, -0.1, -0.1],
                      [-0.1, 0, -0.1, -0.1],
                      [-0.1, -0.1, 0, -0.1],
                      [-0.1, -0.1, -0.1, 0]]
        lamb_walk = [ [0, -0.1, 0.1, -0.1],
                      [-0.1, 0, -0.1, 0.1],
                      [-0.1, 0.1, 0, -0.1],
                      [0.1, -0.1, -0.1, 0]]
        lamb_trot = [ [0, -0.1, -0.1, 0.1],
                      [-0.1, 0, 0.1, -0.1],
                      [-0.1, 0.1, 0, -0.1],
                      [0.1, -0.1, -0.1, 0]]
        lamb_bound = [ [0, 0.1, -0.1, -0.1],
                       [0.1, 0, -0.1, -0.1],
                       [-0.1, -0.1, 0, 0.1],
                       [-0.1, -0.1, 0.1, 0]]
        lamb_pace = [ [0, -0.1, 0.1, -0.1],
                      [-0.1, 0, -0.1, 0.1],
                      [0.1, -0.1, 0, -0.1],
                      [-0.1, 0.1, -0.1, 0]]
        lamb = lamb_walk


        for i in range(4):
            current_i = i
            chosen_x_foot = start_x_foot
            osc_foot= odeint(van_der_pol_coupled_foot, [start_y_foot[i], start_x_foot[i]], [count-oscillator_step, count])
            x = osc_foot[1][1]
            y = osc_foot[1][0]
            x_list_foot.append(x)
            y_list_foot.append(y)
            new_y_foot[i] = y
            new_x_foot[i] = x

        for i in range(4):
            current_i2 = i
            chosen_x_hip = start_x_hip
            osc_hip= odeint(van_der_pol_coupled_hip, [start_y_hip[i], start_x_hip[i]], [count-oscillator_step, count])
            x2 = osc_hip[1][1]
            y2 = osc_hip[1][0]
            x_list_hip.append(x2)
            y_list_hip.append(y2)

            new_y_hip[i] = y2
            new_x_hip[i] = x2

        start_y_foot = new_y_foot
        start_x_foot = new_x_foot
        start_y_hip = new_y_hip
        start_x_hip = new_x_hip


        time_array.append(timer)
        for n in range(4):
            oscillator_values[n].append(x_list_foot[n])
            oscillator_values2[n].append(x_list_hip[n])






        feet = [front_right_foot, back_right_foot, front_left_foot, back_left_foot]
        hips = [front_right_hip, back_right_hip, front_left_hip, back_left_hip]
        shoulders = [front_right_shoulder, back_right_shoulder, front_left_shoulder, back_left_shoulder]

        for i, v in enumerate(feet):
            realval = p.getJointState(quadruped, v)[0]
            limb_values[i].append(realval)
            p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL,-jointOffsets[v]+(foot_angle*x_list_foot[i]*van_multi), force=maxForce)

        for i, v in enumerate(hips):
            realval = p.getJointState(quadruped, v)[0]
            limb_values[i+4].append(realval)
            p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL, -jointOffsets[v]+(hip_angle*x_list_hip[i]*van_multi), force=maxForce)

        for i, v in enumerate(shoulders):
            p.setJointMotorControl2(quadruped, v,p.POSITION_CONTROL,  0, force=maxForce)

run_name = 'laikago-sin-'+datetime.datetime.today().strftime('%Y-%s')+"-"
run_log = open(run_name+"log.txt", "w+")


for items in run_array:
    string = ""
    string = str(items[0])+": [ "
    for specific in items[1]:
        string += "[V: " + str(specific[0]) + " T: " + str(specific[1]) + "]"
    string += " ]\n"
    run_log.write(string)
run_log.close()
plot = "oscillators"
if plot == "osc_hip":
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
    foot_labels = ["Front Right Foot", "Back Right Foot", "Front Left Foot", "Back Left Foot"]
    hip_labels = ["Front Right Hip", "Back Right Hip", "Front Left Hip", "Back Left Hip"]
    plt.title("Foot Imaginary Coupled Oscillator")
    plt1, =plt.plot(time_array, oscillator_values[0])
    plt2, =plt.plot(time_array, oscillator_values[1])
    plt3, =plt.plot(time_array, oscillator_values[2])
    plt4, =plt.plot(time_array, oscillator_values[3])
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
    peakind = signal.find_peaks_cwt(oscillator_values[0], time_array, min_snr=5)
    peakind = np.array(peakind)
    print (peakind)


    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
    foot_labels = ["Front Right Foot", "Back Right Foot", "Front Left Foot", "Back Left Foot"]
    hip_labels = ["Front Right Hip", "Back Right Hip", "Front Left Hip", "Back Left Hip"]
    plt.subplot(3,2,1)
    plt.title("Foot Imaginary Coupled Oscillator")
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
    im = ax.imshow(lamb)
    plt.title("Coupled Oscillator Coefficients")
    ax.set_xticks(np.arange(len(foot_labels)))
    ax.set_yticks(np.arange(len(foot_labels)))
    ax.set_xticklabels(foot_labels)
    ax.set_yticklabels(foot_labels)

    plt.show()
if plot == "physics2":
    plt.figure(figsize=(20,20))
    # plt.subplot(1,1,1)
    plt.subplot(4,1,1)
    plt.ylim([0,20])
    plt.title("Cost Of Transport")
    plt.xlabel("Time Step (t) (Measurement taken every second)")
    plt.ylabel("Cost Of Transport ")
    plt.plot(sampling_array, cost_of_transport)
    plt.subplot(4,1 ,2)

    plt.title("Forces")
    plt.ylim([0,np.max(force_values)])
    plt.plot(sampling_array, force_values)
    plt.subplot(4,1,3)
    plt.title("velocity")
    plt.ylim([0,20])
    plt.plot(sampling_array, velocity_array)

    plt.subplot(4,1,4)
    plt.title("displacement")
    plt.ylim([0,np.max(displacement_array)])
    plt.plot(sampling_array, displacement_array)
    #
    # plt.subplot(4,1,4)
    # plt.title("Displacement Array")
    # plt.scatter(displacement_array, force_values)
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
    print(force_values)

    plt.ylabel("Z Value")
    plt.xlabel("Time Step (t)")
    plt.savefig(run_name, dpi=250)
    plt.show()
