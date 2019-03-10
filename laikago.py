import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
from scipy.integrate import odeint




start_y = [1,1,1,1]
start_x = [0,0,0,0]
new_y = [1,1,1,1]
new_x = [0,0,0,0]

start_y2 = [1,1,1,1]
start_x2 = [0,0,0,0]
new_y2 = [1,1,1,1]
new_x2 = [0,0,0,0]
count = 0
# Initial Configuration (Can Later Be Changed Through User Parameters)
run_array = []
gravity = -9.8
time_step = 1./500
frequency_multiplier = 175
foot_angle = 5
hip_angle = 5
max_force = 25
w = 20
mu = 1
p_v = 2
# Hip Configurations (SET, DO NOT CHANGE)start_x
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

# Initial Runtime Array configuration
run_array.append(["gravity", [[gravity, 0]], 'kg/s^2'])
run_array.append(["time_step", [[time_step, 0]], 's'])
run_array.append(["frequency_multiplier", [[frequency_multiplier, 0]], 'n/a'])
run_array.append(["foot_angle", [[foot_angle, 0]], 'deg'])
run_array.append(["hip_angle", [[hip_angle, 0]], 'deg'])
run_array.append(["max_force", [[foot_angle, 0]], 'deg'])

run_simulation = 0
p.connect(p.GUI)


plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, gravity)

p.setTimeStep(time_step)
p.setDefaultContactERP(0)
urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
# urdfFlags = p.URDF_USE_SELF_COLLISION
debug = False;
cube = p.loadURDF("cube.urdf", [0.6,0,0.5],[0,1,0, 0], flags = urdfFlags, useFixedBase=True)
cube2 = p.loadURDF("cube.urdf", [-0.6,0,0.5],[0,1,0, 0], flags = urdfFlags, useFixedBase=True)

if (debug):
    quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=True)
else:
    quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)

base_dynamics_info = p.getDynamicsInfo(quadruped, -1)
frh_dynamics_info = p.getDynamicsInfo(quadruped, front_right_hip)
flh_dynamics_info = p.getDynamicsInfo(quadruped, front_left_hip)


run_array.append(['base mass', [[base_dynamics_info[0], 0]]])

#enable collision between lower legs
for j in range (p.getNumJoints(quadruped)):
        print(p.getJointInfo(quadruped,j))

#2,5,8 and 11 are the lower legs
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
maxForceId = p.addUserDebugParameter("maxForce",0,100,max_force)

for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0.5, angularDamping=0.5)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                jointIds.append(j)


p.getCameraImage(480,320)
p.setRealTimeSimulation(run_simulation)


joints=[]
#


for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0.5, angularDamping=0.5)
        info = p.getJointInfo(quadruped,j)
        js = p.getJointState(quadruped,j)
        # print(info)
        jointName = info[1]
        jointType = info[2]
        # if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        #         paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-jointOffsets[j])/jointDirections[j]))


# c = paramIds[i]
#
# targetPos = p.readUserDebugParameter(c)
maxForce = p.readUserDebugParameter(maxForceId)
wId = p.addUserDebugParameter("w (Legs)",0,100,w)
muId = p.addUserDebugParameter("mu (Legs)", 0, 100, mu)
pvId = p.addUserDebugParameter("p_v (Legs)", 0, 100, p_v)

phaseTimeId = p.addUserDebugParameter("Frequency Multiplier",frequency_multiplier-(frequency_multiplier/2) ,frequency_multiplier+(frequency_multiplier/2),frequency_multiplier)
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
lamb_trot = -0.2
lamb_pace = -0.2
lamb_bound = -0.2
lamb_walk = 0.2
#
# lamb = [  [0,lamb_bound,lamb_pace,lamb_trot],
#           [lamb_bound, 0, lamb_walk, lamb_pace],
#           [lamb_pace, lamb_walk, 0, lamb_bound],
#           [lamb_trot, lamb_pace, lamb_bound, 0]]

lamb = [ [0, lamb_bound, lamb_pace, lamb_trot],
       [lamb_bound, 0, lamb_trot, lamb_pace],
       [lamb_pace, lamb_trot, 0, lamb_pace],
       [lamb_trot, lamb_pace, lamb_bound, 0]]
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

def van_der_pol_coupled(x, t):
    # global chosen_x
    x0 = x[1]
    x_ai =x[0]
    for j in range(4):
        x_ai += x[0]-(lamb[current_i][j]*chosen_x[j])
    # x_ai *= time_step
    # osc = odeint(van_der_pol_oscillator_deriv, [x[0], x[1]], [t-time_step, t])
    x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w + (0.5+feedback[current_i])
    res = np.array([x0, x1])
    return res

def van_der_pol_coupled2(x, t):
    # global chosen_x2
    x0 = x[1]
    x_ai =x[0]
    # print(current_i2)
    for j in range(4):
        x_ai += x[0]-(lamb[current_i2][j]*chosen_x2[j])
    # x_ai *= time_step
    # osc = odeint(van_der_pol_oscillator_deriv, [x[0], x[1]], [t-time_step, t])
    x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w + feedback2[current_i2]
    res = np.array([x0, x1])
    return res

qKey = ord('q')
pKey = ord('p')
rKey = ord('r')
run_string= maxForce + frequency_multiplier + foot_angle + hip_angle

time_step2 = 0.03
oscillator_values = [[],[],[],[]]
oscillator_values2 = [[],[],[],[]]
force_values= [[],[]]
limb_values = [[],[],[],[], [], [], [], []]
time_stepId = p.addUserDebugParameter("Oscillator Time Step", 0.01, 1, time_step2)
time.sleep(1)
speed_array = []
total_displacement = 0
total_distance = 0
total_force = 0
force_expended = 0
cost_of_transport= [[],[]]
sample_timer = 0
# TODO, CHANGE SAMPLING RATE TO TIME FOR A FULL OSCILLATION
sampling_rate = 30*time_step
while (1):
    keys = p.getKeyboardEvents()
    # if rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED:cebook.com/
    #     p.resetBasePositionAndOrientation(quadruped, 0, 0, 0)
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        break;
    if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
        run_simulation = not run_simulation
        p.setRealTimeSimulation(run_simulation)
    if (run_simulation):
        x_list = []
        x_list2 = []
        time_step2 = p.readUserDebugParameter(time_stepId)
        count += time_step2
        timer += time_step
        if (maxForce != p.readUserDebugParameter(maxForceId)):
            run_array[5][1].append([p.readUserDebugParameter(maxForceId), timer])
        maxForce = p.readUserDebugParameter(maxForceId)
        frequency_multiplier = p.readUserDebugParameter(phaseTimeId)
        foot_angle = deg_to_rad(p.readUserDebugParameter(foot_angleId))
        hip_angle = deg_to_rad(p.readUserDebugParameter(hip_angleId))
        w = p.readUserDebugParameter(wId)
        mu = p.readUserDebugParameter(muId)
        p_v = int(p.readUserDebugParameter(pvId))
        sin = np.sin((timer)*frequency_multiplier)
        sin2 = np.sin(np.pi*0.5+(timer)*frequency_multiplier)
        sin3 = np.sin(np.pi*1/4+(timer)*frequency_multiplier)
        sin4 = np.sin(np.pi*3/4+(timer)*frequency_multiplier)

        # sin3 = np.sin((timer)*frequency_multiplier)
        # sin4 = np.sin(np.pi*0.5+(timer)*frequency_multiplier)
        # sin = np.sin(np.pi*1/4+(timer)*frequency_multiplier)
        # sin2 = np.sin(np.pi*3/4+(timer)*frequency_multiplier)
        # sin_debug = np.sin(timer)
        pos_ori = p.getBasePositionAndOrientation(quadruped)*2

        # plt.scatter(pos_ori[0][0]+pos_ori[0][1]+pos_ori[0][2], counter)
        # plt.pause(0.000001)
        # print(p.getBasePositionAndOrientation(quadruped))
        # print (pos_ori[0][0])
        # Angles are in Radians (PI = 360)
        # frf_state = p.getJointState(quadruped, front_right_foot)
        p.addUserDebugLine((pos_ori[0][0], pos_ori[0][1], pos_ori[0][2]), (pos_ori[0][0]+0.1, pos_ori[0][1], pos_ori[0][2]))
        distance = (((pos_ori[0][1])**2) + ((pos_ori[0][0])**2))**(1/2)
        try:
            displacement = (distance-distance_array[-1])
        except Exception as e:
            displacement = 0
        distance_array.append(distance)
        speed = displacement/time_step
        total_distance = distance
        speed_array.append(speed)
        height_array.append(pos_ori[0][2])
        turn_array.append(pos_ori[0][0])
        x_tilt_array.append(pos_ori[1][0])
        y_tilt_array.append(pos_ori[1][1])
        z_tilt_array.append(pos_ori[1][2])
        if (sample_timer >= sampling_rate):
            force_values[0].append(timer)
            measurements_taken = (sample_timer/time_step)
            force_values[1].append(total_force/measurements_taken)
            total_force = 0
            sample_timer = 0
            # cost_of_transport[0].append(time r)
            # cost_of_transport[1].append()
            # force_values[0].append(total_froce)
            # force_expended = 0
        # else:
        sample_timer += time_step
        force_expended =0
        for n in range(12):
            force_expended += (abs(p.getJointState(quadruped, n)[2][0]/100)*max_force)**2
        force_expended = force_expended**(1/2)#
        print(force_expended)
        total_force += force_expended

        total_displacement = distance
        # force_values[0].append(force_expended)

        # print(pos_ori[0][0])
        # print(pos_ori[0][1])
        # print(pos_ori[0][2])
        # print (left_leg)

            # print(n)


        fr_foot_forces = abs(p.getJointState(quadruped, front_right_foot)[2][0]/100*maxForce)
        br_foot_forces = abs(p.getJointState(quadruped, back_right_foot)[2][0]/100*maxForce)
        fl_foot_forces = abs(p.getJointState(quadruped, front_left_foot)[2][0]/100*maxForce)
        bl_foot_forces = abs(p.getJointState(quadruped, back_left_foot)[2][0]/100*maxForce)
        fr_hip_forces = abs(p.getJointState(quadruped, front_right_hip)[2][0]/100*maxForce)
        br_hip_forces = abs(p.getJointState(quadruped, back_right_hip)[2][0]/100*maxForce)
        fl_hip_forces = abs(p.getJointState(quadruped, front_left_hip)[2][0]/100*maxForce)
        bl_hip_forces = abs(p.getJointState(quadruped, back_left_hip)[2][0]/100*maxForce)

        fr_foot_rot = p.getJointState(quadruped, front_right_foot)[0]
        br_foot_rot = p.getJointState(quadruped, back_right_foot)[0]
        fl_foot_rot = p.getJointState(quadruped, front_left_foot)[0]
        bl_foot_rot = p.getJointState(quadruped, back_left_foot)[0]


        fr_hip_rot = p.getJointState(quadruped, front_right_hip)[0]
        fl_hip_rot = p.getJointState(quadruped, front_left_hip)[0]
        br_hip_rot = p.getJointState(quadruped, back_right_hip)[0]
        bl_hip_rot = p.getJointState(quadruped, back_left_hip)[0]
        feedback = [fr_foot_rot, br_foot_rot, fl_foot_rot, bl_foot_rot]
        feedback2 = [fr_hip_rot, br_hip_rot, fl_hip_rot, bl_hip_rot]
        lamb_pace = p.readUserDebugParameter(lambId)
        lamb_bound = p.readUserDebugParameter(lambBoundId)
        lamb_trot = p.readUserDebugParameter(lambTrotId)
        lamb_walk = p.readUserDebugParameter(lambWalkId)
        lamb = [ [0, lamb_bound, lamb_pace, lamb_trot],
               [lamb_bound, 0, lamb_trot, lamb_pace],
               [lamb_pace, lamb_trot, 0, lamb_pace],
               [lamb_trot, lamb_pace, lamb_bound, 0]]


        for i in range(4):
            current_i = i
            chosen_x = start_x
            osc= odeint(van_der_pol_coupled, [start_y[i], start_x[i]], [count-time_step2, count])


            x = osc[1][1]
            y = osc[1][0]
            x_list.append(x)
            new_y[i] = y
            new_x[i] = x

        for i in range(4):
            current_i2 = i
            chosen_x2 = start_x2
            # mu = 1
            osc2= odeint(van_der_pol_coupled2, [start_y2[i], start_x2[i]], [count-time_step2, count])
            x2 = osc2[1][1]
            y2 = osc2[1][0]
            x_list2.append(x2)
            new_y2[i] = y2
            new_x2[i] = x2

        start_y = new_y
        start_x = new_x
        start_y2 = new_y2
        start_x2 = new_x2


        time_array.append(timer)
        oscillator_values2[0].append(x_list2[0])
        oscillator_values2[1].append(x_list2[1])
        oscillator_values2[2].append(x_list2[2])
        oscillator_values2[3].append(x_list2[3])
        oscillator_values[0].append(x_list[0])
        oscillator_values[1].append(x_list[1])
        oscillator_values[2].append(x_list[2])
        oscillator_values[3].append(x_list[3])
        limb_values[0].append(fr_foot_rot)
        limb_values[1].append(br_foot_rot)
        limb_values[2].append(fl_foot_rot)
        limb_values[3].append(bl_foot_rot)

        limb_values[4].append(fr_hip_rot)
        limb_values[5].append(br_hip_rot)
        limb_values[6].append(fl_hip_rot)
        limb_values[7].append(bl_hip_rot)

        # p.addUserDebugLine((frf_state[0], frf_state[1], pos_ori[0][2]), (frf_state[0], frf_state[1], pos_ori[0][2]))
        if (debug):
            # p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, foot_debug*sin_debug, force=maxForce)
            p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, hip_debug*sin_debug, force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL, hip_debug*sin_debug, force=maxForce)
            p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, foot_debug, force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, foot_debug, force=maxForce)
        # Begins timer to allow for sin
        else:
            p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL,-jointOffsets[2]+(foot_angle*x_list[0]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, front_left_foot,p.POSITION_CONTROL, -jointOffsets[5]+(foot_angle*x_list[2]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, -jointOffsets[8]+(foot_angle*x_list[1]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_left_foot,p.POSITION_CONTROL,  -jointOffsets[11]+(foot_angle*x_list[3]*0.1), force=maxForce)
            #
            p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1]+(hip_angle*x_list2[0]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL,  -jointOffsets[4]+(hip_angle*x_list2[2]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL,  -jointOffsets[7]+(hip_angle*x_list2[1]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,   -jointOffsets[10]+(hip_angle*x_list2[3]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, front_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, front_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, back_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)


    # p.setJointMotorControl2(quadruped, 10,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
    # p.setJointMotorControl2(quadruped, 7,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
    # p.setJointMotorControl2(quadruped, 4,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)

run_name = 'laikago-sin-'+datetime.datetime.today().strftime('%Y-%s')+"-"
run_log = open(run_name+"log.txt", "w+")

print (total_distance)
print (timer)
print (total_distance/timer)
print (total_force)
power = total_force * (total_distance/timer)
print (power)
for items in run_array:
    string = ""
    string = str(items[0])+": [ "
    for specific in items[1]:
        string += "[V: " + str(specific[0]) + " T: " + str(specific[1]) + "]"
    string += " ]\n"
    run_log.write(string)
run_log.close()
plot = "physics"
if plot == "oscillators":
    plt.figure(figsize=(15,15))
    plt.subplot(4,1,1)
    plt.title("X Value 1")
    plt.xlabel("Time Step (t)")
    plt.ylabel("X Value")
    plt.plot(time_array, oscillator_values[0])
    plt.plot(time_array, oscillator_values[1])
    plt.plot(time_array, oscillator_values[2])
    plt.plot(time_array, oscillator_values[3])
    plt.subplot(4,1,2)

    plt.title("X Value 1")
    plt.xlabel("Time Step (t)")
    plt.ylabel("X Value")
    plt.plot(time_array, oscillator_values2[0])
    plt.plot(time_array, oscillator_values2[1])
    plt.plot(time_array, oscillator_values2[2])
    plt.plot(time_array, oscillator_values2[3])
    plt.title("X Value 1")
    plt.subplot(4,1,3)
    plt.plot(time_array, limb_values[0], c="r")
    plt.plot(time_array, limb_values[1],  c="b")
    plt.plot(time_array, limb_values[2],  c="g")
    plt.plot(time_array, limb_values[3],  c="y")
    plt.subplot(4,1,4)
    plt.plot(time_array, limb_values[4])
    plt.plot(time_array, limb_values[5])
    plt.plot(time_array, limb_values[6])
    plt.plot(time_array, limb_values[7])
    plt.show()
if plot == "physics2":
    plt.figure(figsize=(15,15))
    plt.subplot(1,1,1)
    plt.title("Cost Of Transport")
    plt.xlabel("Time Step (t) (Measurement taken every second)")
    plt.ylabel("Cost Of Transport ")

if plot == "physics":
    plt.figure(figsize=(15,15))
    plt.subplot(3, 3, 1)
    plt.title("Z Distance Travelled Over Time")
    plt.xlabel("Time Step (t/"+str(time_step)+")")
    plt.ylabel("Distance")
    plt.plot(time_array, distance_array)
    plt.subplot(3, 3, 2)
    plt.title("Speed Variation Over Time")
    plt.plot(time_array, speed_array)
    plt.ylabel("velocity (displacement/time)")
    plt.xlabel("Time Step (t)")
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
    plt.title("Forces")
    plt.ylim([0,max_force])
    plt.plot(force_values[0], force_values[1])
    # plt.plot(time_array, force_values[1])
    # plt.plot(time_array, force_values[2])
    # plt.plot(time_array, force_values[3])
    # plt.plot(time_array, force_values[4])
    # plt.plot(time_array, force_values[5])
    # plt.plot(time_array, force_values[6])
    # plt.plot(time_array, force_values[7])
    plt.ylabel("Z Value")
    plt.xlabel("Time Step (t)")
    plt.savefig(run_name, dpi=250)
    plt.show()
