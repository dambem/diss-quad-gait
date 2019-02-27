import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
from scipy.integrate import odeint

lamb = [  [0,-0.2,-0.2,-0.2],
          [-0.2, 0, -0.2, -0.2],
          [-0.2, -0.2, 0, -0.2],
          [-0.2, -0.2, -0.2, 0]]

lamb2 = [[0, -0.2, 0.2, -0.2],
         [-0.2, 0, -0.2, 0.2],
         [0.2, -0.2, 0, -0.2],
         [-0.2, 0.2, -0.2, 0]]

lamb_control = [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]
current_i = 0
def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p_v - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res
def van_der_pol_coupled(x, t):
    x0 = x[1]
    x_ai =x[0]
    for j in range(4):
        x_ai += x[0]-(lamb[current_i][j]*start_x[j])
    # x_ai *= time_step
    # osc = odeint(van_der_pol_oscillator_deriv, [x[0], x[1]], [t-time_step, t])
    x1 =  mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
    res = np.array([x0, x1])
    return res


start_y = [np.pi, np.pi, np.pi, np.pi]
start_x = [0,0,0,0]
new_y = [np.pi, np.pi, np.pi, np.pi]
new_x = [0,0,0,0]
count = 0
# Initial Configuration (Can Later Be Changed Through User Parameters)
run_array = []
gravity = -9.8
time_step = 1./500
frequency_multiplier = 175
foot_angle = 15
hip_angle = 6
max_force = 90
w = 1
mu = 1
p_v = 1
# Hip Configurations (SET, DO NOT CHANGE)
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
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        #print(info)
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
jointOffsets[1] -= -0.5
jointOffsets[4] -= -0.5
jointOffsets[7] -= -0.5
jointOffsets[10] -= -0.5
sin =  np.sin(0)
sin2 = np.sin(np.pi*0.5)
sin3 = np.sin(np.pi)
sin4 = np.sin(np.pi*3/4)
run_simulation = 1
p.setRealTimeSimulation(run_simulation)
# Begins timer to allow for sin function to work (Will replace with vanderpol in future)
p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1], force=maxForce)
p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL, -jointOffsets[4], force=maxForce)
p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL, -jointOffsets[7], force=maxForce)
p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,  -jointOffsets[10], force=maxForce)
p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, -jointOffsets[2]-(0.3*sin), force=maxForce)
p.setJointMotorControl2(quadruped, front_left_foot,p.POSITION_CONTROL, -jointOffsets[5]-(0.3*sin3), force=maxForce)
p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, -jointOffsets[8]+(0.3*sin2), force=maxForce)
p.setJointMotorControl2(quadruped, back_left_foot,p.POSITION_CONTROL, -jointOffsets[11]+(0.3*sin4), force=maxForce)
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

qKey = ord('q')
pKey = ord('p')
rKey = ord('r')
run_string= maxForce + frequency_multiplier + foot_angle + hip_angle

time_step2 = 0.04
oscillator_values = [[],[],[],[]]
limb_values = [[],[],[],[], [], [], [], []]
time_stepId = p.addUserDebugParameter("Oscillator Time Step", 0.01, 1, time_step2)
while (1):
    keys = p.getKeyboardEvents()
    # if rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED:
    #     p.resetBasePositionAndOrientation(quadruped, 0, 0, 0)
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        break;
    if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
        run_simulation = not run_simulation
        p.setRealTimeSimulation(run_simulation)
    if (run_simulation):
        x_list = []
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
        pos_ori = p.getBasePositionAndOrientation(quadruped)


        # plt.scatter(pos_ori[0][0]+pos_ori[0][1]+pos_ori[0][2], counter)
        # plt.pause(0.000001)
        # print(p.getBasePositionAndOrientation(quadruped))
        # print (pos_ori[0][0])
        # Angles are in Radians (PI = 360)
        # frf_state = p.getJointState(quadruped, front_right_foot)
        p.addUserDebugLine((pos_ori[0][0], pos_ori[0][1], pos_ori[0][2]), (pos_ori[0][0]+0.1, pos_ori[0][1], pos_ori[0][2]))
        distance_array.append(pos_ori[0][1])
        height_array.append(pos_ori[0][2])
        turn_array.append(pos_ori[0][0])
        x_tilt_array.append(pos_ori[1][0])
        y_tilt_array.append(pos_ori[1][1])
        z_tilt_array.append(pos_ori[1][2])

        # print(pos_ori[0][0])
        # print(pos_ori[0][1])
        # print(pos_ori[0][2])
        # print (left_leg)
        for i in range(4):
            current_i = i
            osc= odeint(van_der_pol_coupled, [start_y[i], start_x[i]], [count-time_step2, count])
            x = osc[1][1]
            y = osc[1][0]
            x_list.append(x)
            new_y[i] = y
            new_x[i] = x



        start_y = new_y
        start_x = new_x
        fr_foot_rot = p.getJointState(quadruped, front_right_foot)[0]
        fl_foot_rot = p.getJointState(quadruped, front_left_foot)[0]
        br_foot_rot = p.getJointState(quadruped, back_right_foot)[0]
        bl_foot_rot = p.getJointState(quadruped, back_left_foot)[0]

        fr_hip_rot = p.getJointState(quadruped, front_right_hip)[0]
        fl_hip_rot = p.getJointState(quadruped, front_left_hip)[0]
        br_hip_rot = p.getJointState(quadruped, back_right_hip)[0]
        bl_hip_rot = p.getJointState(quadruped, back_left_hip)[0]

        time_array.append(timer)
        oscillator_values[0].append(x_list[0])
        oscillator_values[1].append(x_list[1])
        oscillator_values[2].append(x_list[2])
        oscillator_values[3].append(x_list[3])
        limb_values[0].append(fr_foot_rot)
        limb_values[1].append(fl_foot_rot)
        limb_values[2].append(br_foot_rot)
        limb_values[3].append(bl_foot_rot)

        limb_values[4].append(fr_hip_rot)
        limb_values[5].append(fl_hip_rot)
        limb_values[6].append(br_hip_rot)
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
            p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1]+(hip_angle*x_list[0]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL,  -jointOffsets[4]+(hip_angle*x_list[2]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL,  -jointOffsets[7]+(hip_angle*x_list[1]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,   -jointOffsets[10]+(hip_angle*x_list[3]*0.1), force=maxForce)
            p.setJointMotorControl2(quadruped, front_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, front_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, back_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
            p.setJointMotorControl2(quadruped, back_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)


    # p.setJointMotorControl2(quadruped, 10,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
    # p.setJointMotorControl2(quadruped, 7,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
    # p.setJointMotorControl2(quadruped, 4,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)

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
if plot == "oscillators":
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.title("X Value 1")
    plt.xlabel("Time Step (t)")
    plt.ylabel("X Value")
    plt.plot(time_array, oscillator_values[0], c='red')
    plt.plot(time_array, oscillator_values[1], c='blue')
    plt.plot(time_array, oscillator_values[2], c='green')
    plt.plot(time_array, oscillator_values[3], c='yellow')
    plt.subplot(3,1,2)
    plt.plot(time_array, limb_values[0], c='red')
    plt.plot(time_array, limb_values[1], c='blue')
    plt.plot(time_array, limb_values[2], c='green')
    plt.plot(time_array, limb_values[3], c='yellow')
    plt.subplot(3,1,3)
    plt.plot(time_array, limb_values[4], c='red')
    plt.plot(time_array, limb_values[5], c='blue')
    plt.plot(time_array, limb_values[6], c='green')
    plt.plot(time_array, limb_values[7], c='yellow')
    plt.show()
if plot == "physics":
    plt.figure(figsize=(15,15))
    plt.subplot(3, 3, 1)
    plt.title("Z Distance Travelled Over Time")
    plt.xlabel("Time Step (t/"+time_step+")")
    plt.ylabel("Distance")
    plt.plot(time_array, distance_array)
    plt.subplot(3, 3, 2)
    plt.title("Height Variation Over Time")
    plt.plot(time_array, height_array)
    plt.ylabel("Height from Center Point")
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
    plt.title("Z Rotation Over Time")
    plt.plot(time_array, z_tilt_array)
    plt.ylabel("Z Value")
    plt.xlabel("Time Step (t)")
    plt.savefig(run_name, dpi=250)
    plt.show()
