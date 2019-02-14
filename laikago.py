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
    x_ai = x[0]
    for j in range(4):
        x_ai += (lamb_control[current_i][j]*start_x[j])
    x1 = mu * ((p_v - (x_ai** 2.0))* x0) - x_ai*w
    res = np.array([x0, x1])
    return res
start_y = [1,1,1,1]
start_x = [0,0,0,0]
new_y = [1,1,1,1]
new_x = [0,0,0,0]
time_step = 0.05

count = 0
# Initial Configuration (Can Later Be Changed Through User Parameters)
run_array = []
gravity = -9.8
time_step = 1./500
frequency_multiplier = 175
foot_angle = 15
hip_angle = 6
max_force = 15

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
	jointOffsets.append(-0.7)
	jointOffsets.append(0.7)
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
# with open("data1.txt","r") as filestream:
# 	for line in filestream:
# 		maxForce = p.readUserDebugParameter(maxForceId)
# 		currentline = line.split(",")
# 		frame = currentline[0]
# 		t = currentline[1]
# 		joints=currentline[2:14]
# 		for j in range (12):
# 			targetPos = float(joints[j])
# 			p.setJointMotorControl2(quadruped,jointIds[j],p.POSITION_CONTROL,jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)
# 		p.stepSimulation()
# 		for lower_leg in lower_legs:
# 			#print("points for ", quadruped, " link: ", lower_leg)
# 			pts = p.getContactPoints(quadruped,-1, lower_leg)
# 			#print("num points=",len(pts))
# 			#for pt in pts:
# 			#	print(pt[9])
# 		time.sleep(1./500.)


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
time.sleep(5);

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
# plt.scatter(0,1)
qKey = ord('q')
pKey = ord('p')
run_string= maxForce + frequency_multiplier + foot_angle + hip_angle

w = 20
mu = 1
p_v = 2
w = 20
while (1):
	keys = p.getKeyboardEvents()
	if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
	    break;
	if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
		run_simulation = not run_simulation
		p.setRealTimeSimulation(run_simulation)
	if (run_simulation):
		timer += time_step
		if (maxForce != p.readUserDebugParameter(maxForceId)):
			run_array[5][1].append([p.readUserDebugParameter(maxForceId), timer])
		maxForce = p.readUserDebugParameter(maxForceId)
		frequency_multiplier = p.readUserDebugParameter(phaseTimeId)
		foot_angle = deg_to_rad(p.readUserDebugParameter(foot_angleId))
		hip_angle = deg_to_rad(p.readUserDebugParameter(hip_angleId))

		sin = np.sin((timer)*frequency_multiplier)
		sin2 = np.sin(np.pi*0.5+(timer)*frequency_multiplier)
		sin3 = np.sin(np.pi*1/4+(timer)*frequency_multiplier)
		sin4 = np.sin(np.pi*3/4+(timer)*frequency_multiplier)

		# sin3 = np.sin((timer)*frequency_multiplier)
		# sin4 = np.sin(np.pi*0.5+(timer)*frequency_multiplier)
		# sin = np.sin(np.pi*1/4+(timer)*frequency_multiplier)
		# sin2 = np.sin(np.pi*3/4+(timer)*frequency_multiplier)
		sin_debug = np.sin(timer)
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
		time_array.append(timer)
		# p.addUserDebugLine((frf_state[0], frf_state[1], pos_ori[0][2]), (frf_state[0], frf_state[1], pos_ori[0][2]))
		if (debug):
			# p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, foot_debug*sin_debug, force=maxForce)
			p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, hip_debug*sin_debug, force=maxForce)
			p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL, hip_debug*sin_debug, force=maxForce)
			p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, foot_debug, force=maxForce)
			p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, foot_debug, force=maxForce)
		# Begins timer to allow for sin
		else:
			p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL,-jointOffsets[2]+(foot_angle*sin), force=maxForce)
			p.setJointMotorControl2(quadruped, front_left_foot,p.POSITION_CONTROL, -jointOffsets[5]+(foot_angle*sin3), force=maxForce)
			p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, -jointOffsets[8]+(foot_angle*sin2), force=maxForce)
			p.setJointMotorControl2(quadruped, back_left_foot,p.POSITION_CONTROL,  -jointOffsets[11]+(foot_angle*sin4), force=maxForce)
			#
			p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1]+(hip_angle*sin), force=maxForce)
			p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL,  -jointOffsets[4]+(hip_angle*sin3), force=maxForce)
			p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL,  -jointOffsets[7]+(hip_angle*sin2), force=maxForce)
			p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,   -jointOffsets[10]+(hip_angle*sin4), force=maxForce)
			p.setJointMotorControl2(quadruped, front_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
			p.setJointMotorControl2(quadruped, front_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
			p.setJointMotorControl2(quadruped, back_left_shoulder,p.POSITION_CONTROL,  0, force=maxForce)
			p.setJointMotorControl2(quadruped, back_right_shoulder,p.POSITION_CONTROL,  0, force=maxForce)


	# p.setJointMotorControl2(quadruped, 10,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
	# p.setJointMotorControl2(quadruped, 7,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
	# p.setJointMotorControl2(quadruped, 4,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)

run_name = 'laikago-sin-'+datetime.datetime.today().strftime('%YYYY-%MM-%DD-%HH-%mm-%ss')+"-"
run_log = open(run_name+"log.txt", "w+")
for items in run_array:
	string = ""
	string = str(items[0])+": [ "
	for specific in items[1]:
		string += "[V: " + str(specific[0]) + " T: " + str(specific[1]) + "]"
	string += " ]\n"
	run_log.write(string)
run_log.close()

plt.figure(figsize=(15,15))
plt.subplot(3, 3, 1)
plt.title("Z Distance Travelled Over Time")
plt.xlabel("Time Step (t)")
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
