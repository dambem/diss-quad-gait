import pybullet as p
import numpy as np
import time

p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
p.setDefaultContactERP(0)
urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
# urdfFlags = p.URDF_USE_SELF_COLLISION
debug = False;
if (debug):
	quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=True)
else:
	quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)

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

maxForceId = p.addUserDebugParameter("maxForce",0,100,40)

for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        #print(info)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                jointIds.append(j)


p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

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


front_right_hip = 1
front_left_hip = 4
back_right_hip = 7
back_left_hip = 10
front_right_foot = 2
front_left_foot = 5
back_right_foot = 8
back_left_foot = 11
front_right = 3
front_left = 6
back_right = 9
back_left = 0
# c = paramIds[i]
#
# targetPos = p.readUserDebugParameter(c)
maxForce = p.readUserDebugParameter(maxForceId)

jointOffsets[1] -= -0.5
jointOffsets[4] -= -0.5
jointOffsets[7] -= -0.5
jointOffsets[10] -= -0.5
sin =  np.sin(0)
sin2 = np.sin(np.pi*0.5)
sin3 = np.sin(np.pi)
sin4 = np.sin(np.pi*3/4)

p.setRealTimeSimulation(1)
# Begins timer to allow for sin function to work (Will replace with vanderpol in future)
p.setJointMotorControl2(quadruped, front_right_hip,p.POSITION_CONTROL, -jointOffsets[1], force=maxForce)
p.setJointMotorControl2(quadruped, front_left_hip,p.POSITION_CONTROL, -jointOffsets[4], force=maxForce)
p.setJointMotorControl2(quadruped, back_right_hip,p.POSITION_CONTROL, -jointOffsets[7], force=maxForce)
p.setJointMotorControl2(quadruped, back_left_hip,p.POSITION_CONTROL,  -jointOffsets[10], force=maxForce)
p.setJointMotorControl2(quadruped, front_right_foot,p.POSITION_CONTROL, -jointOffsets[2]-(0.3*sin), force=maxForce)
p.setJointMotorControl2(quadruped, front_left_foot,p.POSITION_CONTROL, -jointOffsets[5]-(0.3*sin3), force=maxForce)
p.setJointMotorControl2(quadruped, back_right_foot,p.POSITION_CONTROL, -jointOffsets[8]+(0.3*sin2), force=maxForce)
p.setJointMotorControl2(quadruped, back_left_foot,p.POSITION_CONTROL, -jointOffsets[11]+(0.3*sin4), force=maxForce)
time.sleep(1);
start = time.time();

def deg_to_rad(deg):
	return deg*(np.pi/180)
phase_time = 20
foot_angle = deg_to_rad(9)
hip_angle = deg_to_rad(6)
foot_debug = deg_to_rad(180)
hip_debug = deg_to_rad(60)
maxForceId = p.addUserDebugParameter("Limb Velocity",0,100,phase_time)
while (1):
	end = time.time()
	maxForce = p.readUserDebugParameter(maxForceId)
	sin = np.sin((start-end)*phase_time)
	sin2 = np.sin(np.pi*0.5+(start-end)*phase_time)
	sin3 = np.sin(np.pi*1/4+(start-end)*phase_time)
	sin4 = np.sin(np.pi*3/4+(start-end)*phase_time)
	sin_debug = np.sin(start-end)
	# Angles are in Radians (PI = 360)


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
		p.setJointMotorControl2(quadruped, front_left,p.POSITION_CONTROL,  0, force=maxForce)
		p.setJointMotorControl2(quadruped, front_right,p.POSITION_CONTROL,  0, force=maxForce)
		p.setJointMotorControl2(quadruped, back_left,p.POSITION_CONTROL,  0, force=maxForce)
		p.setJointMotorControl2(quadruped, back_right,p.POSITION_CONTROL,  0, force=maxForce)
	# p.setJointMotorControl2(quadruped, 10,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
	# p.setJointMotorControl2(quadruped, 7,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
	# p.setJointMotorControl2(quadruped, 4,p.POSITION_CONTROL,jointDirections[i]*targetPos+sin+jointOffsets[i], force=maxForce)
