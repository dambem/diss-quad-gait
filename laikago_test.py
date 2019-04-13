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
# foot_angle = deg_to_rad(float(sys.argv[5]))
# hip_angle = deg_to_rad(float(sys.argv[6]))
#
# max_force = float(sys.argv[1])
# oscillator_step = float(sys.argv[2])
hip_height = 1
w = 20
van_multi = 0.1

mu = 1
p_v = 2
num_iterations = 11000
num_epochs = 10
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
# position_array = np.zeros((num_epochs, 3, num_iterations))
# time_array = np.zeros((num_epochs, num_iterations))
# # displacement_array = np.zeros(num_iterations)
# force_array = np.zeros((num_epochs,num_iterations))
# distance_array = np.zeros((num_epochs,num_iterations))
# period_foot = np.zeros((num_epochs, num_iterations))
# tilt_array = np.zeros((num_epochs, 3, num_iterations))
# height_array = np.zeros((num_epochs,num_iterations))
# turn_array = np.zeros((num_epochs, num_iterations))
# print(str(e/num_epochs*100)+ "%")
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
# plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, gravity)
p.setTimeStep(time_step)
p.setDefaultContactERP(0)

urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

debug = False;
# cube = p.loadURDF("cube.urdf", [0.31,0,0.36],[0,5,0, 0], flags = urdfFlags, useFixedBase=True)
# cube2 = p.loadURDF("cube.urdf", [-0.31,0,0.36],[0,5,0, 0], flags = urdfFlags, useFixedBase=True)
quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)
# quadruped2 = p.loadURDF("laikago/laikago.urdf",[0,1,0.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)
base_dynamics_info = p.getDynamicsInfo(quadruped, -1)
frh_dynamics_info = p.getDynamicsInfo(quadruped, front_right_hip)
flh_dynamics_info = p.getDynamicsInfo(quadruped, front_left_hip)
pos_ori = p.getBasePositionAndOrientation(quadruped)
# pos_ori2 = p.getBasePositionAndOrientation(quadruped2)
print(pos_ori)
# print(pos_ori2)
# p.createVisualShape(p.GEOM_PLANE)
cube = p.loadURDF("cube2.urdf", [0.1, 0, 0.5], [1, 1, 0, 0], flags = urdfFlags, useFixedBase=True)
p.getCameraImage(1920,1080)
p.setRealTimeSimulation(1)
while True:
    p.setRealTimeSimulation(0)
# while True:
    # print ("Working ")
