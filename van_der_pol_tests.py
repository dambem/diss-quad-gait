# Oscillator Values, Initiated at 1
start_y_foot = [1,1,1,1]
start_x_foot = [0,0,0,0]
new_y_foot = [1,1,1,1]
new_x_foot = [0,0,0,0]

start_y_hip = [1,1,1,1]
start_x_hip = [0,0,0,0]
new_y_hip = [1,1,1,1]
new_x_hip = [0,0,0,0]
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
for t in range(100):
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
