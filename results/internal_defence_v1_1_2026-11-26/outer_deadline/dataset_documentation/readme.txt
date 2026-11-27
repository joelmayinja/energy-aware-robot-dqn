
The files in this folder are related to the paper:
"Extensive Experimental Investigation for the Optimization of the Energy Consumption of a High Payload Industrial Robot with Open Research Dataset".

Look at the paper for more details.

If you found any problem, please communicate it at:
michele.gadaleta@unimore.it

The Authors
Michele Gadaleta, Giovanni Berselli, Marcello Pellicciari, Federico Grassia





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Shared data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The folder "data" contains all the experimental data subdivided into three files, one for each experiment type (extract the files before use).
The MATLAB script "graphs.m" generates all the graphs in the paper.
The MATLAB script "maps.m" generates all the maps in the paper.


DATA                                            SIZE        MEASUREMENT UNIT
                                                
exp.velocity_parameter                          [1,1]       [%] for PtP motions   [m/s] for Linear motions
exp.acceleration_parameter                      [1,1]       [%] for PtP motions   [m/s^2] for Linear motions
exp.oil_temperature                             [1,1]       [°C]

exp.electrical_measurements.time                [K,1]       [s] 
exp.electrical_measurements.total_power         [K,1]       [W]
exp.electrical_measurements.total_energy        [K,1]       [J]

exp.kuka_trace_data.time                        [J,1]       [s]
exp.kuka_trace_data.total_power                 [J,1]       [W]
exp.kuka_trace_data.total_energy                [J,1]       [J]
exp.kuka_trace_data.mechanical_power            [J,6]       [W]
exp.kuka_trace_data.mechanical_energy           [J,6]       [J]
exp.kuka_trace_data.motor_position              [J,6]       [°]
exp.kuka_trace_data.motor_torque                [J,6]       [Nm]
exp.kuka_trace_data.motor_current               [J,6]       [A]
exp.kuka_trace_data.motor_temperature           [J,6]       [°C]
exp.kuka_trace_data.joint_position              [J,6]       [°]
exp.kuka_trace_data.joint_velocity              [J,6]       [°/s]
exp.kuka_trace_data.joint_acceleration          [J,6]       [°/s^2]
exp.kuka_trace_data.joint_torque                [J,6]       [Nm]
exp.kuka_trace_data.target_motor_position       [J,6]       [°]
exp.kuka_trace_data.target_joint_position       [J,6]       [°]
exp.kuka_trace_data.time_motion_start           [1,1]       [s]
exp.kuka_trace_data.time_motion_end             [1,1]       [s]


NOTE: 
Data from direct electrical measurement and data extracted from KUKA trace are not synchronized.
To properly align the data, the time vectors of the direct electrical measurements have been shifted preserving the whole data set (hence, negative time values are present).





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Experiments moving axis 1 with increasing temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_point = { A1:-180, A2:-90, A3:90, A4:0, A5:90, A6:0 }
end_point   = { A1: 180, A2:-90, A3:90, A4:0, A5:90, A6:0 }

motion1  -->  motion from start_point to end_point.
motion2  -->  motion from end_point to start_point.


EXPERIMENT SELECTION:

iM = 1..2    -->  [ motion1, motion2 ]
iL = 1..3    -->  [ load1, load2, load3 ]
iR = 1..200

exp = exp_1axis_increasing_temperature.motion{iM}.load{iL}.repetition{iR}





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Experiments moving axis 1 with constant temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_point = { A1:-180, A2:-90, A3:90, A4:0, A5:90, A6:0 }
end_point   = { A1: 180, A2:-90, A3:90, A4:0, A5:90, A6:0 }

motion1  -->  motion from start_point to end_point.
motion2  -->  motion from end_point to start_point.


EXPERIMENT SELECTION:

iM = 1..2  -->  [ motion1, motion2 ]
iL = 1..3  -->  [ load1, load2, load3 ]
iV = 1..8  -->  [ 1, 2, 5, 10, 25, 50, 75, 100 ] [%]
iA = 1..8  -->  [ 1, 2, 5, 10, 25, 50, 75, 100 ] [%]

exp = exp_1axis_constant_temperature.motion{iM}.load{iL}.velocity{iV}.acceleration{iA}





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Experiments moving 6 axes with constant temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_point = { X:1860.0, Y:1620.0, Z:1520.0, A: 0.0, B:90.0, C:0.0, S:22, T:27 }
end_point   = { X:-745.0, Y:1300.0, Z: 215.0, A:90.0, B:90.0, C:0.0, S:22, T:27 }

motion1  -->  motion from start_point to end_point.
motion2  -->  motion from end_point to start_point.


EXPERIMENT SELECTION:

iM = 1..2   -->  [ motion1, motion2 ]
iT = 1..2   -->  [ PtP, Linear ]
iL = 1..3   -->  [ load1, load2, load3 ]
iV = 1..10  -->  for PtP motions    [   10,   20,   30,   40,   50,   60,   70,   80,   90,  100 ] [%]
            -->  for Linear motions [ 0.17, 0.34, 0.51, 0.68, 0.85, 1.02, 1.19, 1.36, 1.53, 1.70 ] [m/s]
iA = 1..10  -->  for PtP motions    [   10,   20,   30,   40,   50,   60,   70,   80,   90,  100 ] [%]
            -->  for Linear motions [    1,    2,    3,    4,    5,    6,    7,    8,    9,   10 ] [m/s^2]
iB = 1..2   -->  [ 1, 20 ] [s]

exp = exp_6axes_constant_temperature.motion{iM}.motion_type{iT}.load{iL}.velocity{iV}.acceleration{iA}.brake_delay{iB}




