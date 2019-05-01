# Central Pattern Generator For Quadruped Robot
## Creator - Damian Bemben
The following files implement a central pattern generator for a quadruped robot in PyBullet as well as a method for parsing and measuring quadruped data based on the Dynamic Similarity Hypothesis.

# Acknowledgments
[Original Van Der Pol Oscillator Code - Andrew Walker](http://dropbearcode.blogspot.com/2012/05/python-simulation-of-van-der-pol.html)

[PyBullet - Erwin Coumans](http://pybullet.org)
## Requirements
In order to run this code, the following libraries are required. The libraries can be installed by running the following command:

```
pip install -r pipmodules.txt
```
[pybullet](https://pypi.org/project/pybullet/)
[numpy](https://pypi.org/project/numpy/)
[matplotlib](https://pypi.org/project/matplotlib/)
[scipy](https://pypi.org/project/scipy/)
[sklearn](https://pypi.org/project/sklearn/)
[pandas](https://pypi.org/project/pandas/)

### Example Run
In order to run an example of the Laikago in a graphic environment with a stable gait, run the following command.
```
laikago_GUI.py 40 0.006 0 x 10 10
```
## laikago.py
Takes in 6 parameters for running, outputs a file containing mean and standard deviation values for experiments:
1. Force - (20 - 100)
2. Oscillator Timesteps (0.01 - 0.002)
3. Gait - (0 - Walking, 1- Trotting, 2- Bounding)
4. Folder to save to - (Any String ("all" = default))
5. Foot Angle - (5-20)
6. Hip Angle - (5-20)
Runs for 10000 iterations with 10 repetitions, prints out results in python file, with the following format:
```
Oscillator Step: ()
Max Force: ()
Gait: ()
Leg Rotation: ()
Hip Rotation: ()
Velocity Mean: Velocity SD
Froude Mean: Froude SD
Distance Mean: Distance SD
Cost Of Locomotion Mean: Cost of Locomotion SD
Time Period Mean: Time Period SD
```
An example of parsing can be found in plot_main_results.py -> parse_all()
### Variations on laikago.py
Although laikago.py runs the majority of experiments, laikago_structural.py can be used to find successful gaits. This uses the same parameters, but only runs for 10000 iterations, and only outputs the following format:
```
(Distance)
(Froude Number)
```
ttest.py runs for 10000 iterations and produces a paired t-test of values.
## Experiments
Experiments can be seen by running plot_main_results.py. This parses a folder and returns all experiments. Current set of experiments are the following:

print_percentages() -  Prints percentage of values below froude number values of 0.4
oscillator_vs_froude_bar() - Prints a bar chart of average oscillator time-step vs froude number
plot_3d() - plots a 3d graph of all combinations of values
plot_distribution() - plots distributions found in dissertation
plot_froude() - plots froude number graph found in dissertation

##  Misc
### van.py
produces a single uncoupled van der pol oscillator
### van_coupled_rt.py
produces a real-time graph of coupled van der pol oscillators
<!-- ## file_parser.py
Produces graphs from running results -->

<!-- ## experiment_batch.sh
Produces experiment batch used in final dissertation

## tandptests.sh
Produces experiments -->
