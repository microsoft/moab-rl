# Moab Simulation Environment
The Moab simulation model tracks the position of the ball on an `(x,y)` plane, the velocity of the ball, and the changing angles of the plate over time.

## Observable State

On every training iteration, the agent receives an observable state from the simulator with the following state information:

- `ball_x`: current position of ball on the x-axis in meters
- `ball_y`: current position of ball on the y-axis in meters
- `ball_vel_x`: current velocity of the ball on the x-axis in meters/sec
- `ball_vel_y`: current velocity of the ball on the y-axis in meters/sec

## Action

The agent can alter the angle of the plate to alter the velocity of the ball:

- `input_pitch`: new angle (radians) for the plate when rotated on the x-axis. A negative value indicates "forward" and a positive value indicates "back".
- `input_roll`: new angle (radians) for the plate when rotated on the y-axis. A negative value indicates "left" and a positive value indicates "right".

## Configuration
To make your agent more robust and adaptable, you can randomize the initial conditions of each episode using the `config` dictionary in the `reset()` method. This dictionary contains the following parameters that you can adjust:

- `ball_radius`: the radius of the ball in meters
- `ball_shell`: the thickness of the ball shell in meters
- `ball.x`: the initial position of the ball on the x-axis in meters
- `ball.y`: the initial position of the ball on the y-axis in meters
- `ball_vel.x`: the initial velocity of the ball on the x-axis in meters/sec
- `ball_vel.y`: the initial velocity of the ball on the y-axis in meters/sec
- `roll`: the initial angle of the plate when rotated on the y-axis in radians
- `pitch`: the initial angle of the plate when rotated on the x-axis in radians

The `config` dictionary is defined in the `MoabSim` class in `sim_env.py`, which is the main class that handles the simulation environment for RL.

## Reward
The `reward()` function is defined in the `MoabSim` class in `sim_env.py`. The current reward function is:

$$
r = 1 - \sqrt{(ball\_x - x_0)^2 + (ball\_y - y_0)^2} / plate\_radius
$$


where (x, y) is the position of the ball and (x_0​, y_0​) is the center of the plate.

We encourage you to experiment with the reward function and observe how it affects the agent’s learning process.
