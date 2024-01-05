# Reinforcement Learning with Moab
## _Train an Agent to Balance Objects on a Platform_

Welcome to Moab: a hardware kit that helps you learn about autonomous systems in a fun and interactive way. In this repository, we'll help you get started by providing the basic code to train and deploy your own reinforcement learning (RL) agent. Your agent will learn to balance objects on the Moab device using sensors and actuators.
 
Our code features a Python Gymnasium simulation environment for the Moab bot, along with Python scripts compatible with two popular RL frameworks: Stable Baselines3 and RLlib. If you’re curious about how well your trained agent performs, we’ve got you covered! Our repository also contains Jupyter notebooks for assessing your agent in simulation, as well as step-by-step instructions on deploying it to the Moab hardware.

## Getting Started

To get started with this project, you need to have the following requirements:

- Python 3.7 or higher (we recommend using [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html#anaconda-or-miniconda))
- pip3 or conda
- A Moab hardware kit (optional, but recommended)

You can either download this repo or clone it with the following command in your terminal:

```
git clone https://github.com/microsoft/moab-rl.git
```

Then, install the dependencies using either pip3 or conda (we recommend using a virtual or conda environment):

```
conda env create -f environment.yml
```

or

```
pip3 install -r requirements.txt
```


## Project Structure

The project is organized into four main components:

- __Sim__: This folder contains the code for the simulation environment, which consists of a a physics-based simulation (`src/sim/model.py`) and a gymnasium.Env class that wraps the simulation for RL (`src/sim/sim_env.py`). This folder also contains a README file with more details about the simulation environment and how you can change it to improve generalization.

- __Train__: Python scripts for two popular reinforcement learning frameworks, stable baselines and RLlib, are provided:

    - `src/train_stablebaselines3.py`: Stable Baselines3 is a library for single-machine RL that supports several policy gradient and actor-critic methods. Its user-friendly design makes it easy to get started quickly.
    - `src/train_rllib.py`: Ray RLlib is a library for single or distributed RL that supports a wider range of algorithms and customizations. [Microsoft’s Plato Toolkit](https://github.com/Azure/plato) provides several examples for distributed RL on Azure clusters.

    These scripts contain the foundational code for you to build on for experimentation. Both scripts will save your trained agent as an ONNX file, which is a standard format for exchanging models across different platforms and frameworks. However, the RLlib ONNX policy will return logits instead of actions. An extra step is required to convert the logits to actions.

- __Assess__: Once you have an ONNX policy saved, assessment can be done in both simulation and on the Moab hardware. We have provided two Jupyter notebooks for assessment with the simulation environment (`src/sb3_assess.ipynb` and `src/rllib_assess.ipynb`). The notebooks include code for loading your ONNX policy and running it on the simulation environment.

- __Deploy__: Please see the section ["Deploy Agent on Moab Hardware"](#deploy-agent-on-moab-hardware) for detailed instructions on how to deploy your ONNX policy on a physical Moab device.

## Train your Agent

1. Choose your RL framework (Stable Baselines3 or RLlib) and open the corresponding Python script in the `src` folder.
2. (Optional) Adjust the hyperparameters for the training process, such as the number training iterations and the algorithm to use.
3. Run the script, then wait for training to finish. The script will save the trained ONNX policy file in the `src/outputs` folder.

```
python src/train_stablebaselines3.py
```

4. You can evaluate the policy using the simulation environment in the  Jupyter notebook provided, or deploy to the hardware itself (instructions are available in the next section).


(Note) If you use RLlib, the saved ONNX policy will return logits instead of actions. You need to apply a softmax function to the logits to get the probabilities of each action, and then sample an action from the distribution. The function `logits_to_actions()` is available in `rllib_assess.ipynb`.

## Deploy Agent on Moab Hardware

1. __Connect to Moab__

    Follow the [connection instructions and SSH into your Moab](https://github.com/microsoft/moab-rl/blob/main/docs/connecting.md).

2. __(Optional) Calibrate Moab__

    To get the best performance out of your Moab, you may need to calibrate it. Calibration adjusts Moab’s settings to match your ball color, position, and plate level. This improves Moab’s ability to detect and balance the ball accurately. Our [calibration guide](https://github.com/microsoft/moab-rl/blob/main/docs/calibration.md) will take you through the steps to adjust your Moab’s settings for ball hue, ball position, and servo offset.


3.	__Install `onnxruntime`__

    Your Moab is equipped with a Raspberry Pi 4 and the armv7l architecture. This architecture requires a specific wheel package for `onnxruntime` that is not available on PyPI and, therefore, cannot be installed directly with pip3.

    Instead, please download the correct wheel for your platform from the [built-onnxruntime-for-raspberrypi-linux repository](https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/tree/master). For example, if your Moab device has Debian version 10.4 and Python version 3.7, you might select: _/wheels/buster/onnxruntime-1.8.1-cp37-cp37m-linux_armv7l.whl_

    You can add the wheel to your Moab device by copy-pasting it, or with secure copy:
    ```
    scp <filename>.whl pi@moab:~/moab
    ```
    Then install it in your Moab's terminal with:
    ```
    pip3 install <filename>.whl
    ```

4.	__Add your ONNX policy file to Moab__

    You can copy-paste or use secure copy to add your ONNX model file to the Moab. For example:

    ```
    scp sb3_model.onnx pi@moab:~/moab
    ```

5.	__Modify the software controller (`/moab/sw/controllers.py`)__

    Add one or both of these functions based on how you generated your models.

    **RLlib:**
    ```python
    def rllib_onnx_controller(max_angle=22, **kwargs,):
        import onnxruntime
        import math
        session = onnxruntime.InferenceSession("/home/pi/moab/sw/rllib_model.onnx")

        # Define a function for converting logits to actions with a SquashedGaussian distribution for continuous action spaces.
        def logits_to_actions(logits):
            # Define low and high values
            low = np.array([-1, -1], dtype=np.float32)
            high = np.array([1, 1], dtype=np.float32)
            # Define a constant for the minimum log value to avoid numerical issues
            MIN_LOG_VALUE = -1e7
            # Split the logits into mean and log_std
            split_index = len(logits) // 2
            means, log_stds = logits[:split_index], logits[split_index:]
            actions = []
            i = 0
            for mean, log_std in zip(means, log_stds):
                # Clip the log_std to a reasonable range
                log_std = max(min(log_std, -MIN_LOG_VALUE), MIN_LOG_VALUE)
                # Compute the std from the log_std
                std = math.exp(log_std)
                # Create a normal distribution with the mean and std
                normal_dist = [mean, std]
                # Sample a value from the normal distribution
                normal_sample = mean # use the mean for deterministic output
                # Apply a tanh function to the normal sample
                tanh_sample = math.tanh(normal_sample)
                # Scale the tanh sample by the low and high bounds of the action space
                action = low[i] + (high[i] - low[i]) * (tanh_sample + 1) / 2
                actions.append(action)
                i += 1
            # Return the list of actions
            return actions

        def next_action(state):
            env_state, ball_detected, buttons = state
            x, y, vel_x, vel_y, sum_x, sum_y = env_state

            if ball_detected:
                obs = np.array([x, y, vel_x , vel_y])
                logits = session.run(None, {"obs": [obs], "state_ins": [None]})[0][0]
                actions = logits_to_actions(logits)
                pitch = actions[0]
                roll = actions[1]
                # Scale, clip and convert to integer
                pitch = int(np.clip(pitch * max_angle, -max_angle, max_angle))
                roll = int(np.clip(roll * max_angle, -max_angle, max_angle))
                action = Vector2(-pitch, roll)
            else:
                # Move plate back to flat
                action = Vector2(0, 0)
            return action, {}

        return next_action
    ```

    **Stable Baselines3:**
    ```python
    def sb3_onnx_controller(max_angle=22, **kwargs,):
        import onnxruntime
        session = onnxruntime.InferenceSession("/home/pi/moab/sw/sb3_model.onnx")
        def next_action(state):
            env_state, ball_detected, buttons = state
            x, y, vel_x, vel_y, sum_x, sum_y = env_state

            if ball_detected:
                obs = np.array([x, y, vel_x , vel_y])
                actions = session.run(None, {"input": [obs]})[0][0]
                pitch = actions[0]
                roll = actions[1]
                # Scale, clip and convert to integer
                pitch = int(np.clip(pitch * max_angle, -max_angle, max_angle))
                roll = int(np.clip(roll * max_angle, -max_angle, max_angle))
                action = Vector2(-pitch, roll)
            else:
                # Move plate back to flat
                action = Vector2(0, 0)
            return action, {}

        return next_action
    ```

6.	__Modify the Moab menu (`/moab/sw/menu.py`)__

    Import the function(s) created in Step 5.

    In the build_menu function, add one or both of the menu options below to the *top_menu* list:
    ```python
    MenuOption(
        name="RLlib",
        closure=rllib_onnx_controller,
        kwargs={},
        decorators=[log_csv] if log_on else None
    ),
    ```

    ```
    MenuOption(
        name="SB3",
        closure=sb3_onnx_controller,
        kwargs={},
        decorators=[log_csv] if log_on else None
    ),
    ```

7. __Quickly tap the power button to reset Moab__

8. __Select and test your policy__



## License

This project is licensed under the MIT License - see the [LICENSE](^1^) file for details.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Precommit Hooks
To get a smoother developer experience, please install pre-commit hooks in your environment:

```
pip install pre-commit
pre-commit install
```

Now every time you try to commit, the code is linted and issues fixed automatically when possible, otherwise the offending lines are shown on the screen.
