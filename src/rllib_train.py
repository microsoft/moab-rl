"""
Main script for training an agent with RLlib.
"""
import sys
import os

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from sim.sim_env import MoabSim as SimEnv

# Register the simulation as an RLlib environment.
register_env("sim_env", lambda config: SimEnv(config))


def train(local=True):
    # Define the algo for training the agent
    algo = (
        PPOConfig()
        # Modify also instance_count in job definition to use more than one worker
        # Setting workers to zero allows using breakpoints in sim for debugging
        .rollouts(num_rollout_workers=1 if not local else 0)
        .resources(num_gpus=0)
        # Set the training batch size to the appropriate number of steps
        .training(train_batch_size=4_000)
        .environment(env="sim_env")
        .build()
    )
    # Train for 100 iterations
    for i in range(100):
        result = algo.train()
        print(pretty_print(result))

    output_path = os.path.join(os.getcwd(), "outputs/")
    algo.get_policy().export_model(output_path, onnx=9)
    os.rename(output_path + "model.onnx", output_path + "rllib_model.onnx")
    print(f"Checkpoint saved in directory {output_path}")


if __name__ == "__main__":

    train()
    sys.exit()
