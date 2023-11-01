"""
Main script for training an agent with Stable-Baselines3.
"""
import sys
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sim.sim_env import MoabSim as SimEnv


def train():
    # Create a wrapped, monitored VecEnv
    vec_env = make_vec_env(SimEnv, env_kwargs={"env_config": None}, n_envs=4, seed=1)

    # Create and train the agent
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=1000000)

    # Export to ONNX
    # https://stable-baselines3.readthedocs.io/en/master/guide/export.html#export-to-onnx
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net

        def forward(self, observation):
            # NOTE: You may have to process (normalize) observation in the correct
            #       way before using this. See `common.preprocessing.preprocess_obs`
            action_hidden, value_hidden = self.extractor(observation)
            return self.action_net(action_hidden), self.value_net(value_hidden)

    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
    )

    observation_size = model.observation_space.shape
    dummy_input = torch.randn(1, *observation_size)

    output_folder = "outputs"
    if output_folder not in os.listdir():
        os.mkdir(output_folder)
    # output_path = os.path.join(os.getcwd(), "outputs")
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        output_folder + "/sb3_model.onnx",
        opset_version=9,
        input_names=["input"],
    )
    print(f"Model saved in directory {output_folder}")


if __name__ == "__main__":

    train()
    sys.exit()
