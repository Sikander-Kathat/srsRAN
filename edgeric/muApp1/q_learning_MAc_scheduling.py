import argparse
import sys
import pickle
import os
import torch
import numpy as np
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
from datetime import datetime
from threading import Thread
import math
import time 
import gym
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from core.q_learning_agent import QLearningAgent
from stream_rl.registry import ENVS
from hydra.utils import to_absolute_path
from edgeric_messenger import get_metrics_multi, send_scheduling_weight  # Importing send_scheduling_weight

log = logging.getLogger(__name__)

# Global variable to accumulate bitrate (in bytes) over episodes.
total_brate = []

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.bins = bins
        self.low = env.action_space.low
        self.high = env.action_space.high
        # Create a discrete action space with "bins" number of actions.
        self.action_space = gym.spaces.Discrete(bins)

    def action(self, action):
        # If action is a tensor, handle conversion appropriately.
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action = action.item()
            else:
                action = action.cpu().numpy()
        # Map the discrete action(s) to a continuous action.
        continuous_action = self.low + (action / (self.bins - 1)) * (self.high - self.low)
        return np.array(continuous_action, dtype=self.env.action_space.dtype)

    def step(self, action, *args, **kwargs):
        continuous_action = self.action(action)
        return self.env.step(continuous_action, *args, **kwargs)


def eval_loop_model(num_episodes, env_cls, env_config, conf):
    global total_brate
    log.info("Evaluating Q-learning agent")
    
    model_save_path = to_absolute_path(os.path.join(os.path.dirname(__file__), 'rl_model', 'fully_trained_model', 'q_learning_model.pth'))
    log.info(f"Loading Q-learning model from: {model_save_path}")
    
    # Update environment configuration for evaluation
    env_config.update({"seed": 9})
    env_config.update({"cqi_trace": env_config["cqi_trace_eval"]})
    env = env_cls(env_config)
    env = DiscretizeActionWrapper(env, bins=10) 
    # Get state and action dimensions (assumes discrete action space)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Instantiate Q-learning agent using hyperparameters from configuration
    q_params = conf["agent_config"]["q_learning_params"]
    agent = QLearningAgent(state_dim, action_dim, **q_params)
    agent.q_network.to(torch.device("cpu"))
    
    agent.q_network.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
    agent.q_network.eval()  # Set to evaluation mode

    # For each episode, take a single measurement rather than accumulating over multiple steps.
    for episode in range(num_episodes):
        log.info(f"Episode {episode}")
        # Get the current UE metrics once for this episode.
        ue_data = get_metrics_multi()  # Returns a dictionary of UE metrics
        numues = len(ue_data)
        txb = [data['Tx_brate'] for data in ue_data.values()]
        brate = np.sum(txb)  # Instantaneous bitrate (in bytes)
        total_brate.append(brate)
        
        # Build an observation vector from the current UE metrics.
        CQIs  = [data['CQI'] for data in ue_data.values()]
        RNTIs = list(ue_data.keys())
        BLs   = [data['Backlog'] for data in ue_data.values()]
        mbs   = np.ones(numues) * 300000  # Base station capacity per UE
        obs = np.array([param[ue] for ue in range(numues) for param in (BLs, CQIs, mbs)], dtype=np.float32)
        obs = torch.from_numpy(obs)
        obs = torch.unsqueeze(obs, dim=0)
        
        # Use the Q-learning agent to select an action.
        with torch.no_grad():
            action = agent.select_action(obs)
            if isinstance(action, int):
                action = torch.tensor([action] * numues)
            else:
                action = torch.squeeze(action)
        
        # Compute scheduling weights based on the action.
        weight = np.zeros(numues * 2)
        action_sum = action.sum().item()
        for ue in range(numues):
            percentage_RBG = (action[ue].item() / action_sum) if action_sum != 0 else 0.0
            weight[ue*2] = RNTIs[ue]
            weight[ue*2 + 1] = percentage_RBG
        send_scheduling_weight(weight, True)
        
        # Convert the instantaneous bitrate to throughput in kilobits.
        current_throughput = brate * 8 / 1000  # bytes -> kilobits
        log.info(f"Throughput: {current_throughput:.2f} kilobits")

if __name__ == "__main__":
    import hydra
    @hydra.main(config_path="conf", config_name="edge_ric_q_learning")
    def main(conf):
        global total_brate
        env_cls = ENVS[conf["env"]]
        num_eval_episodes = conf["num_eval_episodes"]
        
        # Run evaluation (total_brate will be updated as a global variable)
        eval_loop_model(num_eval_episodes, env_cls, conf["env_config"], conf)
        
        # Print overall throughput using the global total_brate.
        if total_brate:
            avg_throughput = np.mean(total_brate) * 8 / 1000  # Convert bytes to kilobits
            max_throughput = np.max(total_brate) * 8 / 1000  # bytes -> kilobits
            log.info(f"Average Throughput over {num_eval_episodes} episodes: {avg_throughput:.2f} kilobits")
            log.info(f"Max Throughput over {num_eval_episodes} episodes: {max_throughput:.2f} kilobits")
        else:
            log.info("No bitrate data collected.")
    main()
