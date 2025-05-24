import argparse
import sys
import os
import torch
import numpy as np
import logging
import gym
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from edgeric_messenger import get_metrics_multi, send_scheduling_weight
from stream_rl.registry import ENVS

log = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.bins = bins
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = gym.spaces.Discrete(bins)

    def action(self, action):
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action = action.item()
            else:
                action = action.cpu().numpy()
        continuous_action = self.low + (action / (self.bins - 1)) * (self.high - self.low)
        return np.array(continuous_action, dtype=self.env.action_space.dtype)

    def step(self, action, *args, **kwargs):
        continuous_action = self.action(action)
        return self.env.step(continuous_action, *args, **kwargs)

def eval_loop_model(num_episodes, env_cls, env_config, conf):
    log.info("Evaluating Actor-Critic agent")

    # Construct the path to the saved checkpoint.
    model_save_path = to_absolute_path(os.path.join(os.path.dirname(__file__), 'actor critic', 'edge_ric_sac_model.pth'))
    log.info(f"Loading Actor-Critic model from: {model_save_path}")

    # Update environment configuration for evaluation.
    env_config.update({"seed": 9})
    env_config.update({"cqi_trace": env_config["cqi_trace_eval"]})
    env = env_cls(env_config)
    env = DiscretizeActionWrapper(env, bins=10)
    
    # Force the model's architecture to match the checkpoint.
    state_dim = 12   # Force input size to 12.
    action_dim = 2   # Force output (action) size to 2.
    
    # Create an instance of the actor network with the correct dimensions.
    hidden_dim = conf["agent_config"]["sac_params"]["hidden_dim"]
    actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
    actor.to(torch.device("cpu"))
    
    checkpoint = torch.load(model_save_path, map_location=torch.device("cpu"))
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    for episode in range(num_episodes):
        log.info(f"Episode {episode}")
        ue_data = get_metrics_multi()  # Get UE metrics (a dictionary)
        numues = len(ue_data)
        txb = [data['Tx_brate'] for data in ue_data.values()]
        brate = np.sum(txb)
        
        # Build an observation vector from UE metrics.
        # For each UE, we use BL, CQI, and mbs. With 2 UEs this gives 6 features.
        CQIs  = [data['CQI'] for data in ue_data.values()]
        RNTIs = list(ue_data.keys())
        BLs   = [data['Backlog'] for data in ue_data.values()]
        mbs   = np.ones(numues) * 300000  # Base station capacity per UE.
        obs = np.array([param[ue] for ue in range(numues) for param in (BLs, CQIs, mbs)], dtype=np.float32)
        # If observation is 6-dimensional, duplicate it to get 12.
        if obs.shape[0] == 6:
            obs = np.concatenate([obs, obs])
        obs = torch.from_numpy(obs).unsqueeze(0)  # Shape: [1, 12]

        # Use the actor to select an action.
        with torch.no_grad():
            logits = actor(obs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        # Ensure 'action' is a tensor with one entry per UE.
        if not torch.is_tensor(action):
            action = torch.tensor([action] * numues)
        elif action.dim() == 0:
            action = torch.tensor([action.item()] * numues)
        elif action.dim() == 1 and action.size(0) != numues:
            action = action.repeat(numues)

        # Calculate scheduling weights based on the action.
        weight = np.zeros(numues * 2)
        action_sum = action.sum().item()
        for ue in range(numues):
            percentage_RBG = (action[ue].item() / action_sum) if action_sum != 0 else 0.0
            weight[ue*2] = RNTIs[ue]
            weight[ue*2 + 1] = percentage_RBG
        send_scheduling_weight(weight, True)
        
        current_throughput = brate * 8 / 1000  # Convert bytes to kilobits.
        log.info(f"Throughput: {current_throughput:.2f} kilobits")

if __name__ == "__main__":
    import hydra
    @hydra.main(config_path="conf", config_name="edge_ric_sac")
    def main(conf):
        env_cls = ENVS[conf["env"]]
        num_eval_episodes = conf["num_eval_episodes"]
        eval_loop_model(num_eval_episodes, env_cls, conf["env_config"], conf)
    main()