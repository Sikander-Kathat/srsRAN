import argparse
import hydra
import gym
import os
import sys
import pickle
import time
import logging
import torch
import numpy as np

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edgeric_messenger import *
torch.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent_original import Agent
from core.q_learning_agent import QLearningAgent

from stream_rl.registry import ENVS
from stream_rl.plots import (
    visualize_edgeric_training,
    visualize_edgeric_evaluation,
    plot_cdf,
    visualize_policy_cqi,
    visualize_policy_backlog_len,
)
from hydra.core.hydra_config import HydraConfig
# Setup logger
log = logging.getLogger(__name__)

# Parse command-line arguments (only keeping those relevant for Q-learning)
parser = argparse.ArgumentParser(description="PyTorch Q-learning example")
parser.add_argument("--env-name", default="Hopper-v2", metavar="G", help="Name of the environment to run")
parser.add_argument("--model-path", metavar="G", help="Path of a pre-trained model (optional)")
parser.add_argument("--render", action="store_true", default=False, help="Render the environment")
parser.add_argument("--gamma", type=float, default=0.9, metavar="G", help="Discount factor")
parser.add_argument("--learning-rate", type=float, default=3e-3, metavar="G", help="Learning rate (overridden by config)")
parser.add_argument("--num-threads", type=int, default=1, metavar="N", help="Number of threads for agent (default: 1)")
parser.add_argument("--seed", type=int, default=1, metavar="N", help="Random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1, metavar="N", help="Interval between training logs")
parser.add_argument("--gpu-index", type=int, default=0, metavar="N", help="GPU index")
args = parser.parse_args()

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.bins = bins
        self.low = env.action_space.low
        self.high = env.action_space.high
        # Define a discrete action space with 'bins' number of actions
        self.action_space = gym.spaces.Discrete(bins)

    def action(self, action):
        # Map the discrete action to a continuous action
        continuous_action = self.low + (action / (self.bins - 1)) * (self.high - self.low)
        return np.array(continuous_action, dtype=self.env.action_space.dtype)

    def step(self, action, *args, **kwargs):
        # Forward any additional arguments to the underlying environment's step method
        continuous_action = self.action(action)
        return self.env.step(continuous_action, *args, **kwargs)

# Use Hydra to load the Q-learning configuration file
@hydra.main(config_path="conf", config_name="edge_ric_q_learning")
def main(conf):
    # Set default tensor type and device
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    # Create the environment from the registry using the configuration
    env_cls = ENVS[conf["env"]]
    env = env_cls(conf["env_config"])
    env = DiscretizeActionWrapper(env, bins=10)  # Adjust 'bins' as needed
    action_dim = env.action_space.n  # Now this will correctly return a discrete action count


    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if hasattr(env, 'seed'):
    #env.seed(args.seed)

    # Determine state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # Q-learning requires a discrete action space

    # Instantiate the Q-learning agent using parameters from the configuration
    q_params = conf["agent_config"]["q_learning_params"]
    agent = QLearningAgent(state_dim, action_dim, **q_params)
    agent.q_network.to(device)

    # Optionally load a pre-trained Q-network if provided
    if args.model_path is not None:
        agent.q_network.load_state_dict(torch.load(args.model_path, map_location=device))
        log.info(f"Loaded pre-trained model from {args.model_path}")

    # Retrieve training hyperparameters from config
    num_iters = conf["num_iters"]
    max_steps_per_episode = conf["training_config"]["max_steps_per_episode"]
    log_interval = conf["training_config"]["log_interval"]

    # Main Q-learning training loop
        # Main Q-learning training loop
    all_rewards = []
    for iter_num in range(num_iters):
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done and step < max_steps_per_episode:
            if args.render:
                env.render()

            # Select an action using epsilon-greedy policy
            action = agent.select_action(state)

            # Get metrics from EDGERIC (similar to the PPO eval loop)
            ue_data = get_metrics_multi()         # Returns a dictionary of UE metrics
            numues = len(ue_data)
            env.num_UEs = numues  # Update environment with number of UEs if needed

            # Extract the required values as in the PPO code
            CQIs   = [data['CQI'] for data in ue_data.values()]
            RNTIs  = list(ue_data.keys())
            BLs    = [data['Backlog'] for data in ue_data.values()]
            mbs    = np.ones(numues) * 300000      # e.g., base station capacity per UE
            txb    = [data['Tx_brate'] for data in ue_data.values()]
            tx_bytes = np.sum(txb)

            # Take a step in the environment (the wrapper forwards the extra arguments)
            next_state, reward, done, info = env.step(action, RNTIs, CQIs, BLs, tx_bytes, mbs)

            # Update the Q-network with the transition
            loss = agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1

        # Force epsilon decay at the end of the episode,
        # since the environment might not signal done.
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        all_rewards.append(episode_reward)
        if iter_num % log_interval == 0:
            log.info(f"Iteration: {iter_num}\tEpisode Reward: {episode_reward:.2f}\tEpsilon: {agent.epsilon:.3f}")


    # Save the trained Q-network model if saving is enabled
    if conf["training_config"]["save_model_interval"] > 0:
        hydra_cfg = HydraConfig.get()
        # Get the Hydra output directory (runtime.output_dir if available, otherwise run.dir)
        output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
        # Construct the full model save path
        model_save_path = os.path.join(output_dir, conf["training_config"]["model_save_path"])
        # Create the parent directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        # Save the Q-learning model's state dictionary
        torch.save(agent.q_network.state_dict(), model_save_path)
        log.info(f"Q-learning model saved to {model_save_path}")
        # Optionally visualize the training rewards
    
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    visualize_edgeric_training([all_rewards])

if __name__ == "__main__":
    main()

