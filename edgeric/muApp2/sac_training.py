import argparse
import hydra
import gym
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edgeric_messenger import get_metrics_multi, send_scheduling_weight
torch.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream_rl.registry import ENVS
from stream_rl.plots import visualize_edgeric_training
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Soft Actor-Critic for Scheduling")
parser.add_argument("--env-name", default="Hopper-v2", metavar="G", help="Name of the environment")
parser.add_argument("--model-path", metavar="G", help="Path of a pre-trained model (optional)")
parser.add_argument("--render", action="store_true", default=False, help="Render the environment")
parser.add_argument("--num-threads", type=int, default=1, metavar="N", help="Number of threads")
parser.add_argument("--seed", type=int, default=1, metavar="N", help="Random seed")
parser.add_argument("--log-interval", type=int, default=1, metavar="N", help="Log interval")
parser.add_argument("--gpu-index", type=int, default=0, metavar="N", help="GPU index")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Discretize Action Wrapper (kept from your original code)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Define Networks for Discrete SAC
# ---------------------------------------------------------------------
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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# ---------------------------------------------------------------------
# Simple Replay Buffer for Off-Policy Learning
# ---------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, s_next, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------
# Discrete Soft Actor-Critic Agent
# ---------------------------------------------------------------------
class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, alpha, device):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        if eval:
            action = torch.argmax(probs, dim=-1)
        else:
            m = Categorical(probs)
            action = m.sample()
        return action.item()

    def update_parameters(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_logits = self.policy(next_state)
            next_probs = torch.softmax(next_logits, dim=-1)
            log_next_probs = torch.log(next_probs + 1e-10)
            target_q1 = self.target_q1(next_state)
            target_q2 = self.target_q2(next_state)
            target_q = torch.min(target_q1, target_q2)
            # Compute V(s') = Σ_a π(a|s') [ Q(s',a) - α log π(a|s') ]
            target_v = (next_probs * (target_q - self.alpha * log_next_probs)).sum(dim=1, keepdim=True)
            target_q_value = reward + (1 - done) * self.gamma * target_v

        current_q1 = self.q1(state).gather(1, action)
        current_q2 = self.q2(state).gather(1, action)
        q1_loss = F.mse_loss(current_q1, target_q_value)
        q2_loss = F.mse_loss(current_q2, target_q_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy update
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        q1_pi = self.q1(state)
        q2_pi = self.q2(state)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update targets
        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)

        return q1_loss.item() + q2_loss.item() + policy_loss.item()

# ---------------------------------------------------------------------
# Main Training Loop (Hydra entry point)
# ---------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="edge_ric_sac")
def main(conf):
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    env_cls = ENVS[conf["env"]]
    env = env_cls(conf["env_config"])
    # Wrap the environment to discretize actions.
    env = DiscretizeActionWrapper(env, bins=10)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine number of UEs (User Equipments) from initial metrics.
    initial_ue_data = get_metrics_multi()
    num_ues = len(initial_ue_data)
    log.info(f"Determined number of UEs: {num_ues}")
    state_dim = env.observation_space.shape[0]
    # For discrete SAC, we force action_dim = 2 (as in your original code).
    action_dim = 2
    conf["env_config"]["num_UEs"] = num_ues

    sac_params = conf["agent_config"]["sac_params"]
    training_conf = conf["training_config"]

    agent = DiscreteSACAgent(
        state_dim,
        action_dim,
        hidden_dim=sac_params["hidden_dim"],
        actor_lr=sac_params["actor_lr"],
        critic_lr=sac_params["critic_lr"],
        gamma=sac_params["gamma"],
        tau=sac_params["tau"],
        alpha=sac_params["alpha"],
        device=device
    )

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location=device)
        agent.policy.load_state_dict(checkpoint["actor_state_dict"])
        agent.q1.load_state_dict(checkpoint["q1_state_dict"])
        agent.q2.load_state_dict(checkpoint["q2_state_dict"])
        log.info(f"Loaded pre-trained model from {args.model_path}")

    policy_scheduler = StepLR(agent.policy_optimizer, step_size=100, gamma=0.99)
    q1_scheduler = StepLR(agent.q1_optimizer, step_size=100, gamma=0.99)
    q2_scheduler = StepLR(agent.q2_optimizer, step_size=100, gamma=0.99)

    num_iters = conf["num_iters"]
    max_steps_per_episode = training_conf["max_steps_per_episode"]
    update_interval = training_conf["update_interval"]
    reward_scale = training_conf.get("reward_scale", 1.0)
    batch_size = training_conf.get("batch_size", 256)
    replay_buffer_capacity = training_conf.get("replay_buffer_capacity", 1000000)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    episode_count = 0
    batch_losses = []
    all_episode_rewards = []
    moving_avg_window = 10

    for iter_num in range(num_iters):
        state = env.reset()
        # If state shape is lower-dimensional, duplicate it (as in your original code).
        if state.shape[0] == 6:
            state = np.concatenate([state, state])
        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            if args.render:
                env.render()
            # Select action using SAC (discrete version).
            action = agent.select_action(state)
            # Retrieve UE data and compute the throughput reward.
            ue_data = get_metrics_multi()
            num_ues_actual = len(ue_data)
            env.num_UEs = num_ues_actual
            CQIs  = [data['CQI'] for data in ue_data.values()]
            cqi_map = conf["env_config"]["cqi_map"]
            optimal_weights = np.array([float(cqi_map[int(cqi)][0]) for cqi in CQIs])
            computed_reward = optimal_weights.sum()
            reward = computed_reward
            scaled_reward = reward * reward_scale

            # Build weight vector (as in your original code) and send scheduling weight.
            RNTIs = list(ue_data.keys())
            weight = np.zeros(num_ues_actual * 2)
            for ue in range(num_ues_actual):
                weight[ue*2] = RNTIs[ue]
                # Here we set a dummy weight for the second entry (since agent probabilities aren’t used).
                weight[ue*2 + 1] = 0.0
            send_scheduling_weight(weight, True)

            # Take a step in the environment.
            next_state, _, done, info = env.step(
                action,
                RNTIs,
                CQIs,
                [data['Backlog'] for data in ue_data.values()],
                np.sum([data['Tx_brate'] for data in ue_data.values()]),
                np.ones(num_ues_actual) * 300000
            )
            if next_state.shape[0] == 6:
                next_state = np.concatenate([next_state, next_state])
            # Store transition in replay buffer.
            replay_buffer.push(state, action, scaled_reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1

            if len(replay_buffer) > batch_size:
                loss = agent.update_parameters(replay_buffer, batch_size)
                batch_losses.append(loss)

        episode_count += 1
        all_episode_rewards.append(episode_reward)
        moving_avg = np.mean(all_episode_rewards[-moving_avg_window:]) if len(all_episode_rewards) >= moving_avg_window else np.mean(all_episode_rewards)
        # Print throughput (episode reward) every 5 iterations
        if iter_num % 5 == 0:
            log.info(f"Iteration: {iter_num} - Throughput: {episode_reward:.2f}")
            
        log.info(f"Iteration: {iter_num}\tEpisode Reward: {episode_reward:.2f}\tMoving Avg Reward: {moving_avg:.2f}")

        if episode_count % update_interval == 0:
            policy_scheduler.step()
            q1_scheduler.step()
            q2_scheduler.step()

    if training_conf["save_model_interval"] > 0:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
        model_save_path = os.path.join(output_dir, training_conf["model_save_path"])
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            "actor_state_dict": agent.policy.state_dict(),
            "q1_state_dict": agent.q1.state_dict(),
            "q2_state_dict": agent.q2.state_dict()
        }, model_save_path)
        log.info(f"SAC model saved to {model_save_path}")

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    visualize_edgeric_training([batch_losses])

if __name__ == "__main__":
    main()
