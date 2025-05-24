import argparse
import hydra
import gym
import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR

# Remove explicit renderer setting for kaleido since it causes issues.
# (Ensure that kaleido is installed if you want to export images.)
#
# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edgeric_messenger import get_metrics_multi, send_scheduling_weight
torch.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import *
from models.mlp_policy import Policy       # Actor network; expects hidden sizes as a list
from models.mlp_critic import Value         # Critic network; expects hidden sizes as a list
from stream_rl.registry import ENVS
from stream_rl.plots import visualize_edgeric_training
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Improved Actor-Critic with Throughput Reward")
parser.add_argument("--env-name", default="Hopper-v2", metavar="G", help="Name of the environment")
parser.add_argument("--model-path", metavar="G", help="Path of a pre-trained model (optional)")
parser.add_argument("--render", action="store_true", default=False, help="Render the environment")
parser.add_argument("--num-threads", type=int, default=1, metavar="N", help="Number of threads")
parser.add_argument("--seed", type=int, default=1, metavar="N", help="Random seed")
parser.add_argument("--log-interval", type=int, default=1, metavar="N", help="Log interval")
parser.add_argument("--gpu-index", type=int, default=0, metavar="N", help="GPU index")
args = parser.parse_args()


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


def compute_gae(rewards, values, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation (GAE) for one episode."""
    advantages = []
    gae = 0.0
    T = len(rewards)
    for t in reversed(range(T)):
        next_value = values[t+1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, device):
        self.actor = Policy(state_dim, action_dim, [hidden_dim]).to(device)
        self.critic = Value(state_dim, [hidden_dim]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device
        self.episode_buffer = []  # Stores episodes for batch updates

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        output = self.actor(state_tensor)  # Expected shape: [1, action_dim]
        logits = output[0] if isinstance(output, tuple) else output  # [action_dim]
        probs = torch.softmax(logits, dim=-1)  # Agent's scheduling distribution.
        dist = Categorical(probs)
        action = dist.sample()
        value = self.critic(state_tensor)
        return action.item(), dist.log_prob(action), dist.entropy(), value.item(), probs

    def store_episode(self, episode):
        self.episode_buffer.append(episode)

    def finish_batch(self, gae_lambda, entropy_coef, max_norm=0.5):
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        all_entropies = []
        for ep in self.episode_buffer:
            rewards = ep['rewards']
            log_probs = ep['log_probs']
            values = ep['values']
            entropies = ep['entropies']
            advantages, returns = compute_gae(rewards, values, self.gamma, gae_lambda)
            all_log_probs.extend(log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_values.extend(values)
            all_entropies.extend(entropies)
        log_probs_tensor = torch.stack(all_log_probs)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(all_values, dtype=torch.float32, device=self.device)
        entropies_tensor = torch.stack(all_entropies)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)
        actor_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
        critic_loss = (returns_tensor - values_tensor).pow(2).mean()
        entropy_loss = -entropies_tensor.mean()
        total_loss = actor_loss + critic_loss + entropy_coef * entropy_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        # Log gradient norms for debugging.
        actor_grad_norm = sum(p.grad.norm(2).item()**2 for p in self.actor.parameters() if p.grad is not None)**0.5
        critic_grad_norm = sum(p.grad.norm(2).item()**2 for p in self.critic.parameters() if p.grad is not None)**0.5
        log.info(f"Gradient Norms: Actor = {actor_grad_norm:.4f}, Critic = {critic_grad_norm:.4f}")

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.episode_buffer = []
        return total_loss.item()


@hydra.main(config_path="conf", config_name="edge_ric_actor_critic")
def main(conf):
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    env_cls = ENVS[conf["env"]]
    env = env_cls(conf["env_config"])
    env = DiscretizeActionWrapper(env, bins=10)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine number of UEs from initial metrics.
    initial_ue_data = get_metrics_multi()
    num_ues = len(initial_ue_data)
    log.info(f"Determined number of UEs: {num_ues}")
    state_dim = env.observation_space.shape[0]
    action_dim = num_ues  # The actor will output a scheduling distribution over UEs.
    conf["env_config"]["num_UEs"] = num_ues

    ac_params = conf["agent_config"]["actor_critic_params"]
    training_conf = conf["training_config"]
    agent = ActorCriticAgent(
        state_dim, 
        action_dim,
        hidden_dim=ac_params["hidden_dim"],
        actor_lr=ac_params["actor_lr"],
        critic_lr=ac_params["critic_lr"],
        gamma=ac_params["gamma"],
        device=device
    )

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        log.info(f"Loaded pre-trained model from {args.model_path}")

    actor_scheduler = StepLR(agent.actor_optimizer, step_size=100, gamma=0.99)
    critic_scheduler = StepLR(agent.critic_optimizer, step_size=100, gamma=0.99)

    num_iters = conf["num_iters"]
    max_steps_per_episode = training_conf["max_steps_per_episode"]
    update_interval = training_conf["update_interval"]
    # Set reward_scale to 1.0 so that the throughput reward is used as-is.
    reward_scale = training_conf.get("reward_scale", 1.0)
    gae_lambda = ac_params["gae_lambda"]
    entropy_coef = ac_params["entropy_coef"]

    episode_count = 0
    batch_losses = []
    all_episode_rewards = []
    moving_avg_window = 10

    for iter_num in range(num_iters):
        state = env.reset()
        # If your state is lower-dimensional than expected by the network, modify it here.
        if state.shape[0] == 6:
            state = np.concatenate([state, state])
        state_dim = state.shape[0]  # now should be 12

        # Instead of using the number of UEs, force action_dim = 2.
        action_dim = 2
        episode_reward = 0.0
        done = False
        step = 0
        episode = {'rewards': [], 'log_probs': [], 'values': [], 'entropies': []}

        while not done and step < max_steps_per_episode:
            if args.render:
                env.render()
            action, log_prob, entropy, value, agent_probs = agent.select_action(state)
            ue_data = get_metrics_multi()
            num_ues_actual = len(ue_data)
            env.num_UEs = num_ues_actual
            CQIs  = [data['CQI'] for data in ue_data.values()]
            cqi_map = conf["env_config"]["cqi_map"]
            optimal_weights = np.array([float(cqi_map[int(cqi)][0]) for cqi in CQIs])
            # Use only as many agent probabilities as UEs.
            agent_probs_np = agent_probs.detach().cpu().numpy().flatten()[:num_ues_actual]
            # Log the computed throughput reward for debugging.
            computed_reward = optimal_weights.sum()
            log.info(f"Computed throughput reward (raw): {computed_reward:.2f}")
            # For simplicity, we now use the computed throughput reward directly.
            reward = computed_reward  
            scaled_reward = reward * reward_scale

            RNTIs = list(ue_data.keys())
            weight = np.zeros(num_ues_actual * 2)
            for ue in range(num_ues_actual):
                weight[ue*2] = RNTIs[ue]
                weight[ue*2 + 1] = agent_probs_np[ue] if ue < len(agent_probs_np) else 0.0
            send_scheduling_weight(weight, True)

            next_state, _, done, info = env.step(action, RNTIs, CQIs, 
                                                  [data['Backlog'] for data in ue_data.values()],
                                                  np.sum([data['Tx_brate'] for data in ue_data.values()]),
                                                  np.ones(num_ues_actual)*300000)
            # Ensure next_state has the correct shape.
            if next_state.shape[0] == 6:
                next_state = np.concatenate([next_state, next_state])
            episode['rewards'].append(scaled_reward)
            episode['log_probs'].append(log_prob)
            episode['values'].append(value)
            episode['entropies'].append(entropy)
            state = next_state
            episode_reward += reward
            step += 1

        agent.store_episode(episode)
        episode_count += 1
        all_episode_rewards.append(episode_reward)
        moving_avg = np.mean(all_episode_rewards[-moving_avg_window:]) if len(all_episode_rewards) >= moving_avg_window else np.mean(all_episode_rewards)
        log.info(f"Iteration: {iter_num}\tEpisode Reward: {episode_reward:.2f}\tMoving Avg Reward: {moving_avg:.2f}")

        if episode_count % update_interval == 0:
            loss = agent.finish_batch(gae_lambda, entropy_coef)
            batch_losses.append(loss)
            log.info(f"Batch update (after {update_interval} episodes): Loss = {loss:.4f}")
            actor_scheduler.step()
            critic_scheduler.step()

    if training_conf["save_model_interval"] > 0:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
        model_save_path = os.path.join(output_dir, training_conf["model_save_path"])
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict()
        }, model_save_path)
        log.info(f"Actor-Critic model saved to {model_save_path}")

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.get("runtime", {}).get("output_dir", hydra_cfg["run"]["dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    visualize_edgeric_training([batch_losses])


if __name__ == "__main__":
    main()
