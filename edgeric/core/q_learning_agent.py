import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3,
                 gamma=0.99, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-Network and optimizer
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-network based on a single transition."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Current Q-value for the taken action
        q_value = self.q_network(state_tensor)[0, action]
        
        # Compute the target Q-value
        with torch.no_grad():
            next_q_value = self.q_network(next_state_tensor).max(1)[0].item()
        target = reward + (0 if done else self.gamma * next_q_value)
        target_tensor = torch.FloatTensor([target])
        
        # Calculate loss and perform gradient descent
        loss = self.loss_fn(q_value, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate at episode end
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
        return loss.item()
