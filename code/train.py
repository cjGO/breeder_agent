from models import Actor, Critic
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


list_size = 100
hs = 100
lr = 2e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize actor and critic
actor = Actor(list_size,hs).to(device)
critic = Critic(list_size,hs).to(device)

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

# Training loop parameters
num_episodes = 1000000
gamma = 0.99
tolerance = 1e-5  # Convergence tolerance
last_avg_rewards = None
reporter = 1000

reward_history = []

for episode in range(num_episodes):
    state = torch.rand(1, list_size).to(device)  # Simulate an environment state, moved to GPU
    state = F.normalize(state, p=2.0, dim=1)

    for _ in range(1):
        action_probs = actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        true_value = torch.max(state)
        chosen_value = state[0, action]  # Assuming action is the index of the chosen value

        reward_scale = 1.0
        reward = (chosen_value / true_value) * reward_scale
        reward = reward.unsqueeze(-1)  # Adjust shape for critic if necessary
        
        value = critic(state)
        
        advantage = reward + gamma * value.detach() - value
        
        critic_loss = advantage.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        actor_loss = -dist.log_prob(action) * advantage.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        reward_history.append(reward.item())

    if episode % reporter == 0 and episode > 0:
        avg_rewards = np.mean(reward_history[-reporter:])
        print(f"Episode {episode}, Average Reward: {avg_rewards}")
        
        if last_avg_rewards is not None and abs(last_avg_rewards - avg_rewards) < tolerance:
            print("Convergence criteria met. Stopping training.")
            break
        last_avg_rewards = avg_rewards
