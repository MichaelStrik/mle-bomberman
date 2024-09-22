import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .callbacks import state_to_features, QNetwork
import events as e
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.memory = deque(maxlen=10000)
    self.batch_size = 64
    self.gamma = 0.99  # Discount-Faktor 
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1
    self.update_counter = 0
    self.target_update_frequency = 10  # How often the target network is updated

    # Initialization of the Q and target networks
    self.q_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))
    self.target_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    self.loss_fn = nn.MSELoss()

    # Replay Memory
    self.memory_size = 100000000
    self.memory = deque(maxlen=self.memory_size)
    
    # Check whether a saved Q-table exists
    model_path = "dqn-model.pth"
    if os.path.isfile(model_path):
        self.q_network.load_state_dict(torch.load(model_path))
        self.target_network.load_state_dict(torch.load(model_path))
        self.logger.info("Loaded Q-network from dqn-model.pth")
    else:
        self.logger.info("No saved model found. Starting with a new Q-network.")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events, old_game_state, new_game_state)
    
    old_state_tensor = torch.tensor(old_state, dtype=torch.float32).unsqueeze(0)
    new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
    action_idx = ACTIONS.index(self_action)
    reward_tensor = torch.tensor(reward, dtype=torch.float32)
    
    # Calculate q-value
    current_q_values = self.q_network(old_state_tensor).gather(1, torch.tensor([[action_idx]], dtype=torch.int64)).squeeze(1)
    
    # Choose the next action and calculate the Q value
    if np.random.rand() <= self.epsilon:
        next_action_idx = np.random.choice(len(ACTIONS))
    else:
        with torch.no_grad():
            q_values = self.q_network(new_state_tensor)
            next_action_idx = torch.argmax(q_values).item()
    
    next_q_values = self.q_network(new_state_tensor).gather(1, torch.tensor([[next_action_idx]], dtype=torch.int64)).squeeze(1)
    
    # Calculate target Q-value
    target_q_value = reward_tensor + (self.gamma * next_q_values)
    
    # Calculate loss
    loss = self.loss_fn(current_q_values, target_q_value)
    
    # optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Update the target network
    self.update_counter += 1
    if self.update_counter % self.target_update_frequency == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    reward = reward_from_events(self, events, last_game_state, None)
    last_state = state_to_features(last_game_state)
    self.memory.append((last_state, ACTIONS.index(last_action), reward, None))

    # save the model
    torch.save(self.q_network.state_dict(), "dqn-model.pth")
    self.logger.info("Saved Q-network to dqn-model.pth.")

def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -1,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -10,
        "STUCK_IN_LOOP": -8
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])
    
    # reward if closer to coin 
    if old_game_state is not None and new_game_state is not None:
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)

        if old_features is not None and new_features is not None:
            # Distance to coins 
            old_min_distance = old_features[2]  
            new_min_distance = new_features[2]  

            # Reward if closer to coin 
            if new_min_distance < old_min_distance:
                reward_sum += 0.5
            elif new_min_distance == old_min_distance:
                reward_sum -= 0.05  
            else:
                reward_sum -= 0.5  

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
