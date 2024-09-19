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
    self.memory = deque(maxlen=10000)
    self.batch_size = 64
    self.gamma = 0.99  # Discount-Faktor für zukünftige Belohnungen
    self.epsilon = 1.0  # Initiale Epsilon-Wahrscheinlichkeit für Exploration
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1
    self.update_counter = 0
    self.target_update_frequency = 10  # Wie oft das Zielnetzwerk aktualisiert wird

    # Initialisierung der Q- und Zielnetzwerke
    self.q_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))
    self.target_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    self.loss_fn = nn.MSELoss()

    # Replay Memory
    self.memory_size = 100000000
    self.memory = deque(maxlen=self.memory_size)
    
    # Prüfen, ob ein gespeichertes Modell vorhanden ist
    model_path = "dqn-model.pth"
    if os.path.isfile(model_path):
        # Lade das gespeicherte Modell
        self.q_network.load_state_dict(torch.load(model_path))
        self.target_network.load_state_dict(torch.load(model_path))
        self.logger.info("Loaded Q-network from dqn-model.pth")
    else:
        self.logger.info("No saved model found. Starting with a new Q-network.")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events, old_game_state, new_game_state)
    
    # Zustand und Aktion vorbereiten
    old_state_tensor = torch.tensor(old_state, dtype=torch.float32).unsqueeze(0)
    new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
    action_idx = ACTIONS.index(self_action)
    reward_tensor = torch.tensor(reward, dtype=torch.float32)
    
    # Berechnung des aktuellen Q-Wertes
    current_q_values = self.q_network(old_state_tensor).gather(1, torch.tensor([[action_idx]], dtype=torch.int64)).squeeze(1)
    
    # Auswahl der nächsten Aktion und Berechnung des Q-Werts
    if np.random.rand() <= self.epsilon:
        next_action_idx = np.random.choice(len(ACTIONS))
    else:
        with torch.no_grad():
            q_values = self.q_network(new_state_tensor)
            next_action_idx = torch.argmax(q_values).item()
    
    next_q_values = self.q_network(new_state_tensor).gather(1, torch.tensor([[next_action_idx]], dtype=torch.int64)).squeeze(1)
    
    # Ziel-Q-Wert berechnen
    target_q_value = reward_tensor + (self.gamma * next_q_values)
    
    # Verlust berechnen
    loss = self.loss_fn(current_q_values, target_q_value)
    
    # Optimierungsschritt
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Update des Zielnetzwerks
    self.update_counter += 1
    if self.update_counter % self.target_update_frequency == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


def end_of_round(self, last_game_state, last_action, events):
    reward = reward_from_events(self, events, last_game_state, None)
    last_state = state_to_features(last_game_state)
    self.memory.append((last_state, ACTIONS.index(last_action), reward, None))

    # Speichere das Modell
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
    
    # Zusätzliche Belohnung basierend auf der Annäherung an Münzen
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        if new_game_state['coins']:
            old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(coin)) for coin in old_game_state['coins']])
            new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(coin)) for coin in new_game_state['coins']])

            # Belohnung für Annäherung an die nächste Münze
            if new_min_dist < old_min_dist:
                reward_sum += 0.5
        else:
            # Falls keine Münzen mehr vorhanden sind, kann man eine alternative Belohnung in Erwägung ziehen
            reward_sum += 0.1  # Zum Beispiel eine kleine Belohnung, da alle Münzen eingesammelt wurden

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
