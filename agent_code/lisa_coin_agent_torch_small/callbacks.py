import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    self.logger.info("Setting up agent with Deep Q-Learning.")
    
    # Hyperparameter
    self.alpha = 0.001  # Lernrate für Adam-Optimizer
    self.gamma = 0.99  # Diskontfaktor
    self.epsilon = 0.1 # Epsilon für epsilon-greedy
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1
    self.batch_size = 64
    self.memory = deque(maxlen=10000)
    
    # Netzwerke initialisieren
    self.q_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))  # Annahme: 3 Features aus `state_to_features`
    self.target_network = QNetwork(input_dim=3, output_dim=len(ACTIONS))
    self.target_network.load_state_dict(self.q_network.state_dict())
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
    self.loss_fn = nn.MSELoss()
    
    # Prüfen, ob ein gespeichertes Modell vorhanden ist
    if os.path.isfile("dqn-model.pth"):
        self.q_network.load_state_dict(torch.load("dqn-model.pth"))
        self.target_network.load_state_dict(torch.load("dqn-model.pth"))
        self.logger.info("Loaded Q-network from dqn-model.pth")
    else:
        self.logger.info("No saved model found. Starting with a new Q-network.")

def act(self, game_state: dict) -> str:
    """
    The agent decides on an action based on the current state.
    """
    state = torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)

    # Epsilon-greedy action selection
    if random.random() < self.epsilon:
        valid_actions = get_valid_actions(game_state)
        action = random.choice(valid_actions)
    else:
        with torch.no_grad():
            q_values = self.q_network(state)
            valid_actions = get_valid_actions(game_state)
            valid_q_values = [q_values[0][ACTIONS.index(action)].item() for action in valid_actions]
            action = valid_actions[np.argmax(valid_q_values)]

    return action

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.
    """
    if game_state is None:
        return None

    x, y = game_state['self'][3]
    coins = game_state['coins']
    field = game_state['field']

    if coins:
        distances = []
        for coin in coins:
            path_length = bfs(field, (x, y), coin)
            distances.append(path_length)
        
        min_distance = min(distances)
        closest_coin = coins[distances.index(min_distance)]
        dx = closest_coin[0] - x
        dy = closest_coin[1] - y
        return np.array([dx, dy, min_distance])
    else:
        return np.array([0, 0, 0])


def get_valid_actions(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    valid_actions = []

    directions = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }
    
    for action, (nx, ny) in directions.items():
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:  # Check if the next position is free
            valid_actions.append(action)
    
    if len(valid_actions) == 0:
        valid_actions.append('WAIT')
    
    return valid_actions


def bfs(field, start, target):
    """
    Breadth-First Search (BFS): Algorithmus, um den kürzesten Weg von einem Startpunkt zu einem Zielpunkt
    in einem Gitterfeld zu finden.
    """
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == target:
            return dist

        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))
    
    return float('inf')

