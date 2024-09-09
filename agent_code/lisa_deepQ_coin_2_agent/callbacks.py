import os
import numpy as np
from .train import DQNAgent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    self.logger.info("Setting up agent from saved state.")
    
    self.agent = DQNAgent()

    # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden ist
    if os.path.isfile("my-dqn-model.h5"):
        self.agent.load_model("my-dqn-model.h5")
        self.logger.info("Loaded model from my-dqn-model.h5")
    else:
        self.logger.info("No saved model found. Starting with a fresh model.")

def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)

def state_to_features(game_state: dict) -> np.array:
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

def bfs(field, start, target):
    from collections import deque
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
