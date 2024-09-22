import os
import pickle
import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up agent from saved state.")
    
    # Check whether a saved Q-table exists
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.epsilon = 0.2
    self.epsilon_decay = 1.0
    self.epsilon_min = 0.1

    
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    state = tuple(features)

    # Q-Table
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    q_values = self.q_table[state]

    valid_actions = get_valid_actions(game_state)

    # Exploration vs Exploitation
    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
    else:
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
    
    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
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
        if field[nx, ny] == 0:  # Check if the next position is free - no wall/create (box)
            valid_actions.append(action)
    
    if len(valid_actions) == 0:
        valid_actions.append('WAIT')
    
    return valid_actions


def bfs(field, start, target):
    """
    Breadth-First Search (BFS): Algorithm for finding the shortest path 
    from a starting point to a target point in a grid field.
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
