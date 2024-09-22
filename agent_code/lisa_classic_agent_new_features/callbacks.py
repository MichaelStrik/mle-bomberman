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

    # Check whether a saved Q-table exists
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.2  # Learning rate
    self.gamma = 0.95  # Discount factor
    self.epsilon = 0.01 # Epsilon for epsilon-greedy strategy
    self.epsilon_decay = 0.995  # Epsilon decay
    self.epsilon_min = 0.0
    
    
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

    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
    else:
        valid_q_values = np.array([q_values[ACTIONS.index(action)] for action in valid_actions])
        best_actions = np.where(valid_q_values == valid_q_values.max())[0]
        action = valid_actions[np.random.choice(best_actions)]
        #valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        #action = valid_actions[np.argmax(valid_q_values)]
        
    # Decay epsilon for exploration/exploitation balance
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    return action


def get_valid_actions(game_state):
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_map = np.ones(arena.shape) * 5  # Default bomb map value
    explosion_map = game_state['explosion_map']
    
    # Populate the bomb map
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if 0 <= i < bomb_map.shape[0] and 0 <= j < bomb_map.shape[1]:
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check valid actions: UP, DOWN, LEFT, RIGHT, WAIT
    directions = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]  # (dx, dy) pairs for actions
    valid_actions = []
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy  # New position
        if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
            if arena[nx, ny] == 0 and explosion_map[nx, ny] <= 1 and bomb_map[nx, ny] > 0:
                valid_actions.append((nx, ny))

    actions = []
    if (x - 1, y) in valid_actions: actions.append('LEFT')
    if (x + 1, y) in valid_actions: actions.append('RIGHT')
    if (x, y - 1) in valid_actions: actions.append('UP')
    if (x, y + 1) in valid_actions: actions.append('DOWN')

    # Add BOMB if valid and no bomb dropped recently
    if bombs_left > 0 and bomb_map[x, y] > 0:  # Check bomb map value at current position
        actions.append('BOMB')

    # If the actions list is empty, add 'WAIT'
    if not actions:
        actions.append('WAIT')

    return actions



def is_safe_to_place_bomb(game_state, position=None):
    """Überprüft, ob es sicher ist, eine Bombe zu platzieren."""
    if position is None:
        position = game_state['self'][3]  # Aktuelle Position des Agenten
    x, y = position
    field = game_state['field']

    # Prüfen, ob der Agent nach dem Platzieren der Bombe einen sicheren Fluchtweg hat
    for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and is_safe_position((nx, ny), game_state):
                return True
    return False


def is_safe_position(game_state, position, bomb_timers):
    """
    Check if a position is safe considering the bombs' timers and their blast radius.
    
    :param game_state: The current game state dictionary.
    :param position: The (x, y) tuple of the position to check.
    :param bomb_timers: A list of (position, timer) tuples for each bomb.
    :return: True if the position is safe, False otherwise.
    """
    field = game_state['field']
    x, y = position

    for bomb_pos, timer in bomb_timers:
        bx, by = bomb_pos

        if timer <= 0:
            continue

        # Check horizontal and vertical lines for blast danger
        if bx == x and abs(by - y) <= 2 and timer <= 4:  # Same column and within blast range
            return False
        if by == y and abs(bx - x) <= 2 and timer <= 4:  # Same row and within blast range
            return False

    return True


def bfs(field, start, target):
    """
    Breadth-First Search to calculate the shortest path from start to target.
    Returns the length of the path or a large value if not reachable.
    """
    queue = deque([(start, 0)])
    visited = set([start])
    
    while queue:
        (x, y), distance = queue.popleft()
        
        if (x, y) == target:
            return distance
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and field[nx, ny] == 0:  # 0 means walkable
                visited.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    
    return float('inf')  # If no path found

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

    # Initialize the feature map (channels: field, coins, bombs, others)
    arena = game_state['field']
    bomb_map = np.zeros_like(arena)
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = game_state['others']

    # Mark bombs and their timers
    for (x_bomb, y_bomb), timer in bombs:
        bomb_map[x_bomb, y_bomb] = timer

    # Set coin positions in a separate channel
    coin_map = np.zeros_like(arena)
    for x_coin, y_coin in coins:
        coin_map[x_coin, y_coin] = 1

    # Set positions of other players in another channel
    others_map = np.zeros_like(arena)
    for _, _, _, (x_other, y_other) in others:
        others_map[x_other, y_other] = 1

    # Combine all maps (arena, bombs, coins, others, explosion map) into one feature array
    features = np.stack([arena, bomb_map, coin_map, others_map, explosion_map], axis=0)
    # Flatten the array to make it hashable
    flattened_features = features.flatten()

    return flattened_features


def bfs(field, start, target):
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

def find_crates(game_state):
    """
    Findet alle Kisten auf dem Spielfeld und gibt deren Positionen zurück.

    :param game_state: Das aktuelle Spielzustand-Dictionary.
    :return: Eine Liste von Tupeln mit den Positionen der Kisten.
    """
    crates = []
    field = game_state['field']  # Das Spielfeld
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 1:  # 1 steht normalerweise für eine Kiste
                crates.append((x, y))
    return crates

def get_bomb_timers(game_state):
    """
    Get the positions and timers of all bombs on the field.
    
    :param game_state: The current game state dictionary.
    :return: A list of (position, timer) tuples for each bomb.
    """
    bombs = game_state['bombs']  # List of tuples with ((x, y), timer)
    return [(bomb[0], bomb[1]) for bomb in bombs]

import math

def safe_direction_away_from_bomb(agent_pos: tuple, bomb_pos: tuple) -> int:
    """
    Returns the direction the agent should take to move away from a bomb. 
    1: Right, 2: Up, 3: Left, 4: Down
    """
    assert agent_pos != bomb_pos

    # Calculate the angle between agent and bomb
    angle = math.atan2(bomb_pos[1] - agent_pos[1], bomb_pos[0] - agent_pos[0])
    if angle < 0:
        angle += 2 * math.pi

    # Adjust for grid orientation (array coordinates)
    angle -= math.pi / 2

    # Calculate the opposite direction (180 degrees away from the bomb)
    opposite_direction = (angle - math.pi) % (2 * math.pi)

    if 0 <= opposite_direction < math.pi / 4:
        return 1  # RIGHT
    elif math.pi / 4 <= opposite_direction < 3 * math.pi / 4:
        return 2  # UP
    elif 3 * math.pi / 4 <= opposite_direction < 5 * math.pi / 4:
        return 3  # LEFT
    elif 5 * math.pi / 4 <= opposite_direction < 7 * math.pi / 4:
        return 4  # DOWN
    else:
        return 1  # Default to RIGHT if something goes wrong

