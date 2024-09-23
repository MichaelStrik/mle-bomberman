import os
import pickle
import numpy as np
from collections import deque
from random import choice, random, shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

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
    # Load the Q-table if it exists
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        # Initialize an empty Q-table
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.steps_since_last_bomb = 0
    self.epsilon = 0.05  # Epsilon for epsilon-greedy strategy during action selection
    self.last_two_actions = deque(['WAIT', 'WAIT'], maxlen=2)  # Initialize with 'WAIT' - for prevention of wobbeling

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(self, game_state)
    state = features

    # Initialize Q-values for unseen states
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    q_values = self.q_table[state]

    valid_actions = get_valid_actions(game_state, self)

    # Epsilon-greedy
    if random() < self.epsilon:
        action = choice(valid_actions)
    else:
        # Select the action with the highest Q-value among valid actions
        valid_q_indices = [ACTIONS.index(a) for a in valid_actions]
        valid_q_values = q_values[valid_q_indices]
        max_q = np.max(valid_q_values)
        best_actions = [valid_actions[i] for i in range(len(valid_actions)) if valid_q_values[i] == max_q]
        action = choice(best_actions)

    # Update steps since last bomb
    if action == 'BOMB':
        self.steps_since_last_bomb = 0
    else:
        self.steps_since_last_bomb += 1

    # Update last two actions
    self.last_two_actions.append(action)

    return action

def state_to_features(self, game_state):
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

    field = game_state['field']
    x, y = game_state['self'][3]

    # 5x5 surrounding
    radius = 2
    x_min = max(0, x - radius)
    x_max = min(field.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(field.shape[1], y + radius + 1)

    surroundings = field[x_min:x_max, y_min:y_max]

    # Padding, if at the edge
    pad_width_x = (radius - (x - x_min), radius - (x_max - x - 1))
    pad_width_y = (radius - (y - y_min), radius - (y_max - y - 1))
    surroundings = np.pad(surroundings, (pad_width_x, pad_width_y), 'constant', constant_values=-1)

    surroundings = tuple(surroundings.flatten())

    # Relative position to the current target
    target = get_next_target(game_state)
    if target:
        dx, dy = target[0] - x, target[1] - y
    else:
        dx, dy = 0, 0

    # add last two steps
    last_actions_indices = [ACTION_TO_INDEX[action] for action in self.last_two_actions]

    features = surroundings + (dx, dy) + tuple(last_actions_indices)
    return features

def get_valid_actions(game_state, self):
    """
    Returns the list of valid actions from the current position.
    """
    arena = game_state['field']
    x, y = game_state['self'][3]
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = [other[3] for other in game_state['others']]
    bomb_positions = [bomb[0] for bomb in bombs]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_map = {(-1, 0): 'LEFT', (1, 0): 'RIGHT', (0, -1): 'UP', (0, 1): 'DOWN'}
    valid_actions = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and
                arena[nx, ny] == 0 and
                explosion_map[nx, ny] <= 0 and
                (nx, ny) not in bomb_positions and
                (nx, ny) not in others):
            valid_actions.append(action_map[(dx, dy)])

    can_place_bomb = True
    radius = 2  # 5x5 

    # Check for bombs or explosions in the 5x5 field
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if (0 <= i < arena.shape[0] and 0 <= j < arena.shape[1]):
                if explosion_map[i, j] > 0:
                    can_place_bomb = False
                if (i, j) in bomb_positions:
                    can_place_bomb = False

    if can_place_bomb:
        valid_actions.append('BOMB')
    
    if self.last_two_actions[-1] != 'BOMB' and not can_place_bomb:
        valid_actions.append('WAIT')
    

    # Only add 'WAIT' if no other actions are available
    if not valid_actions:
        valid_actions.append('WAIT')

    return valid_actions


def get_next_target(game_state):

    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = [other[3] for other in game_state['others']]
    free_space = field == 0

    in_danger = False
    for (bx, by), timer in bombs:
        if (x, y) in get_bomb_radius((bx, by), field):
            in_danger = True
            break

    if in_danger:
        # 1. Find next safe position 
        safe_positions = []
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if free_space[i, j] and is_safe_position(game_state, (i, j), bombs):
                    safe_positions.append((i, j))
        target = look_for_targets(free_space, (x, y), safe_positions)
        if target:
            return target

    # 2. save place to place bomb which destroys create
    crates = np.argwhere(field == 1)
    bomb_positions = []
    for crate in crates:
        cx, cy = crate
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and
                    free_space[nx, ny] and is_safe_position(game_state, (nx, ny), bombs)):
                bomb_positions.append((nx, ny))
    target = look_for_targets(free_space, (x, y), bomb_positions)
    if target:
        return target

    # 3. Find a coin
    coins = game_state['coins']
    if coins:
        target = look_for_targets(free_space, (x, y), coins)
        if target:
            return target

    # 4. find an enemy
    if others:
        target = look_for_targets(free_space, (x, y), others)
        if target:
            return target

    return None

def look_for_targets(free_space, start, targets, logger=None):
    """Find the direction to the next target that can be reached."""
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # calculations of distance to targets
        #d=bfs_distance(targets, current)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(nx, ny) for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[nx, ny]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Find first step to target
    current = best
    while parent_dict[current] != start:
        current = parent_dict[current]
    return current

def is_safe_position(game_state, position, bombs):
    x, y = position
    arena = game_state['field']
    explosion_map = game_state['explosion_map']

    for (bx, by), timer in bombs:
        if timer <= 0:
            continue
        affected_tiles = get_bomb_radius((bx, by), arena)
        if (x, y) in affected_tiles:
            return False
    return True

def get_bomb_radius(bomb_pos, arena):
    x_bomb, y_bomb = bomb_pos
    radius = [(x_bomb, y_bomb)]  # Include bomb position itself

    # Explosion in vier Richtungen (Reichweite: 3 Felder)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, 4):
            nx, ny = x_bomb + dist * dx, y_bomb + dist * dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
                if arena[nx, ny] == -1:  # Wand blockiert Explosion
                    break
                radius.append((nx, ny))
                if arena[nx, ny] == 1:  # Kiste blockiert Explosion nach diesem Feld
                    break
            else:
                break
    return radius


def bfs_distance(arena, start, target):
    """
    Breadth-first search (BFS): calculates the shortest path distance from a starting point to a target on a game board
    """
    if start == target:
        return 0
    from collections import deque
    queue = deque()
    visited = set()
    queue.append((start, 0))
    visited.add(start)

    while queue:
        position, distance = queue.popleft()
        x, y = position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and
                    arena[nx, ny] == 0 and (nx, ny) not in visited):
                if (nx, ny) == target:
                    return distance + 1
                queue.append(((nx, ny), distance + 1))
                visited.add((nx, ny))
    return float('inf')

