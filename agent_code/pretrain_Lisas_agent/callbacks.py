import os
import pickle
import numpy as np
from collections import deque
from random import choice, random, shuffle

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
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
    self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy during action selection
    self.last_two_actions = deque(['WAIT', 'WAIT'], maxlen=2)  # Initialize with 'WAIT'

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    
def state_to_features(self, game_state):
    """
    Wandelt den Spielzustand in einen Zustandsvektor für die Q-Tabelle um.
    """
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]

    # Umgebung extrahieren (5x5 Gitter)
    radius = 2
    x_min = max(0, x - radius)
    x_max = min(field.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(field.shape[1], y + radius + 1)

    surroundings = field[x_min:x_max, y_min:y_max]

    # Padding, falls am Rand
    pad_width_x = (radius - (x - x_min), radius - (x_max - x - 1))
    pad_width_y = (radius - (y - y_min), radius - (y_max - y - 1))
    surroundings = np.pad(surroundings, (pad_width_x, pad_width_y), 'constant', constant_values=-1)

    surroundings = tuple(surroundings.flatten())

    # Relative Position zum aktuellen Ziel
    target = get_next_target(game_state)
    if target:
        dx, dy = target[0] - x, target[1] - y
    else:
        dx, dy = 0, 0

    # Inklusion der letzten zwei Aktionen
    last_actions_indices = [ACTION_TO_INDEX[action] for action in self.last_two_actions]

    features = surroundings + (dx, dy) + tuple(last_actions_indices)
    return features

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def get_valid_actions(game_state, self):
    """
    Gibt die Liste der gültigen Aktionen von der aktuellen Position zurück.
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


    # 'BOMB' hinzufügen, wenn Bedingungen erfüllt sind
    if self.steps_since_last_bomb >= 3:
        valid_actions.append('BOMB')
    
    # 'WAIT' nur hinzufügen, wenn keine anderen Aktionen verfügbar sind
    if not valid_actions:
        valid_actions.append('WAIT')

    return valid_actions

# Importieren der notwendigen Funktionen aus train.py, um Duplikate zu vermeiden
# Stellen Sie sicher, dass keine unnötigen Funktionen vorhanden sind

def get_next_target(game_state):
    """
    Bestimmt das nächste Ziel für den Agenten basierend auf den Prioritäten.
    """
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = [other[3] for other in game_state['others']]
    free_space = field == 0

    # Prüfen, ob der Agent in Gefahr ist
    in_danger = False
    for (bx, by), timer in bombs:
        if (x, y) in get_bomb_radius((bx, by), field):
            in_danger = True
            break

    if in_danger:
        # 1. Nächsten sicheren Platz finden
        safe_positions = []
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if free_space[i, j] and is_safe_position(game_state, (i, j), bombs):
                    safe_positions.append((i, j))
        target = look_for_targets(free_space, (x, y), safe_positions)
        if target:
            return target

    # 2. Sicherer Bombenplatz, der eine Kiste zerstört
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

    # 3. Nächste Münze finden
    coins = game_state['coins']
    if coins:
        target = look_for_targets(free_space, (x, y), coins)
        if target:
            return target

    # 4. Gegnerischer Agent
    if others:
        target = look_for_targets(free_space, (x, y), others)
        if target:
            return target

    return None

def look_for_targets(free_space, start, targets, logger=None):
    """Finde die Richtung zum nächsten Ziel, das über freie Felder erreichbar ist."""
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Abstand zu den Zielen berechnen
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Ziel gefunden
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
    # Ersten Schritt zum Ziel finden
    current = best
    while parent_dict[current] != start:
        current = parent_dict[current]
    return current

def is_safe_position(game_state, position, bombs):
    """
    Überprüft, ob eine Position sicher ist, unter Berücksichtigung von Bomben und deren Explosionsradius.
    """
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
    """
    Gibt die von der Bombe betroffenen Felder zurück, unter Berücksichtigung von Wänden und Kisten.
    """
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

# Ihre Symmetrie-Funktionen bleiben unverändert
def apply_transformation(state, action, rotation=0, axis=None):
    """Apply rotation or mirror to both state and action."""
    # Annahme: Die Umgebung ist jetzt ein 5x5 Gitter
    surroundings = np.array(state[:-4]).reshape((5, 5))
    dx, dy = state[-4], state[-3]
    last_actions = state[-2:]  # Letzte zwei Aktionen
    action_idx = ACTIONS.index(action) if action in ACTIONS else None

    # Rotation der Umgebung und Aktion
    if rotation:
        k = rotation // 90
        surroundings = np.rot90(surroundings, k=k)
        if action_idx is not None and action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            action_idx = (action_idx + k) % 4  # Aktion rotieren

    # Spiegelung der Umgebung und Aktion
    if axis == 'x':
        surroundings = np.flipud(surroundings)
        if action == 'UP':
            action_idx = ACTIONS.index('DOWN')
        elif action == 'DOWN':
            action_idx = ACTIONS.index('UP')
    elif axis == 'y':
        surroundings = np.fliplr(surroundings)
        if action == 'LEFT':
            action_idx = ACTIONS.index('RIGHT')
        elif action == 'RIGHT':
            action_idx = ACTIONS.index('LEFT')

    # Transformation von dx, dy
    if rotation:
        angle = np.deg2rad(rotation)
        cos_theta = int(np.cos(angle))
        sin_theta = int(np.sin(angle))
        dx_new = cos_theta * dx - sin_theta * dy
        dy_new = sin_theta * dx + cos_theta * dy
        dx, dy = dx_new, dy_new
    if axis == 'x':
        dy = -dy
    elif axis == 'y':
        dx = -dx

    transformed_state = tuple(surroundings.flatten()) + (dx, dy) + last_actions
    transformed_action = ACTIONS[action_idx] if action_idx is not None else action

    return transformed_state, transformed_action

def get_symmetric_states_and_actions(state, action):
    """Generiere alle eindeutigen symmetrischen Zustände und entsprechende Aktionen."""
    transformations = set()

    # Rotationen
    for rotation in [0, 90, 180, 270]:
        transformed_state, transformed_action = apply_transformation(state, action, rotation=rotation)
        transformations.add((transformed_state, transformed_action))

    # Spiegelungen
    for axis in ['x', 'y']:
        transformed_state, transformed_action = apply_transformation(state, action, axis=axis)
        transformations.add((transformed_state, transformed_action))

    return list(transformations)
