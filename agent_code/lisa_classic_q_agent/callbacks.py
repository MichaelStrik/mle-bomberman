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

def act(self, game_state: dict) -> str:
    """
    Decide on an action based on the current game state.
    """
    features = state_to_features(self, game_state)
    state = features

    # Initialize Q-values for unseen states
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    q_values = self.q_table[state]

    valid_actions = get_valid_actions(game_state, self)

    # Epsilon-greedy action selection
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
