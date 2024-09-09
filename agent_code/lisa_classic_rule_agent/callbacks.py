import os
import pickle
import numpy as np
from collections import deque
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.logger.info("Setting up agent from saved state.")
    
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.1
    self.gamma = 0.9
    self.epsilon = 0.1
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.0
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0

    # Zähler für Kisten, Münzen und Schritte nur  für uns um fortschritt zu überprüfen
    self.destroyed_crates = 0
    self.collected_coins = 0
    self.steps_survived = 0

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    state = tuple(features)

    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    
    valid_actions = get_valid_actions(game_state)

    self.logger.debug("Querying Q-table for action.")
    q_values = self.q_table[state]
    valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
    
    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
    else:
        action = valid_actions[np.argmax(valid_q_values)]
    
    # Bombenstrategie: Platziere eine Bombe, wenn Kisten in der Nähe sind und ein sicherer Fluchtweg existiert
    if 'BOMB' in valid_actions and should_place_bomb(game_state):
        action = 'BOMB'
    
    # Wenn eine Bombe platziert wurde, wird sie der Bomben-Historie hinzugefügt
    if action == 'BOMB':
        self.bomb_history.append(game_state['self'][3])

    self.last_action = action
    return action

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # Feature 1: Richtung des nächsten Ziels (Münze, Kiste, Gegner)
    coins = game_state['coins']
    crates = [(ix, iy) for ix, iy in np.argwhere(field == 1)]
    others = [other[3] for other in game_state['others']]
    targets = coins + crates + others

    next_step = look_for_targets(field == 0, (x, y), targets)

    dx, dy = 0, 0
    if next_step:
        dx, dy = next_step[0] - x, next_step[1] - y
    
    # Feature 2: Entfernung zum nächsten Ziel
    min_distance = np.linalg.norm(np.array([dx, dy]))

    # Feature 3: Abstand zur nächsten Bombe
    bomb_distances = [np.linalg.norm(np.array([x, y]) - np.array(bomb[0])) for bomb in bombs]
    min_bomb_distance = min(bomb_distances) if bomb_distances else float('inf')

    # Feature 4: Bombentimer
    bomb_timers = [timer for _, timer in bombs]
    min_bomb_timer = min(bomb_timers) if bomb_timers else float('inf')

    # Feature 5: Anzahl verbleibender Gegner
    num_opponents = len(game_state['others'])

    return np.array([dx, dy, min_distance, min_bomb_distance, min_bomb_timer, num_opponents])


def get_valid_actions(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    valid_actions = []

    directions = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }
    
    for action, (nx, ny) in directions.items():
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0 and explosion_map[nx, ny] == 0):  # Feld ist frei und sicher
            valid_actions.append(action)
    
    if len(valid_actions) == 0:
        valid_actions.append('WAIT')
    
    # Überprüfe, ob BOMB eine gültige Aktion ist (z.B. keine Bombe direkt daneben)
    if (game_state['self'][2] and all(np.linalg.norm(np.array(bomb[:2]) - np.array((x, y))) > 1 for bomb in bombs)):
        valid_actions.append('BOMB')
    
    return valid_actions

def should_place_bomb(game_state):
    """
    Überprüfe, ob es sicher und sinnvoll ist, eine Bombe zu platzieren.
    """
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # Überprüfe, ob es Kisten oder Gegner in der direkten Nachbarschaft gibt
    nearby_crates = any([bfs(field, (x, y), (cx, cy)) == 1 for cx, cy in np.argwhere(field == 1)])
    nearby_opponents = any([bfs(field, (x, y), (ox, oy)) == 1 for ox, oy in [opponent[3] for opponent in game_state['others']]])

    # Überprüfe, ob ein sicherer Fluchtweg existiert
    safe_positions = get_safe_positions_after_bomb(field, (x, y), bombs, explosion_map)

    # Bombe nur platzieren, wenn es einen sicheren Fluchtweg gibt und entweder eine Kiste oder ein Gegner in der Nähe ist
    return (nearby_crates or nearby_opponents) and safe_positions


def get_safe_positions_after_bomb(field, start, bombs, explosion_map):
    """
    Finde sichere Positionen, zu denen der Agent nach dem Platzieren einer Bombe fliehen kann.
    """
    x, y = start
    safe_positions = []

    for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0 and explosion_map[nx, ny] == 0):
            safe_positions.append((nx, ny))
    
    return safe_positions

def look_for_targets(free_space, start, targets, logger=None):
    if len(targets) == 0: 
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: 
        logger.debug(f'Suitable target found at {best}')

    current = best
    while True:
        if parent_dict[current] == start: 
            return current
        current = parent_dict[current]


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

def add_custom_events(self, old_game_state: dict, new_game_state: dict, self_action: str, events: list[str]):
    """
    Fügt benutzerdefinierte Ereignisse basierend auf zusätzlichen Bedingungen hinzu.
    """
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Bewegung erkennen und mögliche Schleifen vermeiden
        if new_position in self.coordinate_history:
            if self.coordinate_history.count(new_position) > 2:
                events.append("STUCK_IN_LOOP")
        self.coordinate_history.append(new_position)

        # Maximale Länge des Verlaufs begrenzen
        if len(self.coordinate_history) > 20:  # Beispiel: Maximal 20 Positionen in der Historie speichern
            self.coordinate_history.pop(0)
