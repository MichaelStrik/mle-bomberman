import os
import pickle
import numpy as np
from random import choice
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    # Laden der Q-Tabelle, falls vorhanden
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        # Initialisiere eine leere Q-Tabelle
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")

    self.epsilon = 0.1   # Epsilon für die epsilon-greedy Strategie
    self.alpha = 0.2     # Lernrate
    self.gamma = 0.95    # Diskontierungsfaktor
    self.target = None
    self.evading_bomb = False  # Flag, um anzuzeigen, ob der Agent einer Bombe ausweicht
    self.next_random = np.random.rand() 

def act(self, game_state: dict) -> str:
    """
    Entscheidet über die nächste Aktion basierend auf dem aktuellen Spielzustand.
    """
    # Zustandsrepräsentation erstellen
    features = state_to_features(self, game_state)
    state = features

    # Initialisiere Q-Werte für unbekannte Zustände
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    q_values = self.q_table[state]

    valid_actions = get_valid_actions(game_state)

    random = self.next_random
    # Epsilon-greedy Aktionsauswahl
    if random < self.epsilon:
        action = choice(valid_actions)
    else:
        # Wähle die Aktion mit dem höchsten Q-Wert unter den gültigen Aktionen
        valid_q_indices = [ACTION_TO_INDEX[a] for a in valid_actions]
        valid_q_values = q_values[valid_q_indices]
        max_q = np.max(valid_q_values)
        best_actions = [valid_actions[i] for i, q in enumerate(valid_q_values) if q == max_q]
        action = choice(best_actions)
    
    self.next_random = np.random.rand()
    
    return action

def state_to_features(self, game_state):
    """
    Wandelt den Spielzustand in einen Zustandsvektor für die Q-Tabelle um.
    """
    if game_state is None:
        return None

    field = game_state['field']
    own_position = game_state['self'][3]
    x, y = own_position

    # Umgebung des Agenten (5x5 Gitter um den Agenten)
    surroundings = field[x - 2:x + 3, y - 2:y + 3]
    # Falls Rand erreicht wird, mit -1 (Wand) auffüllen
    surroundings = np.pad(surroundings, pad_width=((max(0, 2 - x), max(0, x + 3 - field.shape[0])),
                                                   (max(0, 2 - y), max(0, y + 3 - field.shape[1]))),
                          mode='constant', constant_values=-1)
    surroundings = tuple(surroundings.flatten())

    # Relative Position zum aktuellen Ziel
    if self.target is not None:
        dx, dy = self.target[0] - x, self.target[1] - y
    else:
        dx, dy = 0, 0

    features = surroundings + (dx, dy)
    return features

def get_valid_actions(game_state):
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
                arena[nx, ny] == 0 and  # Freies Feld
                explosion_map[nx, ny] <= 0 and  # Keine bevorstehende Explosion
                (nx, ny) not in bomb_positions and  # Keine Bombe
                (nx, ny) not in others):  # Kein anderer Agent
            valid_actions.append(action_map[(dx, dy)])

    # 'WAIT' erlauben, wenn es sicher ist
    if explosion_map[x, y] <= 0 and (x, y) not in bomb_positions:
        valid_actions.append('WAIT')
    valid_actions.append('BOMB')

    if not valid_actions:
        valid_actions.append('WAIT')
    return valid_actions

def is_crate_in_bomb_range(game_state):
    """
    Überprüft, ob sich eine Kiste im Explosionsradius einer Bombe befindet.
    """
    x, y = game_state['self'][3]
    arena = game_state['field']
    explosion_radius = get_bomb_radius((x, y), arena)
    for (cx, cy) in explosion_radius:
        if arena[cx, cy] == 1:  # 1 repräsentiert eine Kiste
            return True
    return False

def get_bomb_radius(bomb_pos, arena):
    """
    Gibt die von der Bombe betroffenen Felder zurück, unter Berücksichtigung von Wänden und Kisten.
    """
    x_bomb, y_bomb = bomb_pos
    radius = [(x_bomb, y_bomb)]  # Bombe selbst

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

def is_safe_position(game_state, position):
    """
    Überprüft, ob eine Position sicher ist, unter Berücksichtigung von Bomben und deren Explosionsradius.
    """
    x, y = position
    arena = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # Überprüfe, ob die Position von aktuellen Explosionen betroffen ist
    if explosion_map[x, y] > 0:
        return False

    # Überprüfe, ob die Position von tickenden Bomben bedroht ist
    for (bx, by), _ in bombs:
        affected_tiles = get_bomb_radius((bx, by), arena)
        if (x, y) in affected_tiles:
            return False
    return True

def find_safe_position(game_state):
    """
    Findet die nächste sichere Position, die nicht im Explosionsradius einer Bombe liegt.
    """
    from collections import deque

    arena = game_state['field']
    x, y = game_state['self'][3]
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = [other[3] for other in game_state['others']]

    queue = deque()
    visited = set()
    queue.append((x, y))
    visited.add((x, y))

    while queue:
        cx, cy = queue.popleft()
        if is_safe_position(game_state, (cx, cy)):
            return (cx, cy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and
                    (nx, ny) not in visited and arena[nx, ny] == 0 and
                    (nx, ny) not in others):
                queue.append((nx, ny))
                visited.add((nx, ny))
    # Falls keine sichere Position gefunden wurde, bleibt der Agent stehen
    return (x, y)
