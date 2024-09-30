import pickle
import numpy as np
import events as e
from random import choice
from collections import deque
from .callbacks import ACTIONS, ACTION_TO_INDEX, state_to_features, is_crate_in_bomb_range,  get_valid_actions, get_bomb_radius, bfs_distance

def setup_training(self):
    """
    Initialisiert die Trainingsparameter.
    """
    self.alpha = 0.1  # Lernrate
    self.gamma = 0.1  # Diskontierungsfaktor
    self.epsilon = 0.3  # Epsilon für Exploration

    self.target = None
    self.evading_bomb = False  # Flag, um anzuzeigen, ob der Agent einer Bombe ausweicht

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Aktualisiert die Q-Werte basierend auf den aufgetretenen Ereignissen.
    """
    if old_game_state is None or new_game_state is None:
        return

    # Handle None self_action
    if self_action is None:
        self_action = 'WAIT'

    old_state = state_to_features(self, old_game_state)
    new_state = state_to_features(self, new_game_state)

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    action_idx = ACTION_TO_INDEX[self_action]

    # Überprüfen, ob die letzte Aktion eine Bombe war
    if self_action == 'BOMB':
        # Nächstes Ziel ist ein sicherer Ort
        self.evading_bomb = True
        self.target = find_safe_location(new_game_state)
    elif self.evading_bomb:
        # Prüfen, ob die Bombe nicht mehr gefährlich ist
        if is_bomb_danger_over(self, new_game_state):
            self.evading_bomb = False
            # Nächstes Ziel bestimmen
            self.target = get_next_target(self, new_game_state)
        else:
            # Weiterhin zum sicheren Ort bewegen
            pass  # self.target bleibt gleich
    else:
        # Normales Ziel bestimmen
        self.target = get_next_target(self, new_game_state)

    # Zusätzliche Ereignisse basierend auf dem Zustandsübergang
    new_position = new_game_state['self'][3]
    old_position = old_game_state['self'][3]
    if self.target is not None:
        old_distance = bfs_distance(old_game_state['field'], old_position, self.target)
        new_distance = bfs_distance(new_game_state['field'], new_position, self.target)
        if new_distance < old_distance:
            events.append("MOVED_TOWARDS_TARGET")
        elif new_distance > old_distance:
            events.append("MOVED_AWAY_FROM_TARGET")

    # Bombe in der Nähe einer Kiste platziert
    if self_action == 'BOMB' and is_crate_in_bomb_range(old_game_state):
        events.append("BOMB_PLACED_NEAR_CRATE")

    # Belohnung berechnen
    reward = reward_from_events(self, events)

    # SARSA-Update
    self.q_table[old_state][action_idx] += self.alpha * (
        reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[old_state][action_idx]
    )

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Wird am Ende jeder Runde aufgerufen, um das Training zu finalisieren.
    """
    if last_action is None:
        last_action = 'WAIT'

    last_state = state_to_features(self, last_game_state)
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTION_TO_INDEX[last_action]

    reward = reward_from_events(self, events)

    # Da es keine nächste Aktion gibt, verwenden wir nur die Belohnung
    self.q_table[last_state][action_idx] += self.alpha * (reward - self.q_table[last_state][action_idx])

    # Q-Tabelle speichern
    with open("my-sarsa-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info("Q-table saved to my-sarsa-model.pt.")

def reward_from_events(self, events: list) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 50.0,
        e.KILLED_OPPONENT: 100.0,
        e.INVALID_ACTION: -10.0,
        e.GOT_KILLED: -100.0,
        e.KILLED_SELF: -200.0,
        "MOVED_TOWARDS_TARGET": 10.0,
        "MOVED_AWAY_FROM_TARGET": -10.0,
        "BOMB_PLACED_NEAR_CRATE": 5.0,
        e.WAITED: -0.5,
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    return reward_sum

def is_bomb_danger_over(self, game_state):
    """
    Überprüft, ob die Bombengefahr vorüber ist (d.h., ob es keine aktiven Explosionen oder tickenden Bomben gibt).
    """
    # Wenn es keine tickenden Bomben oder Explosionen mehr gibt, ist die Gefahr vorüber
    if len(game_state['bombs']) == 0 and np.all(game_state['explosion_map'] == 0):
        return True
    else:
        return False

def find_safe_location(game_state):
    """
    Findet die nächste sichere Position, die nicht im Explosionsradius einer Bombe liegt.
    Wände und Kisten blockieren die Explosion.
    """
    field = game_state['field']
    own_position = game_state['self'][3]
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # Positionen markieren, die gefährlich sind
    dangerous_tiles = set()
    for bomb_pos, bomb_timer in bombs:
        bomb_radius = get_bomb_radius( bomb_pos, field)
        dangerous_tiles.update(bomb_radius)

    # Aktuelle Explosionen hinzufügen
    x_max, y_max = field.shape
    for x in range(x_max):
        for y in range(y_max):
            if explosion_map[x, y] > 0:
                dangerous_tiles.add((x, y))

    # BFS, um die nächste sichere Position zu finden
    queue = deque()
    queue.append((own_position, 0))
    visited = set()
    visited.add(own_position)

    while queue:
        current_pos, distance = queue.popleft()
        if current_pos not in dangerous_tiles:
            return current_pos  # Sichere Position gefunden

        x, y = current_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (x + dx, y + dy)
            if (0 <= next_pos[0] < x_max) and (0 <= next_pos[1] < y_max):
                if is_free(field[next_pos]) and next_pos not in visited:
                    queue.append((next_pos, distance + 1))
                    visited.add(next_pos)

    # Keine sichere Position gefunden, am aktuellen Ort bleiben
    return own_position

def get_next_target(self, game_state):
    """
    Bestimmt das nächste Ziel für den Agenten.
    - Wenn es Münzen gibt, gehe zur nächsten Münze.
    - Wenn es Kisten gibt, gehe zu einer Kiste, wo es sicher ist, eine Bombe zu platzieren.
    - Ansonsten None.
    """
    field = game_state['field']
    coins = game_state['coins']
    own_position = game_state['self'][3]
    others = [agent[3] for agent in game_state['others']]

    # Wenn es Münzen gibt, gehe zur nächsten
    if len(coins) > 0:
        coin_distances = []
        for coin in coins:
            distance = bfs_distance(field, own_position, coin)
            if distance is not None:
                coin_distances.append((distance, coin))
        if coin_distances:
            coin_distances.sort()
            return coin_distances[0][1]

    # Wenn es Kisten gibt, gehe zu einer sicheren Kiste
    crates = np.argwhere(field == 1)
    if len(crates) > 0:
        safe_crate_positions = []
        for crate in crates:
            crate_pos = tuple(crate)
            if is_safe_to_place_bomb(field, own_position, crate_pos, others):
                distance = bfs_distance(field, own_position, crate_pos)
                if distance is not None:
                    safe_crate_positions.append((distance, crate_pos))
        if safe_crate_positions:
            safe_crate_positions.sort()
            return safe_crate_positions[0][1]

    # Kein Ziel gefunden
    return None

def is_safe_to_place_bomb(field, own_position, crate_position, others):
    """
    Überprüft, ob es sicher ist, an der gegebenen Kistenposition eine Bombe zu platzieren.
    """
    x, y = crate_position
    x_max, y_max = field.shape
    safe_spot_found = False
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_pos = (x + dx, y + dy)
        if (0 <= next_pos[0] < x_max) and (0 <= next_pos[1] < y_max):
            if is_free(field[next_pos]) and next_pos != own_position and next_pos not in others:
                safe_spot_found = True
                break
    return safe_spot_found

def is_free(tile):
    """
    Überprüft, ob ein Feld frei begehbar ist.
    """
    return tile == 0  # 0: frei, -1: Wand, 1: Kiste
