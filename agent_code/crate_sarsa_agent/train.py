import pickle
import numpy as np
import events as e
from random import choice
from .callbacks import ACTIONS, ACTION_TO_INDEX, is_safe_position, state_to_features, is_crate_in_bomb_range, get_valid_actions, get_bomb_radius, find_safe_position

def setup_training(self):
    """
    Initialisiert die Trainingsparameter.
    """
    self.alpha = 0.2  # Lernrate
    self.gamma = 0.95  # Diskontierungsfaktor
    self.epsilon = 0.1  # Epsilon für Exploration

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

    # Aktualisieren des Ziels
    if self.evading_bomb:
        # Prüfen, ob die Bombengefahr vorbei ist
        if is_safe_position(new_game_state, new_game_state['self'][3]):
            self.evading_bomb = False
            self.target = get_next_target(new_game_state)
    else:
        # Überprüfen, ob der Agent im Explosionsradius einer Bombe ist
        if not is_safe_position(new_game_state, new_game_state['self'][3]):
            self.evading_bomb = True
            self.target = find_safe_position(new_game_state)
        else:
            self.target = get_next_target(new_game_state)

    # Zusätzliche Ereignisse basierend auf dem Zustandsübergang
    new_position = new_game_state['self'][3]
    old_position = old_game_state['self'][3]
    if self.target is not None:
        old_distance = manhattan_distance(old_position, self.target)
        new_distance = manhattan_distance(new_position, self.target)
        if new_distance < old_distance:
            events.append("MOVED_TOWARDS_TARGET")
        elif new_distance > old_distance:
            events.append("MOVED_AWAY_FROM_TARGET")

    # Bombe in der Nähe einer Kiste platziert
    if self_action == 'BOMB':
        if is_crate_in_bomb_range(old_game_state) and is_safe_position(new_game_state, new_position):
            events.append("BOMB_PLACED_NEAR_CRATE")
        else:
            events.append("UNSAFE_BOMB_PLACED")

    # Belohnung berechnen
    reward = reward_from_events(self, events)

    # next action
    old_valid_actions = get_valid_actions(old_game_state)
    random = self.next_random
    old_q_values = self.q_table[old_state]
    if random < self.epsilon:
        next_action = choice(old_valid_actions)
    else:
        # Wähle die Aktion mit dem höchsten Q-Wert unter den gültigen Aktionen
        valid_q_indices = [ACTION_TO_INDEX[a] for a in old_valid_actions]
        valid_q_values = old_q_values[valid_q_indices]
        max_q = np.max(valid_q_values)
        best_actions = [old_valid_actions[i] for i, q in enumerate(valid_q_values) if q == max_q]
        next_action = choice(best_actions)
    

    # SARSA-Update mit der tatsächlich ausgeführten nächsten Aktion
    next_action_idx = ACTION_TO_INDEX[next_action] 
    self.q_table[old_state][action_idx] += self.alpha * (
        reward + self.gamma * self.q_table[new_state][next_action_idx] - self.q_table[old_state][action_idx]
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
    """
    Berechnet die Belohnung basierend auf den aufgetretenen Ereignissen.
    """
    game_rewards = {
        "MOVED_TOWARDS_TARGET": 10.0,
        "MOVED_AWAY_FROM_TARGET": -10.0,
        "BOMB_PLACED_NEAR_CRATE": 15.0,
        "UNSAFE_BOMB_PLACED": -20.0,
        e.WAITED: -1.0
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    return reward_sum

def get_next_target(game_state):
    """
    Bestimmt das nächste Ziel für den Agenten.
    - Wenn es Münzen gibt, gehe zur nächsten Münze.
    - Wenn es Kisten gibt, die sicher zerstört werden können, gehe zu einer solchen Kiste.
    - Ansonsten None.
    """
    field = game_state['field']
    coins = game_state['coins']
    own_position = game_state['self'][3]

    # Wenn es Münzen gibt, gehe zur nächsten
    if len(coins) > 0:
        distances = [manhattan_distance(own_position, coin) for coin in coins]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        return coins[min_index]

    # Wenn es Kisten gibt, die sicher erreicht werden können
    crates = np.argwhere(field == 1)
    safe_crates = []
    for crate in crates:
        crate_pos = tuple(crate)
        if is_reachable(game_state, own_position, crate_pos):
            safe_crates.append(crate_pos)
    if safe_crates:
        distances = [manhattan_distance(own_position, crate) for crate in safe_crates]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        return safe_crates[min_index]

    # Kein Ziel gefunden
    return None

def is_reachable(game_state, start, target):
    """
    Überprüft, ob das Ziel von der Startposition aus erreichbar ist.
    """
    from collections import deque

    arena = game_state['field']
    x_max, y_max = arena.shape
    queue = deque()
    visited = set()
    queue.append(start)
    visited.add(start)

    while queue:
        x, y = queue.popleft()
        if (x, y) == target:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < x_max and 0 <= ny < y_max and
                    arena[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append((nx, ny))
                visited.add((nx, ny))
    return False

def manhattan_distance(a, b):
    """
    Berechnet die Manhattan-Distanz zwischen zwei Punkten.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
