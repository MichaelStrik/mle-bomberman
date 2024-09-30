import pickle
import numpy as np
import events as e
from random import choice
from .callbacks import ACTIONS, ACTION_TO_INDEX, state_to_features, get_valid_actions, get_bomb_radius, is_safe_position, is_crate_in_bomb_range, get_next_target, bfs_distance

def setup_training(self):
    """
    Initialisiert die Trainingsparameter.
    """
    self.alpha = 0.05  # Lernrate
    self.gamma = 0.95  # Diskontierungsfaktor
    self.epsilon = 0.3  # Epsilon für Exploration

    self.target = None

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

    # Ziel aktualisieren
    self.target = get_next_target(new_game_state)

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
    # Handle None self_action
    if self_action is None:
        self_action = 'WAIT'

    # Überprüfen, ob `self.next_action` vorhanden ist
    if not hasattr(self, 'next_action'):
        self.next_action = 'WAIT'  # Standardaktion, falls nicht gesetzt

    old_state = state_to_features(self, old_game_state)
    new_state = state_to_features(self, new_game_state)

    # Initialisierung der Q-Werte für unbekannte Zustände
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    action_idx = ACTION_TO_INDEX[self_action]
    next_action_idx = ACTION_TO_INDEX[self.next_action]

    # Belohnung berechnen
    reward = reward_from_events(self, events)

    # SARSA-Update mit der tatsächlich ausgeführten nächsten Aktion
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
    game_rewards = {
        e.COIN_COLLECTED: 50.0,
        e.KILLED_OPPONENT: 100.0,
        e.INVALID_ACTION: -10.0,
        e.GOT_KILLED: -100.0,
        e.KILLED_SELF: -200.0,
        "MOVED_TOWARDS_TARGET": 10.0,
        "MOVED_AWAY_FROM_TARGET": -10.0,
        "BOMB_PLACED_NEAR_CRATE": 20.0,
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    return reward_sum
