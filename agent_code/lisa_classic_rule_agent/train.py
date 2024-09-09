import pickle
import os
import numpy as np
from .callbacks import state_to_features
import events as e
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    self.logger.info("Setting up training variables.")
    self.q_table = {}
    self.alpha = 0.1  # Lernrate
    self.gamma = 0.9  # Diskontfaktor
    self.epsilon = 0.97  # Epsilon für epsilon-greedy
    self.epsilon_decay = 0.995  # Epsilon-Decay
    self.epsilon_min = 0.1
    self.last_positions = []
    self.last_actions = []
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    self.logger.info(f"Action {self_action} led to game events {', '.join(map(str, events))}")
    
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    old_state = tuple(old_state)
    new_state = tuple(new_state)

    # Symmetrische Zustände für den alten und neuen Zustand berechnen
    symmetric_old_states = get_symmetric_states(old_state)
    symmetric_new_states = get_symmetric_states(new_state)

    # Q-Werte für symmetrische Zustände initialisieren, falls sie noch nicht existieren
    for state in symmetric_old_states:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
    for state in symmetric_new_states:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
    
    action_idx = ACTIONS.index(self_action)
    
    # Bewegung erkennen
    current_position = new_game_state['self'][3]
    if len(self.last_positions) >= 3 and current_position in self.last_positions[-2:]:
        events.append("STUCK_IN_LOOP")
    
    # Position und Aktion speichern
    self.last_positions.append(current_position)
    self.last_actions.append(self_action)

    # Erkennen von wiederholten Positionen
    if len(self.last_positions) >= 3:
        if current_position == self.last_positions[-2]:
            self.q_table[old_state][action_idx] -= 0.5  # Q-Wert verringern, wenn zur vorherigen Position zurückgekehrt wird

    if self_action == 'BOMB' and "INVALID_ACTION" not in events:
        self.bomb_history.append(current_position)

    # Maximale Länge des Verlaufs begrenzen
    if len(self.last_positions) > 4:
        self.last_positions.pop(0)
        self.last_actions.pop(0)

    reward = reward_from_events(self, events, old_game_state, new_game_state, self_action)
    
    # Kisten und Münzen zählen
    if e.CRATE_DESTROYED in events:
        self.destroyed_crates += 1
    if e.COIN_COLLECTED in events:
        self.collected_coins += 1
    
    # Schritte zählen
    self.steps_survived += 1
    
    
    # Bestrafung für unnötiges Bombenlegen
    if self_action == 'BOMB':
        x, y = old_game_state['self'][3]
        field = old_game_state['field']
        others = [other[3] for other in old_game_state['others']]
        
        # Überprüfen, ob sich Kisten oder Gegner in der Nähe befinden
        close_crates = any([bfs(field, (x, y), (cx, cy)) <= 1 for cx, cy in np.argwhere(field == 1)])
        close_enemies = any([bfs(field, (x, y), enemy) <= 1 for enemy in others])
        
        if not close_crates and not close_enemies:
            events.append("USELESS_BOMB_PLACEMENT")

    # SARSA Update-Regel
    if new_game_state is not None:
        next_action = np.argmax(self.q_table[new_state])
        for idx, (old_sym_state, new_sym_state) in enumerate(zip(symmetric_old_states, symmetric_new_states)):
            old_q_value = self.q_table[old_sym_state][action_idx]
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * self.q_table[new_sym_state][next_action] - old_q_value)
            
            # Q-Tabelle aktualisieren
            self.q_table[old_sym_state][action_idx] = new_q_value

            # Log der Q-Tabellen-Einträge
            if idx == 0:
                self.logger.info(f"Updated Q-value for ORIGINAL state {old_sym_state}: {self.q_table[old_sym_state]}")
            else:
                self.logger.info(f"Updated Q-value for symmetric state {old_sym_state}: {self.q_table[old_sym_state]}")
    else:
        # Terminalzustand
        for idx, old_sym_state in enumerate(symmetric_old_states):
            old_q_value = self.q_table[old_sym_state][action_idx]
            new_q_value = old_q_value + self.alpha * (reward - old_q_value)

            # Q-Tabelle aktualisieren
            self.q_table[old_sym_state][action_idx] = new_q_value

            # Log der Q-Tabellen-Einträge für Terminalzustand
            if idx == 0:
                self.logger.info(f"Updated Q-value (Terminal) for ORIGINAL state {old_sym_state}: {self.q_table[old_sym_state]}")
            else:
                self.logger.info(f"Updated Q-value (Terminal) for symmetric state {old_sym_state}: {self.q_table[old_sym_state]}")

    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    self.logger.debug(f'Events at end of round: {", ".join(map(str, events))}')
    last_state = state_to_features(last_game_state)
    last_state = tuple(last_state)
    
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)
    
    reward = reward_from_events(self, events, last_game_state, None, self.last_action)

    # End of round Q-Update
    self.q_table[last_state][action_idx] = self.q_table[last_state][action_idx] + \
                                           self.alpha * (reward - self.q_table[last_state][action_idx])

    model_file = "my-sarsa-model.pt"
    with open(model_file, "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info(f"Q-table saved to {model_file}.")


    # Ausgabe der Statistiken für die Runde
    self.logger.info(f"End of round {self.current_round}:")
    self.logger.info(f" - Crates destroyed: {self.destroyed_crates}")
    self.logger.info(f" - Coins collected: {self.collected_coins}")
    self.logger.info(f" - Steps survived: {self.steps_survived}")

    # Zurücksetzen der Zähler für die nächste Runde
    self.destroyed_crates = 0
    self.collected_coins = 0
    self.steps_survived = 0
    self.current_round += 1


def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict, self_action: str) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 10,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.7,
        e.BOMB_DROPPED: - 5, 
        e.INVALID_ACTION: -1,
        e.GOT_KILLED: -200,
        e.KILLED_SELF: -300,
        "STUCK_IN_LOOP": -10,
        "BOMB_PLACED_NEAR_BOXES": 1,
        "ESCAPED_BOMB": 1,
        "ENTERED_DANGER": -4,
        "USELESS_BOMB_PLACEMENT": -50
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])
    
    # Zusätzliche Belohnung für Annäherung an Münzen
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        if new_game_state['coins']:
            # Überprüfen, ob es Münzen im alten und neuen Zustand gibt
            if old_game_state['coins']:
                old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(coin)) for coin in old_game_state['coins']])
            else:
                old_min_dist = float('inf')  # Standardwert, wenn keine Münzen vorhanden sind

            if new_game_state['coins']:
                new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(coin)) for coin in new_game_state['coins']])
            else:
                new_min_dist = float('inf')  # Standardwert, wenn keine Münzen vorhanden sind

            if new_min_dist < old_min_dist:
                reward_sum += 10
            else:
                reward_sum -= 0.5  # Bestrafung für sich von Münzen entfernen
        
                # Bonus für das sichere Platzieren einer Bombe

        if "BOMB_PLACED_NEAR_BOXES" in events and "ESCAPED_BOMB" in events:
            reward_sum += 2

        # Bestrafung, wenn der Agent in den Explosionsradius einer Bombe läuft
        if "ENTERED_DANGER" in events:
            reward_sum -= 4
        # Belohnung für Entkommen aus Explosionsradius nach Bombenlegen
        if self_action == 'BOMB' and "INVALID_ACTION" not in events:
            reward_sum += 0.5  # Erfolgreiches Entkommen nach Bombenlegen

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def rotate_state(state, rotation):
    """Rotiert den Zustand um 90, 180 oder 270 Grad"""
    dx, dy, *other_features = state
    if rotation == 90:
        return (-dy, dx, *other_features)
    elif rotation == 180:
        return (-dx, -dy, *other_features)
    elif rotation == 270:
        return (dy, -dx, *other_features)
    return state

def mirror_state(state):
    """Spiegelt den Zustand entlang der Y-Achse"""
    dx, dy, *other_features = state
    return (-dx, dy, *other_features)

def get_symmetric_states(state):
    """Gibt alle symmetrischen Zustände zurück"""
    states = [state]
    for rotation in [90, 180, 270]:
        states.append(rotate_state(state, rotation))
    states.append(mirror_state(state))
    return states
