import pickle
import os
import numpy as np
from .callbacks import state_to_features
from .callbacks import is_safe_position
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up training variables.")
    # Q-table initialisieren
    self.q_table = {}
    self.alpha = 0.2  # Lernrate davor 0.1
    self.gamma = 0.95  # Diskontfaktor um so höher um so stätrker werden zufällige gewinne belohnt
    self.epsilon = 1.0  # Epsilon für epsilon-greedy 
    self.epsilon_decay = 0.995  # Epsilon-Decay für Exploration-Exploitation Tradeoff
    self.epsilon_min = 0.1

    self.last_positions = []
    self.last_actions = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Events: {", ".join(map(str, events))}')

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    old_state = tuple(old_state)
    new_state = tuple(new_state)

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

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
    
    # Belohnung für das Legen einer Bombe in der Nähe von Kisten
    if self_action == 'BOMB' and any([np.linalg.norm(np.array(old_game_state['self'][3]) - np.array(box)) <= 1 for box in np.argwhere(old_game_state['field'] == 1)]) and is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("BOMB_PLACED_NEAR_BOXES")

    # Belohnung für keinen selbstmord
    if self_action == 'BOMB' and is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("SAFE_BOMB_PLACEMENT")

    # Bestrafung für risiko bomben legen
    elif self_action == 'BOMB' and not is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("UNSAFE_BOMB_PLACEMENT")
    
    if not has_made_progress(old_game_state, new_game_state):
        events.append("NO_PROGRESS")

    # Bewegung erkennen und mögliche Schleifen vermeiden
    current_position = new_game_state['self'][3]
    if len(self.last_positions) >= 3 and current_position in self.last_positions[-2:]:
        events.append("STUCK_IN_LOOP")

    # Bomben-Timer und Sicherheitsüberprüfung integrieren
    bomb_timers = get_bomb_timers(new_game_state)
    current_position = new_game_state['self'][3]

    if not is_safe_position(new_game_state, current_position, bomb_timers):
        events.append("MOVED_INTO_DANGER")
        
    # Position und Aktion speichern
    self.last_positions.append(current_position)
    self.last_actions.append(self_action)

    # Maximale Länge des Verlaufs begrenzen
    if len(self.last_positions) > 4:
        self.last_positions.pop(0)
        self.last_actions.pop(0)

    # Reward, jetzt mit alten und neuen Zuständen
    reward = reward_from_events(self, events, old_game_state, new_game_state)

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
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Events at end of round: {", ".join(map(str, events))}')

    last_state = state_to_features(last_game_state)
    last_state = tuple(last_state)
    
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)
    
    # Reward mit den letzten Zuständen
    reward = reward_from_events(self, events, last_game_state, None)

    # End of round Q-Update
    self.q_table[last_state][action_idx] = self.q_table[last_state][action_idx] + \
                                           self.alpha * (reward - self.q_table[last_state][action_idx])

    # Modell speichern
    model_file = "my-sarsa-model.pt"
    with open(model_file, "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info(f"Q-table saved to {model_file}.")

def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5.0,
        e.KILLED_OPPONENT: 10.0,
        e.MOVED_UP: -0.05,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -1.0,
        e.GOT_KILLED: -5.0,
        e.KILLED_SELF: -50.0,
        #"STUCK_IN_LOOP": -5.0,
        "BOMB_PLACED_NEAR_BOXES": 2.0,
        "BOX_DESTROYED": 4.0,
        "ESCAPED_BOMB": 3.0,
        "SAFE_BOMB_PLACEMENT": 5.0,
        "UNSAFE_BOMB_PLACEMENT": -5.0, 
        "NO_PROGRESS": -1.0, 
        "MOVED_INTO_DANGER": -10.0 
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    # Annäherung an Münzen und Kisten belohnen
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Belohnung für das Annähern an eine Kiste
        if new_game_state['field'][new_position] == 1:
            old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(box)) for box in np.argwhere(old_game_state['field'] == 1)])
            new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(box)) for box in np.argwhere(new_game_state['field'] == 1)])
            if new_min_dist < old_min_dist:
                reward_sum += 0.5
            else:
                reward_sum -= 0.5


        # Flucht aus der Bombenreichweite belohnen
        old_bomb_distances = [np.linalg.norm(np.array(old_position) - np.array(bomb[0])) for bomb in old_game_state['bombs']]
        new_bomb_distances = [np.linalg.norm(np.array(new_position) - np.array(bomb[0])) for bomb in new_game_state['bombs']]

        if old_bomb_distances and new_bomb_distances:  # Überprüfen, ob beide Listen nicht leer sind
            if min(new_bomb_distances) > min(old_bomb_distances):
                reward_sum += 1.0  # Belohnung für das Entkommen aus der Bombenreichweite
            else:
                reward_sum -= 1.0

    if "BOMB_PLACED_NEAR_BOXES" in events and "ESCAPED_BOMB" in events:
        reward_sum += game_rewards["SAFE_BOMB_PLACEMENT"]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def is_safe_to_place_bomb(game_state, position):
    """Check if it is safe to place a bomb at the given position."""
    bombs = game_state['bombs']
    x, y = position
    field = game_state['field']  # Hier 'field' anstatt 'arena' verwenden
    directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    for d in directions:
        if field[d] == 0 and is_safe_after_bomb(d, bombs):  # Funktion zum Prüfen der Sicherheit
            return True
    return False



def is_safe_after_bomb(game_state, position):
    """Check if the agent is safe after placing a bomb at the given position."""
    return len(get_safe_positions_after_bomb(game_state, position)) > 0

def get_safe_positions_after_bomb(game_state, bomb_position):
    """Get all safe positions the agent can move to after placing a bomb."""
    if not bomb_position or len(bomb_position) != 2:
        return []  # Falls bomb_position ungültig ist, gebe eine leere Liste zurück

    x, y = bomb_position
    field = game_state['field']
    explosion_map = game_state['explosion_map']

    directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    safe_positions = []

    for nx, ny in directions:
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and explosion_map[nx, ny] == 0:
                safe_positions.append((nx, ny))

    return safe_positions

def has_made_progress(old_game_state, new_game_state):
    """Überprüft, ob der Agent Fortschritt macht (z.B. sich einer Münze, Kiste oder einem anderen Ziel nähert)."""
    if old_game_state is None or new_game_state is None:
        return True 

    old_position = np.array(old_game_state['self'][3])
    new_position = np.array(new_game_state['self'][3])

    # Münzenfortschritt überprüfen
    old_coin_distances = [np.linalg.norm(old_position - np.array(coin)) for coin in old_game_state['coins']]
    new_coin_distances = [np.linalg.norm(new_position - np.array(coin)) for coin in new_game_state['coins']]

    if min(new_coin_distances, default=float('inf')) < min(old_coin_distances, default=float('inf')):
        return True  # näher an eine Münze

    # Fortschritt beim Annähern an Kisten überprüfen
    old_box_positions = np.argwhere(old_game_state['field'] == 1)  # Kisten sind durch 1 im Feld markiert
    new_box_positions = np.argwhere(new_game_state['field'] == 1)

    if old_box_positions.size > 0 and new_box_positions.size > 0:
        old_box_distances = [np.linalg.norm(old_position - box) for box in old_box_positions]
        new_box_distances = [np.linalg.norm(new_position - box) for box in new_box_positions]

        if min(new_box_distances, default=float('inf')) < min(old_box_distances, default=float('inf')):
            return True  # näher an eine Kiste

    # Fortschritt beim Entfernen von Bomben überprüfen
    old_bomb_distances = [np.linalg.norm(old_position - np.array(bomb[0])) for bomb in old_game_state['bombs']]
    new_bomb_distances = [np.linalg.norm(new_position - np.array(bomb[0])) for bomb in new_game_state['bombs']]

    if old_bomb_distances and new_bomb_distances:
        if min(new_bomb_distances) > min(old_bomb_distances):
            return True  # sich weiter von Bomben entfernt

    # Fortschritt beim Annähern an Gegner überprüfen
    old_enemy_positions = [enemy[3] for enemy in old_game_state['others']]
    new_enemy_positions = [enemy[3] for enemy in new_game_state['others']]

    if old_enemy_positions and new_enemy_positions:
        old_enemy_distances = [np.linalg.norm(old_position - np.array(enemy)) for enemy in old_enemy_positions]
        new_enemy_distances = [np.linalg.norm(new_position - np.array(enemy)) for enemy in new_enemy_positions]

        if min(new_enemy_distances, default=float('inf')) < min(old_enemy_distances, default=float('inf')):
            return True  # näher an einen Gegner

    return False  # Keinen Fortschritt gemacht

def get_bomb_timers(game_state):
    bombs = game_state['bombs']
    return [(bomb[0], bomb[1]) for bomb in bombs]



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