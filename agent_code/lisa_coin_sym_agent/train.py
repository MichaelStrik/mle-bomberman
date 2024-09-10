import pickle
import os
import numpy as np
from .callbacks import state_to_features
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
    self.alpha = 0.1  # Lernrate siehe SARSA ALg.
    self.gamma = 0.9  # Diskontfaktor siehe SARSA Alg.
    self.epsilon = 1.0  # Epsilon für epsilon-greedy
    self.epsilon_decay = 0.95  # Epsilon-Decay für Exploration-Exploitation Tradeoff
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

    action_idx = ACTIONS.index(self_action)
    
    # Bewegung erkennen
    current_position = new_game_state['self'][3]
    if len(self.last_positions) >= 3 and current_position in self.last_positions[-2:]:
        events.append("STUCK_IN_LOOP")
    
    # Position und Aktion speichern
    self.last_positions.append(current_position)
    self.last_actions.append(self_action)

    # Positionen in den letzten Zügen erkennen
    if len(self.last_positions) >= 3:
        if current_position == self.last_positions[-2]:
            self.q_table[old_state][action_idx] -= 0.5  # Q-Wert verringern, wenn zur vorherigen Position zurückgekehrt wird

    # Maximale Länge des Verlaufs begrenzen
    if len(self.last_positions) > 4:
        self.last_positions.pop(0)
        self.last_actions.pop(0)

    # Reward, jetzt mit alten und neuen Zuständen
    reward = reward_from_events(self, events, old_game_state, new_game_state)

    # SARSA Update-Regel !!!
    if new_game_state is not None:
        next_action = np.argmax(self.q_table[new_state])
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                              self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
    else:
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                              self.alpha * (reward - self.q_table[old_state][action_idx])

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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -1,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -10,
        "STUCK_IN_LOOP": -8
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])
    
    # Zusätzliche Belohnung basierend auf der Annäherung an Münzen
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        if new_game_state['coins']:
            old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(coin)) for coin in old_game_state['coins']])
            new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(coin)) for coin in new_game_state['coins']])

            # Belohnung für Annäherung an die nächste Münze
            if new_min_dist < old_min_dist:
                reward_sum += 0.5
        else:
            # Falls keine Münzen mehr vorhanden sind, kann man eine alternative Belohnung in Erwägung ziehen
            reward_sum += 0.1  # Zum Beispiel eine kleine Belohnung, da alle Münzen eingesammelt wurden

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

def mirror_state(state, axis='y'):
    """Spiegelt den Zustand entlang der angegebenen Achse ('x' oder 'y')"""
    dx, dy, *other_features = state
    if axis == 'x':
        return (-dx, dy, *other_features)
    elif axis == 'y':
        return (dx, -dy, *other_features)
    return state

def mirror_diagonal(state, diagonal='main'):
    dx, dy, *other_features = state
    if diagonal == 'main':
        return (dy, dx, *other_features)
    elif diagonal == 'secondary':
        return (-dy, -dx, *other_features)
    return state

def rotate_action(action, rotation):
    """Passt die Aktion ('UP', 'DOWN', 'LEFT', 'RIGHT') basierend auf der Rotation an."""
    action_map = {
        'UP': {90: 'RIGHT', 180: 'DOWN', 270: 'LEFT', 0: 'UP'},
        'RIGHT': {90: 'DOWN', 180: 'LEFT', 270: 'UP', 0: 'RIGHT'},
        'DOWN': {90: 'LEFT', 180: 'UP', 270: 'RIGHT', 0: 'DOWN'},
        'LEFT': {90: 'UP', 180: 'RIGHT', 270: 'DOWN', 0: 'LEFT'}
    }
    return action_map[action][rotation]


def mirror_action(action, axis):
    """Passt die Aktion ('UP', 'DOWN', 'LEFT', 'RIGHT') basierend auf der Spiegelung an."""
    if axis == 'x':  # Spiegelung an der x-Achse
        return {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}[action]
    elif axis == 'y':  # Spiegelung an der y-Achse
        return {'UP': 'UP', 'DOWN': 'DOWN', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}[action]
    return action

def transform_action(action, rotation=None, axis=None, diagonal=None):
    """Transformiere die Aktion basierend auf Rotation, Spiegelung oder Diagonalspiegelung."""
    if action in ['BOMB', 'WAIT']:
        return action

    action_map = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3
    }
    reverse_action_map = {v: k for k, v in action_map.items()}
    action_idx = action_map[action]

    if rotation:
        action_idx = (action_idx + rotation // 90) % 4
    
    if axis == 'x':
        if action == 'UP':
            action_idx = action_map['DOWN']
        elif action == 'DOWN':
            action_idx = action_map['UP']
    elif axis == 'y':
        if action == 'LEFT':
            action_idx = action_map['RIGHT']
        elif action == 'RIGHT':
            action_idx = action_map['LEFT']

    if diagonal == 'main':
        if action == 'UP':
            action_idx = action_map['RIGHT']
        elif action == 'RIGHT':
            action_idx = action_map['UP']
        elif action == 'DOWN':
            action_idx = action_map['LEFT']
        elif action == 'LEFT':
            action_idx = action_map['DOWN']
    elif diagonal == 'secondary':
        if action == 'UP':
            action_idx = action_map['LEFT']
        elif action == 'RIGHT':
            action_idx = action_map['DOWN']
        elif action == 'DOWN':
            action_idx = action_map['RIGHT']
        elif action == 'LEFT':
            action_idx = action_map['UP']

    return reverse_action_map[action_idx]

def get_symmetric_states_and_actions(state, action):
    """Gibt alle symmetrischen Zustände und die dazugehörigen symmetrischen Aktionen zurück."""
    states_actions = set()
    
    # Original Zustand und Aktion
    states_actions.add((state, action))
    
    # Rotationen um 90, 180 und 270 Grad
    for rotation in [0, 90, 180, 270]:
        rotated_state = rotate_state(state, rotation)
        rotated_action = transform_action(action, rotation)
        states_actions.add((rotated_state, rotated_action))
    
    # Spiegelungen entlang der x- und y-Achse
    for axis in ['x', 'y']:
        mirrored_state = mirror_state(state, axis)
        mirrored_action = transform_action(action, axis=axis)
        states_actions.add((mirrored_state, mirrored_action))
        
        # Spiegelungen nach jeder Rotation
        for rotation in [0, 90, 180, 270]:
            rotated_state = rotate_state(state, rotation)
            rotated_mirrored_state = mirror_state(rotated_state, axis)
            rotated_mirrored_action = transform_action(action, rotation, axis=axis)
            states_actions.add((rotated_mirrored_state, rotated_mirrored_action))
    
    # Diagonalspiegelungen
    for diagonal in ['main', 'secondary']:
        diagonal_state = mirror_diagonal(state, diagonal)
        diagonal_action = transform_action(action, diagonal=diagonal)
        states_actions.add((diagonal_state, diagonal_action))
        
        for rotation in [0, 90, 180, 270]:
            rotated_state = rotate_state(state, rotation)
            
            rotated_diagonal_state = mirror_diagonal(rotated_state, diagonal)
            rotated_diagonal_action = transform_action(action, rotation, diagonal=diagonal)
            states_actions.add((rotated_diagonal_state, rotated_diagonal_action))

    #print(f"Total unique states_actions: {len(states_actions)}")
    return list(states_actions)