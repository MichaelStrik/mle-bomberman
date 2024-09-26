import pickle
import numpy as np
from .callbacks import ACTIONS, state_to_features, get_valid_actions, get_bomb_radius, is_safe_position, get_symmetric_states_and_actions
import events as e
from random import choice
from collections import deque
# Importieren der notwendigen Funktionen aus callbacks, um Duplikate zu vermeiden
from .callbacks import get_next_target, look_for_targets, is_safe_position, get_bomb_radius, apply_transformation, get_symmetric_states_and_actions

def setup_training(self):
    """
    Initialize self for training purposes.
    """
    self.alpha = 0.2  # Learning rate
    self.gamma = 0.95  # Discount factor
    self.epsilon = 1.0  # Starting epsilon for exploration
    self.epsilon_decay = 0.995  # Epsilon decay rate
    self.epsilon_min = 0.25  # Minimum epsilon
    self.target = None
    self.steps_since_last_bomb = 0
    self.last_positions = deque(maxlen=4)  # Speichern der letzten 4 Positionen zur Schleifenerkennung
    self.last_two_actions = deque(['WAIT', 'WAIT'], maxlen=2)  # Initialisiere mit 'WAIT'

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Update Q-values based on the events that occurred.
    """
    if old_game_state is None or new_game_state is None:
        return

    old_state = state_to_features(self, old_game_state)
    new_state = state_to_features(self, new_game_state)

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    if self_action==None:
        #print('action was none')
        self_action='WAIT'

    action_idx = ACTIONS.index(self_action)

    # Check for bombs in agent's radius
    bombs = new_game_state['bombs']
    agent_pos = new_game_state['self'][3]
    arena = new_game_state['field']
    bomb_in_radius = False

    for (bx, by), timer in bombs:
        affected_tiles = get_bomb_radius((bx, by), arena)
        if agent_pos in affected_tiles:
            bomb_in_radius = True
            break

    # Update target accordingly
    if bomb_in_radius:
        self.target = find_next_safe_position(new_game_state)
    else:
        # Normal target selection (coins or crates)
        self.target = get_next_target(new_game_state)

    # --- Compute additional events based on game state transitions ---

    # Update last positions
    new_position = new_game_state['self'][3]
    self.last_positions.append(new_position)

    # Check if agent moved into danger
    if not is_safe_position(new_game_state, new_position, new_game_state['bombs']):
        events.append("MOVED_INTO_DANGER")
    else:
        events.append("AVOIDED_DANGER")

    # Reward for moving towards target
    if self.target is not None:
        old_position = old_game_state['self'][3]
        old_distance = bfs_distance(old_game_state['field'], old_position, self.target)
        new_distance = bfs_distance(new_game_state['field'], new_position, self.target)
        if new_distance < old_distance:
            events.append("MOVED_TOWARDS_TARGET")
        else:
            events.append("MOVED_AWAY_FROM_TARGET")

    # Penalize if not moving towards safety when bomb is ticking
    if bomb_in_radius and self.target is not None:
        old_distance = bfs_distance(old_game_state['field'], old_position, self.target)
        if new_distance >= old_distance:
            events.append("NOT_MOVING_TOWARDS_SAFETY")

    # Penalize if agent returns to a previous position
    if new_position in list(self.last_positions)[:-1]:
        events.append("REVISITED_POSITION")

    # Detect bomb placement near crate or opponent
    if self_action == 'BOMB':
        x, y = old_game_state['self'][3]
        arena = old_game_state['field']
        # Check for crates in bomb radius
        bomb_radius = get_bomb_radius((x, y), arena)
        crates_nearby = any(arena[cx, cy] == 1 for cx, cy in bomb_radius)
        opponent_nearby = any((cx, cy) in [opponent[3] for opponent in old_game_state['others']] for cx, cy in bomb_radius)
        if crates_nearby:
            events.append("BOMB_PLACED_NEAR_CRATE")
        if opponent_nearby:
            events.append("BOMB_PLACED_NEAR_OPPONENT")

    # --- End of additional events computation ---

    # Compute reward
    reward = reward_from_events(self, events)

    # Next action selection (for SARSA)
    valid_actions = get_valid_actions(new_game_state, self)
    if np.random.rand() < self.epsilon:
        next_action = choice(valid_actions)
    else:
        q_values = self.q_table[new_state]
        valid_q_indices = [ACTIONS.index(a) for a in valid_actions]
        valid_q_values = q_values[valid_q_indices]
        max_q = np.max(valid_q_values)
        best_actions = [valid_actions[i] for i in range(len(valid_actions)) if valid_q_values[i] == max_q]
        next_action = choice(best_actions)

    next_action_idx = ACTIONS.index(next_action)

    # SARSA update
    self.q_table[old_state][action_idx] += self.alpha * (
        reward + self.gamma * self.q_table[new_state][next_action_idx] - self.q_table[old_state][action_idx]
    )

    # Apply symmetric transformations
    symmetries = get_symmetric_states_and_actions(old_state, self_action)
    for sym_state, sym_action in symmetries:
        sym_action_idx = ACTIONS.index(sym_action)
        if sym_state not in self.q_table:
            self.q_table[sym_state] = np.zeros(len(ACTIONS))
        self.q_table[sym_state][sym_action_idx] = self.q_table[old_state][action_idx]

    # Epsilon decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    # Update last two actions
    self.last_two_actions.append(self_action)

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each game to finalize training.
    """
    last_state = state_to_features(self, last_game_state)
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)

    # Since there is no next state, the Q-value update simplifies
    reward = reward_from_events(self, events)

    self.q_table[last_state][action_idx] += self.alpha * (reward - self.q_table[last_state][action_idx])

    # Save the Q-table
    with open("my-sarsa-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info("Q-table saved to my-sarsa-model.pt.")

def reward_from_events(self, events: list) -> float:
    """
    Compute the reward based on events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10.0,
        e.KILLED_OPPONENT: 50.0,
        e.INVALID_ACTION: -5.0,
        e.GOT_KILLED: -50.0,
        e.KILLED_SELF: -100.0,
        e.BOMB_DROPPED: 5.0,
        "BOMB_PLACED_NEAR_CRATE": 15.0,
        "BOMB_PLACED_NEAR_OPPONENT": 30.0,
        "MOVED_TOWARDS_TARGET": 5.0,
        "MOVED_AWAY_FROM_TARGET": -5.0,
        "MOVED_INTO_DANGER": -20.0,
        "AVOIDED_DANGER": 10.0,
        "REVISITED_POSITION": -10.0,  # Bestrafung für das Zurückkehren zu einer vorherigen Position
        "WAITED": -5.0
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    return reward_sum

def find_next_safe_position(game_state):
    """
    Finde die nächste sichere Position für den Agenten, wenn eine Bombe in der Nähe ist.
    """
    x, y = game_state['self'][3]
    arena = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    unsafe_tiles = set()

    # Markiere Felder, die von Bomben betroffen sind
    for (bx, by), timer in bombs:
        affected_tiles = get_bomb_radius((bx, by), arena)
        unsafe_tiles.update(affected_tiles)

    # BFS, um die nächste sichere Position zu finden
    from collections import deque
    queue = deque()
    visited = set()
    queue.append(((x, y), 0))
    visited.add((x, y))

    while queue:
        (cx, cy), dist = queue.popleft()
        if (cx, cy) not in unsafe_tiles and explosion_map[cx, cy] <= 0:
            return (cx, cy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and
                    arena[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))
    return (x, y)  # Keine sichere Position gefunden; aktuelle Position zurückgeben

def bfs_distance(arena, start, target):
    """
    Berechnet die kürzeste Pfaddistanz von Start zu Ziel mittels BFS.
    """
    if start == target:
        return 0
    from collections import deque
    queue = deque()
    visited = set()
    queue.append((start, 0))
    visited.add(start)

    while queue:
        position, distance = queue.popleft()
        x, y = position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and
                    arena[nx, ny] == 0 and (nx, ny) not in visited):
                if (nx, ny) == target:
                    return distance + 1
                queue.append(((nx, ny), distance + 1))
                visited.add((nx, ny))
    return float('inf')  # Wenn das Ziel nicht erreichbar ist

