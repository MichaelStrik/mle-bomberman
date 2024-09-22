import pickle
import os
import numpy as np
from .callbacks import state_to_features, is_safe_position, get_bomb_timers,  get_valid_actions
from agent_code.rule_based_agent.callbacks import look_for_targets  
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Check whether a saved Q-table exists
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.2  # Learning rate
    self.gamma = 0.95  # Discount factor
    self.epsilon = 1.0  # Epsilon for epsilon-greedy strategy
    self.epsilon_decay = 0.995  # Epsilon decay
    self.epsilon_min = 0.1

    # Track last positions and actions to avoid loops
    self.last_positions = []
    self.last_actions = []
    self.target = None  # Keep track of the current target (coin or box)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Process game events, update Q-values based on new and old states, and assign rewards.
    """
    self.logger.debug(f'Events: {", ".join(map(str, events))}')

    old_state = tuple(state_to_features(old_game_state))
    new_state = tuple(state_to_features(new_game_state))

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(self_action)

    # Handle Bomb placement with added strategy and safety checks
    if self_action == 'BOMB' and is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("SAFE_BOMB_PLACEMENT")
        # Reward bombs placed near boxes
        if any([np.linalg.norm(np.array(old_game_state['self'][3]) - np.array(box)) <= 1 for box in np.argwhere(old_game_state['field'] == 1)]):
            events.append("BOMB_PLACED_NEAR_BOXES")

    elif self_action == 'BOMB' and not is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("UNSAFE_BOMB_PLACEMENT")
    
    # Track movement to prevent loops
    current_position = new_game_state['self'][3]
    if len(self.last_positions) >= 3 and current_position in self.last_positions[-2:]:
        events.append("STUCK_IN_LOOP")

    # Determine target and check progress
    self.target = get_next_target(new_game_state)

    if self.target is not None:
        # Prüfen, ob der Agent sich dem Ziel nähert
        if is_closer_to_target(old_game_state, new_game_state, self.target):
            events.append("MOVING_TOWARD_TARGET")
        else:
            events.append("NO_PROGRESS")
    else:
        events.append("NO_TARGET_FOUND")

    # Check if agent moved into danger
    bomb_timers = get_bomb_timers(new_game_state)
    if not is_safe_position(new_game_state, current_position, bomb_timers):
        events.append("MOVED_INTO_DANGER")

    reward = reward_from_events(self, events, old_game_state, new_game_state)

    # SARSA Update-Regel / Epsilon-greedy
    if new_game_state is not None:
        next_action = np.argmax(self.q_table[new_state])
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                            self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
    else:
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                            self.alpha * (reward - self.q_table[old_state][action_idx])


    # Track positions and actions
    self.last_positions.append(current_position)
    if len(self.last_positions) > 5:
        self.last_positions.pop(0)

    self.last_actions.append(self_action)
    if len(self.last_actions) > 5:
        self.last_actions.pop(0)

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game to update Q-values.
    """
    self.logger.debug(f'Events at end of round: {", ".join(map(str, events))}')

    last_state = tuple(state_to_features(last_game_state))
    
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)
    
    reward = reward_from_events(self, events, last_game_state, None)

    # Q-learning update for end of round
    self.q_table[last_state][action_idx] += self.alpha * (reward - self.q_table[last_state][action_idx])

    # Save the model after each round
    model_file = "my-sarsa-model.pt"
    with open(model_file, "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info(f"Q-table saved to {model_file}.")

def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    Assign rewards based on game events and progress towards targets.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5.0,
        e.KILLED_OPPONENT: 10.0,
        e.INVALID_ACTION: -1.0,
        e.GOT_KILLED: -5.0,
        e.KILLED_SELF: -50.0,
        "BOMB_PLACED_NEAR_BOXES": 2.0,
        "SAFE_BOMB_PLACEMENT": 5.0,
        "UNSAFE_BOMB_PLACEMENT": -5.0, 
        "STUCK_IN_LOOP": -3.0,
        "MOVED_INTO_DANGER": -10.0,
        "NO_PROGRESS": -1.0
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    # Add progress-based rewards (e.g., getting closer to coins or avoiding danger)
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Check for progress towards boxes
        old_box_distances = [np.linalg.norm(np.array(old_position) - np.array(box)) for box in np.argwhere(old_game_state['field'] == 1)]
        new_box_distances = [np.linalg.norm(np.array(new_position) - np.array(box)) for box in np.argwhere(new_game_state['field'] == 1)]

        if old_box_distances and new_box_distances:
            if min(new_box_distances) < min(old_box_distances):
                reward_sum += 0.5
            else:
                reward_sum -= 0.5

        # Reward moving away from bomb danger
        old_bomb_distances = [np.linalg.norm(np.array(old_position) - np.array(bomb[0])) for bomb in old_game_state['bombs']]
        new_bomb_distances = [np.linalg.norm(np.array(new_position) - np.array(bomb[0])) for bomb in new_game_state['bombs']]

        if old_bomb_distances and new_bomb_distances:
            if min(new_bomb_distances) > min(old_bomb_distances):
                reward_sum += 1.0
            else:
                reward_sum -= 1.0

    return reward_sum

def is_safe_to_place_bomb(game_state, position):
    """Check if it is safe to place a bomb at the given position."""
    bombs = game_state['bombs']
    x, y = position
    field = game_state['field']
    directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    for d in directions:
        if field[d] == 0 and is_safe_after_bomb(d, bombs):
            return True
    return False

def has_made_progress(old_game_state, new_game_state):
    """Check if the agent made progress towards coins, boxes, or avoiding bombs."""
    if old_game_state is None or new_game_state is None:
        return True  # No progress if game states are missing

    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    
    # Check if the agent is moving closer to a box or coin
    old_boxes = np.argwhere(old_game_state['field'] == 1)
    new_boxes = np.argwhere(new_game_state['field'] == 1)
    
    if len(old_boxes) > 0 and len(new_boxes) > 0:
        old_box_distances = [np.linalg.norm(np.array(old_position) - np.array(box)) for box in old_boxes]
        new_box_distances = [np.linalg.norm(np.array(new_position) - np.array(box)) for box in new_boxes]

        return min(new_box_distances) < min(old_box_distances)

    return True  # Progress is assumed if no boxes are present

def is_safe_after_bomb(position, bombs):
    """Check if the position remains safe after a bomb is placed."""
    pos_x, pos_y = position
    for bomb in bombs:
        bomb_x, bomb_y = bomb[0]
        if abs(bomb_x - pos_x) <= 1 or abs(bomb_y - pos_y) <= 1:
            return False
    return True

def get_next_target(game_state, logger=None):
    arena = game_state['field']
    free_space = (arena == 0)  # Angenommen, freie Flächen sind durch 0 dargestellt
    current_position = game_state['self'][3]  # Agentenposition
    targets = np.argwhere(arena == 1)  # Angenommen, Kisten sind durch 1 dargestellt

    # Verwenden von look_for_targets, um die nächste Bewegung zu bestimmen
    next_move = look_for_targets(free_space, current_position, targets, logger)
    return next_move

def is_closer_to_target(old_game_state, new_game_state, target):
    """Check if the agent is closer to the target than before."""
    old_position = old_game_state['self'][3]  # Alte Position des Agenten
    new_position = new_game_state['self'][3]   # Neue Position des Agenten

    # Berechnung der Manhattan-Distanz zu dem Ziel
    old_distance = np.sum(np.abs(np.subtract(old_position, target)))
    new_distance = np.sum(np.abs(np.subtract(new_position, target)))

    return new_distance < old_distance