import os
from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features, Actions, name_to_action_enum

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
STEP_TOWARD_BFS_COIN = "STEP_TOWARD_BFS_COIN"
COIN_DIST_INCREASED = "COIN_DIST_INCREASED"
TAKEN_DANGEROUS_ACTION = "TAKEN_DANGEROUS_ACTION"
USELESS_BOMB='USELESS_BOMB'
GOOD_BOMB='GOOD_BOMB'
VERY_GOOD_BOMB='VERY_GOOD_BOMB'

# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# hyperparameters
GAMMA = 0.6
ALPHA = 0.3
EPSILON = 0.35
EPSILON_DECAY = 0.996
ALPHA_DECAY = 1.0 # alpha is constant
EPSILON_MIN = 0.0 # not in use

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Q-table
    self.q_table = {}
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Training Q-table from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Continuing training of Q-table from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)

    self.stats=[]
    
    # parameters, info
    self.round = 0
    self.alpha = ALPHA  # Lernrate davor 0.1
    self.gamma = GAMMA  # Diskontfaktor um so höher um so stärker werden zufällige gewinne belohnt
    self.epsilon = EPSILON  # exploration probability (epsilon-greedy method)
    self.epsilon_decay = EPSILON_DECAY  # Epsilon-Decay für Exploration-Exploitation Tradeoff
    self.alpha_decay = ALPHA_DECAY
    self.epsilon_min = EPSILON_MIN


def rotate(state, action, k):
    """
    Rotate 'state' and 'action' for 'k' times counterclockwise by 90°.
    Returns (None, None) if one of 'state' and 'action' is None.
    """

    if state is None or action is None:
        return None, None

    # rotate field environment
    env5x5_field = np.array(state[0])
    env5x5_field = env5x5_field.reshape((5,5))
    env5x5_field = np.rot90(env5x5_field, k=k)
    env5x5_field_rot = env5x5_field.flatten()

    # rotate relative coin positions
    env5x5_coins = state[1]
    k = k%4
    rad = k*np.pi/2
    rot_matrix = np.array(( [np.cos(rad), -np.sin(rad)],
                            [np.sin(rad),  np.cos(rad)]  ))
    env5x5_coins_rot = []
    for coin in env5x5_coins:
        coin = rot_matrix@np.array(coin) #rotate
        env5x5_coins_rot.append( tuple( (int(coin[0]), int(coin[1])) ) )
    env5x5_coins_rot.sort()

    # rotate bfs coin direction (in enum representation)
    # UP --> RIGHT --> DOWN --> LEFT (--> UP)
    # note that we need to rotate the associated vector (see beginning of file)
    coin_step, dist = state[2]
    coin_step_rot = (coin_step+k)%4

    # assemble rotated state
    rot_state = [tuple(env5x5_field_rot), tuple(env5x5_coins_rot), (coin_step_rot, dist)]
    rot_state = tuple(rot_state)

    # rotate action (in enum representation)
    if action == 'UP':
        action_id = Actions.UP.value
        rot_action_id = (action_id+k)%4
        rot_action = Actions(rot_action_id).name

    elif action == 'RIGHT':
        action_id = Actions.RIGHT.value
        rot_action_id = (action_id+k)%4
        rot_action = Actions(rot_action_id).name

    elif action == 'DOWN':
        action_id = Actions.DOWN.value
        rot_action_id = (action_id+k)%4
        rot_action = Actions(rot_action_id).name

    elif action == 'LEFT':
        action_id = Actions.LEFT.value
        rot_action_id = (action_id+k)%4
        rot_action = Actions(rot_action_id).name
        
    elif action == 'WAIT':
        rot_action = 'WAIT'

    elif action == 'BOMB':
        rot_action = 'BOMB'


    return rot_state, rot_action


def mirror(state, action, axis):
    """
    Mirroring state on axis (axis=1,2,3,4) with the following correspondences: 
        axis = 1: The diagonal axis (~matrix transposition)
        axis = 2: Vertical axis
        axis = 3: The second diagonal (bottom left corner to top right corner)
        axis = 4: Horizontal axis

    Returns (None, None) if one of 'state' and 'action' is None.

    """

    if state is None or action is None:
        return None, None
    
    env5x5_field = np.array(state[0]).reshape((5,5))
    coin_step, dist = state[2]
    coin_step_enum = Actions(coin_step)
    coin_step_name = coin_step_enum.name
    if action == 'UP':
        action_id = Actions.UP.value
    elif action == 'RIGHT':
        action_id = Actions.RIGHT.value
    elif action == 'DOWN':
        action_id = Actions.DOWN.value
    elif action == 'LEFT':
        action_id = Actions.LEFT.value
    elif action == 'WAIT':
        action_id = Actions.WAIT.value
    elif action == 'BOMB':
        action_id = Actions.BOMB.value
    
    if axis==1:
        # diagonal (transpose field)
        env5x5_field = env5x5_field.T
        env5x5_coins = [ (-coin[1], -coin[0]) for coin in state[1]]
        if coin_step_name == 'UP' or coin_step_name == 'DOWN':
            coin_step_mirror = (coin_step+1)%4
        elif coin_step_name == 'RIGHT' or coin_step_name == 'LEFT':
            coin_step_mirror = (coin_step-1)%4
        else:
            # coin_step is 'WAIT'
            coin_step_mirror = coin_step
        
        if action == 'DOWN' or action == 'UP':
            action_id_mirror = (action_id+1)%4
        elif action == 'RIGHT' or action == 'LEFT':
            action_id_mirror = (action_id-1)%4
        else:
            # action is 'WAIT' or 'BOMB'
            action_id_mirror = action_id

    elif axis==2:
        # mirror vertically
        env5x5_field = np.fliplr(env5x5_field)
        env5x5_coins = [ (-coin[0], coin[1]) for coin in state[1]]
        if coin_step_name == 'LEFT' or coin_step_name == 'RIGHT':
            coin_step_mirror = (coin_step+2)%4
        else:
            coin_step_mirror = coin_step
        if action == 'LEFT' or action == 'RIGHT':
            action_id_mirror = (action_id+2)%4
        else:
            action_id_mirror = action_id
    
    elif axis==3:
        # second diagonal
        # mirroring on the second diagonal can be done as follows: rotate once, transpose, rotate back

        # rotate 90°
        env5x5_field = np.rot90(env5x5_field, k=1)
        # transpose
        env5x5_field = env5x5_field.T
        # rotate back
        env5x5_field = np.rot90(env5x5_field,k=3)

        # directional vectors flip entries
        env5x5_coins = [ (coin[1], coin[0]) for coin in state[1]]

        if coin_step_name == 'RIGHT' or coin_step_name == 'LEFT':
            coin_step_mirror = (coin_step+1)%4
        elif coin_step_name == 'UP' or coin_step_name == 'DOWN':
            coin_step_mirror = (coin_step-1)%4
        else:
            # coin_step is 'WAIT'
            coin_step_mirror = coin_step
        
        if action == 'RIGHT' or action == 'LEFT':
            action_id_mirror = (action_id+1)%4
        elif action == 'DOWN' or action == 'UP':
            action_id_mirror = (action_id-1)%4
        else:
            # action is 'WAIT' or 'BOMB'
            action_id_mirror = action_id

    elif axis==4:
        # mirror horizontally        
        env5x5_field = np.flipud(env5x5_field)
        env5x5_coins = [ (coin[0], -coin[1]) for coin in state[1] ]
        if coin_step_name == 'UP' or coin_step_name == 'DOWN':
            coin_step_mirror = (coin_step+2)%4
        else:
            coin_step_mirror = coin_step
        if action == 'UP' or action == 'DOWN':
            action_id_mirror = (action_id+2)%4
        else:
            action_id_mirror = action_id

    else:
        raise ValueError("'axis' must be one of [1,2,3,4].")

    # assemble return
    env5x5_field = tuple(env5x5_field.flatten())
    env5x5_coins.sort()
    env5x5_coins = tuple(env5x5_coins)
    mir_state  = tuple([env5x5_field, env5x5_coins, (coin_step_mirror, dist)])
    mir_action = Actions(action_id_mirror).name

    return mir_state, mir_action


def update_q_table(self, old_state, self_action, new_state, reward):
    action_idx = ACTIONS.index(self_action)

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))

    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    # Update: SARSA
    if new_state is not None:
        next_action = np.argmax(self.q_table[new_state])
        self.q_table[old_state][action_idx] = \
            self.q_table[old_state][action_idx] + \
            self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
    else:
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                              self.alpha * (reward - self.q_table[old_state][action_idx])


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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

    new_state = state_to_features(new_game_state)
    old_state = state_to_features(old_game_state)

    old_state = tuple(old_state)
    new_state = tuple(new_state)
    
    action_idx = ACTIONS.index(self_action)

    # Logger: write game events to logger
    self.logger.debug(f'Event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # NOTE generalization from smaller fields can be implemented here

    custom_events = detect_custom_events(old_state, self_action, new_state)
    
    events = events + custom_events

    # Reward: hand out rewards
    reward = reward_from_events(self, events, old_game_state, new_game_state)
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    ## Q-table updates
    # original state
    update_q_table(self, old_state, self_action, new_state, reward)
    
    # rotations
    for k in range(1, 4):
        old_state_rot, self_action_rot = rotate(old_state, self_action, k)
        new_state_rot, _ = rotate(new_state, self_action, k)
        update_q_table(self, old_state_rot, self_action_rot, new_state_rot, reward)

    # mirrors
    for axis in range(1, 5):
        old_state_mir, self_action_mir = mirror(old_state, self_action, axis)
        new_state_mir, _ = mirror(new_state, self_action, axis)
        update_q_table(self, old_state_mir, self_action_mir, new_state_mir, reward)

    # NOTE Lisa adds a decay of epsilon in her code here --> good idea generally. 
    # is there an alternative place?
    # How fast should epsilon decay?


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.round += 1
    self.epsilon *= self.epsilon_decay
    self.alpha   *= self.alpha_decay
    # self.epsilon = self.epsilon*self.round/(self.round+1)

    last_state = state_to_features(last_game_state)
    last_state = tuple(last_state)


    custom_events = detect_custom_events(last_state, last_action, None)
    
    events = events + custom_events

    # NOTE what if state is not existent in q-table yet?
    # do we fill the initial q-table with random values or zeros?
    # (we probably don't actually want to fill the q_table variable for memory reasons)
    
    # Reward mit den letzten Zuständen
    reward = reward_from_events(self, events, last_game_state, None)

    # End of round Q-Update
    update_q_table(self, None, last_action, last_state, reward)
    
    #self.stats=[]
    self.stats.append(last_game_state['step'])

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)
    
    if not os.path.isfile("stats.txt"):
        self.logger.info("new stats.")
        with open('stats.txt', 'w') as file:
            file.write('%s' % self.stats)
    else:
        self.logger.info("Continuing stats.")
        with open('stats.txt', 'a') as file:
            file.write('%s' % self.stats)


def detect_custom_events(old_state, self_action, new_state):
    events = []
    # step toward coin found by BFS
    coin_step_id = old_state[2][0]
    coin_step_name = Actions(coin_step_id).name
    if coin_step_name == self_action and e.COIN_COLLECTED not in events:
        events.append(STEP_TOWARD_BFS_COIN)

    # distance to nearest coin increased
    if new_state is not None:
        coin_dist_old = old_state[2][1]
        coin_dist_new = new_state[2][1]
        if coin_dist_new > coin_dist_old and e.COIN_COLLECTED not in events:
            events.append(COIN_DIST_INCREASED)

    # taken dangerous action
    dangerous_actions = old_state[3]
    action_enum = name_to_action_enum(self_action)
    if action_enum in dangerous_actions:
        events.append(TAKEN_DANGEROUS_ACTION)

    #bomb placed next to crate
    if self_action=='BOMB':
        crate_count=0
        if old_state[0][14]==1:
            crate_count+=1
            if old_state[0][15]==1:
                crate_count+=1
        if old_state[0][18]==1:
            crate_count+=1
            if old_state[0][23]==1:
                crate_count+=1
        if old_state[0][12]==1:
            crate_count+=1
            if old_state[0][11]==1:
                crate_count+=1
        if old_state[0][8]==1:
            crate_count+=1
            if old_state[0][3]==1:
                crate_count+=1
        if crate_count==0:
            events.append(USELESS_BOMB)
        if crate_count==1:
            events.append(GOOD_BOMB)
        if crate_count>=2:
            events.append(VERY_GOOD_BOMB)
                
    return events


def reward_from_events(self, events: List[str], old_game_state, new_game_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    state_old = state_to_features(old_game_state)
    coin_dist = state_old[2][1]

    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.WAITED: -0.001,
        STEP_TOWARD_BFS_COIN: 1/(coin_dist),
        COIN_DIST_INCREASED: -1/(coin_dist),
        TAKEN_DANGEROUS_ACTION: -10,
        USELESS_BOMB: -2,
        GOOD_BOMB: 2,
        VERY_GOOD_BOMB: 4

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
