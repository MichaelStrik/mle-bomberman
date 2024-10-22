import os
import pickle
import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up agent from saved state.")
    
    # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden ist
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        # Falls keine gespeicherte Q-Tabelle vorhanden ist, initialisieren wir sie leer.
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.1
    self.gamma = 0.9
    self.epsilon = 0.1
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    state = tuple(features)

    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    
    # Alle Aktionen überprüfen
    q_values = self.q_table[state]
#hier geändert
    valid_actions = get_valid_actions(game_state)
    
    #for action in ACTIONS:
       # if action == 'BOMB':
           # if is_safe_position(game_state, game_state['self'][3], get_bomb_timers(game_state)):
               # valid_actions.append(action)
        #else:
            #valid_actions.append(action)
    
    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
    else:
        self.logger.debug("Querying Q-table for action.")
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
    
    return action

def bomb_radius(game_state):
    bombradius=[]
    bombs=game_state['bombs']
    field=game_state['field']
    for bomb in bombs:              #kann ja mehrere geben
        if bomb[1]==0:              #nur wenn im nächsten schritt da explosion
            bombradius.append(bomb[0])  #bomb[0] Koordinaten, bomb[1] timer
            if field[bomb[0][0]+1][bomb[0][1]] != -1 and bomb[0][0]+1 < 17:
                bombradius.append((bomb[0][0]+1,bomb[0][1]))
                if field[bomb[0][0]+2][bomb[0][1]] != -1 and bomb[0][0]+2 < 17:
                    bombradius.append((bomb[0][0]+2,bomb[0][1]))
                    if field[bomb[0][0]+3][bomb[0][1]] != -1 and bomb[0][0]+3 < 17:
                        bombradius.append((bomb[0][0]+3,bomb[0][1]))
            if field[bomb[0][0]-1][bomb[0][1]] != -1 and bomb[0][0]-1 < 17:
                bombradius.append((bomb[0][0]-1,bomb[0][1]))
                if field[bomb[0][0]-2][bomb[0][1]] != -1 and bomb[0][0]-2 < 17:
                    bombradius.append((bomb[0][0]-2,bomb[0][1]))
                    if field[bomb[0][0]-3][bomb[0][1]] != -1 and bomb[0][0]-3 < 17:
                        bombradius.append((bomb[0][0]-3,bomb[0][1]))
            if field[bomb[0][0]][bomb[0][1]+1] != -1 and bomb[0][1]+1 < 17:
                bombradius.append((bomb[0][0]-1,bomb[0][1]))
                if field[bomb[0][0]][bomb[0][1]+2] != -1 and bomb[0][1]+2 < 17:
                    bombradius.append((bomb[0][0],bomb[0][1]+2))
                    if field[bomb[0][0]][bomb[0][1]+3] != -1 and bomb[0][1]+3 < 17:
                        bombradius.append((bomb[0][0],bomb[0][1]+3))
            if field[bomb[0][0]][bomb[0][1]-1] != -1 and bomb[0][1]-1 > 0:
                bombradius.append((bomb[0][0]-1,bomb[0][1]))
                if field[bomb[0][0]][bomb[0][1]-2] != -1 and bomb[0][1]-2 > 0:
                    bombradius.append((bomb[0][0],bomb[0][1]-2))
                    if field[bomb[0][0]][bomb[0][1]-3] != -1 and bomb[0][1]-3 > 0:
                        bombradius.append((bomb[0][0],bomb[0][1]-3))
    return bombradius
                

def get_valid_actions(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    crates = find_crates(game_state)
    bombradius=bomb_radius(game_state)
    explosionmap=game_state['explosion_map']

    valid_actions = []

    directions = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }

    # Verfügbarkeit der Richtungen überprüfen
    for action, (nx, ny) in directions.items():
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]: #in Feld
#hier geändert
            #if field[nx, ny] == 0 and not any(bomb[0] == (nx, ny) for bomb in bombs):  # Kein Hindernis und keine Bombe, aber kann Explosion sein
            if field[nx,ny] ==0 and (nx,ny) not in bombenradius and explosionmap[nx,ny]==0:
                valid_actions.append(action)

    if len(valid_actions) == 0:
        valid_actions.append('WAIT')
    
    if 'BOMB' not in valid_actions and any(field[nx, ny] == 1 for nx, ny in directions.values()) and is_safe_to_place_bomb(game_state):
        valid_actions.append('BOMB')

    # Hier entfernen wir die Bombe als Option, wenn es keine sicheren Kisten gibt
    if 'BOMB' in valid_actions:
        safe_crates = []
        for crate in crates:
            if is_safe_position(crate, game_state):  # is_safe_position anstelle von is_safe_to_place_bomb
                safe_crates.append(crate)
        if len(safe_crates) == 0:
            valid_actions.remove('BOMB')

    return valid_actions


def is_safe_to_place_bomb(game_state, position=None):
    """Überprüft, ob es sicher ist, eine Bombe zu platzieren."""
    if position is None:
        position = game_state['self'][3]  # Aktuelle Position des Agenten
    x, y = position
    field = game_state['field']

    # Prüfen, ob der Agent nach dem Platzieren der Bombe einen sicheren Fluchtweg hat
    for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and is_safe_position((nx, ny), game_state):       #==0 heißt free tile, keine kiste oder bombe
                return True
    return False

def is_safe(games_state, position, bomb_timers):
    invalid_actions=[]
    #für wenn mein Agent Bombe legt, dass er ihr aus entkommt
    if timer==0 and bx==x and by==y: 
        invalid_actions.append('WAIT','BOMB')
    if timer==0:
        if x-bx==1 and abs(y-by)<=3:  #eins rechts von der Bombe, kann aber auch drüber oder drunter sein
            invalid_actions.append('LEFT')
        if x-bx==-1 and abs(y-by)<=3: #eins links von der Bombe
            invalid_actions.append('RIGHT')
        if y-by==1 and abs(b-bx)<=3:  #eins unter der Bombe
    if timer==2:
        if x-bx==2 and abs(
        
       
        

def is_safe_position(game_state, position, bomb_timers):
    """
    Check if a position is safe considering the bombs' timers and their blast radius.
    
    :param game_state: The current game state dictionary.
    :param position: The (x, y) tuple of the position to check.
    :param bomb_timers: A list of (position, timer) tuples for each bomb.
    :return: True if the position is safe, False otherwise.
    """
    field = game_state['field']
    x, y = position

    for bomb_pos, timer in bomb_timers:
        bx, by = bomb_pos

        if timer <= 0:
            continue

        # Check horizontal and vertical lines for blast danger
        if bx == x and abs(by - y) <= 2 and timer <= 4:  # Same column and within blast range
            return False
        if by == y and abs(bx - x) <= 2 and timer <= 4:  # Same row and within blast range
            return False

    return True


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    # Feature 1: Position of the agent
    x, y = game_state['self'][3]

    # Feature 2: Distance to the nearest coin
    coins = game_state['coins']
    field = game_state['field']
    if coins:
        distances = [bfs(field, (x, y), coin) for coin in coins]
        min_distance = min(distances)
        closest_coin = coins[distances.index(min_distance)]
        coin_dx = closest_coin[0] - x
        coin_dy = closest_coin[1] - y
        coin_feature = np.array([coin_dx, coin_dy, min_distance])
    else:
        coin_feature = np.array([0, 0, float('inf')])  # Kein Münze in Sicht

    # Feature 3: Distance to the nearest crate (box)
    crates = np.argwhere(field == 1)
    if crates.any():
        crate_distances = [bfs(field, (x, y), tuple(crate)) for crate in crates]
        min_crate_distance = min(crate_distances)
        closest_crate = crates[crate_distances.index(min_crate_distance)]
        crate_dx = closest_crate[0] - x
        crate_dy = closest_crate[1] - y
        crate_feature = np.array([crate_dx, crate_dy, min_crate_distance])
    else:
        crate_feature = np.array([0, 0, float('inf')])  # Keine Kisten in Sicht

    # Feature 4: Distance to the nearest bomb
    bomb_distances = [np.linalg.norm(np.array([x, y]) - np.array(bomb[0])) for bomb in game_state['bombs']]
    bomb_feature = np.array([min(bomb_distances)]) if bomb_distances else np.array([float('inf')])

    # Combine all features into a single array
    return np.concatenate([coin_feature, crate_feature, bomb_feature])

def bfs(field, start, target):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == target:
            return dist

        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))
    
    return float('inf')

def find_crates(game_state):
    """
    Findet alle Kisten auf dem Spielfeld und gibt deren Positionen zurück.

    :param game_state: Das aktuelle Spielzustand-Dictionary.
    :return: Eine Liste von Tupeln mit den Positionen der Kisten.
    """
    crates = []
    field = game_state['field']  # Das Spielfeld
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 1:  # 1 steht normalerweise für eine Kiste
                crates.append((x, y))
    return crates

def get_bomb_timers(game_state):
    """
    Get the positions and timers of all bombs on the field.
    
    :param game_state: The current game state dictionary.
    :return: A list of (position, timer) tuples for each bomb.
    """
    bombs = game_state['bombs']  # List of tuples with ((x, y), timer)
    return [(bomb[0], bomb[1]) for bomb in bombs]
