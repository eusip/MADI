# -*- coding: utf-8 -*-

import numpy as np, itertools, sys
from basic import Game, _ACTION_VECTOR_, _ROOMS_LETTRE_
import os
# from random import randint
import matplotlib.pyplot as plt
import seaborn as sns

_LINUX_=True

# game = Game()
# print(str(game.dungeon.carte))
# print(str(game.dungeon.shape))

_CONSEQUENCE_ = -1
_DECISION_ = -2


class Qlearning():
    def __init__(self, alpha= .9, gamma=0.95, shape=[8, 8], dungeon =None):
        if dungeon is None:
            self.game = Game(shape)
        else:
            self.game = Game(dungeon=dungeon)
        self.action = self.game.action
        self.shape = self.game.dungeon.shape
        self.start_point = (self.shape[0] - 1, self.shape[1] - 1)
        self.start_state = (self.start_point[0], self.start_point[1], 1, 0, 0, 0)
        self.alpha = alpha
        self.gamma = gamma

        # Instantiate a new winnable game
        self.state_set = None
        self.is_winnable = False
        while not self.is_winnable:
            self.game.new_game(shape)
            self.portal_transfer_pos = self.create_portal_transfer_pos()
            self._transitions = None
            self._consequences = None
            self.init()

        self._qtable = {}  # initialize a nested dict containing states, actions, and their respective q values
        self._state_list = []
        self._possible_actions = {}  # a dict containing actions and their respective q values

        self.error=0
    
    def create_portal_transfer_pos(self):
        positions = []
        carte = self.game.dungeon.carte
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if carte[i, j] != self.game.dungeon.room.WALL:
                    positions.append((i, j))
        return positions

    def init(self):
        if self._transitions is None:
            self._transitions = dict()
        if self._consequences is None:
            self._consequences = dict()

        queue = [self.start_state]
        done_set = set()
        while len(queue) > 0:
            state = queue.pop()
            if state[:4] == self.start_point + (1, 1):
                self.is_winnable = True
            done_set.add(state)
            has, proba, n_state_proba = self.check_consequence(state)

            n_state = []
            if proba == 1.0:
                self._transitions.update({state: dict({_CONSEQUENCE_: proba})})
                self._consequences.update({state: n_state_proba})
                n_state.extend(n_state_proba.keys())
            elif proba == 0.0:
                self._transitions.update({state: dict({_DECISION_: (1.0)})})
                actions = self.get_possible_actions(state)
                n_state.extend(self.get_next_state(state, actions))
            else:
                self._transitions.update({state: dict({_CONSEQUENCE_: proba, _DECISION_: (1.0 - proba)})})
                self._consequences.update({state: n_state_proba})
                n_state.extend(n_state_proba.keys())
                actions = self.get_possible_actions(state)
                n_state.extend(self.get_next_state(state, actions))

            for s in n_state:
                if s not in done_set:
                    queue.append(s)
                    
        for s in done_set:
            isfinish = self.get_reward(s)[1]
            if isfinish:
                self._transitions[s] = None
        self.state_set = done_set

    def check_consequence(self, state):
        # while game is not finished
        # state is (pos_x, pos_y, life, treasure, key, sword) 0-5
        rm = self.game.dungeon.carte[state[:2]]
        room = self.game.dungeon.room

        # return has, proba, n_state_proba
        if rm == room.WALL:
            return True, 1.0, dict({self.start_point + state[2:]: 1.0})  # concatenation

        elif rm == room.ENEMY and state[5] == 0:
            return True, 0.3, dict({state[:2] + (0,) + state[3:]: 1.0})
        elif rm == room.TRAP:
            return True, 0.4, dict({state[:2] + (0,) + state[3:]: 0.25,
                                    self.start_point + state[2:]: 0.75})
        elif rm == room.CRACK:
            return True, 1.0, dict({state[:2] + (0,) + state[3:]: 1.0})
        elif rm == room.TREASURE and state[3] == 0:
            return True, 1.0, dict({state[:3] + (1,) + state[4:]: 1.0})
        elif rm == room.KEY and state[4] == 0:
            return True, 1.0, dict({state[:4] + (1,) + state[5:]: 1.0})
        elif rm == room.SWORD and state[5] == 0:
            return True, 1.0, dict({state[:5] + (1,): 1.0})
        elif rm == room.PORTAL:
            p = 1.0 / len(self.portal_transfer_pos)
            rest_state = state[2:]
            return True, 1.0, dict([(x + rest_state, p) for x in self.portal_transfer_pos])
        elif rm == room.PLATFORM:
            actions = self.get_possible_actions(state)
            n_state = self.get_next_state(state, actions)
            p = 1.0 / len(n_state)
            return True, 1.0, dict([(x, p) for x in n_state])
        return False, 0.0, None

    def get_next_state(self, state, actions: list):
        n_state = []
        pos = np.array(state[:2])
        for a in actions:
            next_pos = _ACTION_VECTOR_[a] + pos
            n_state.append(tuple(next_pos) + state[2:])
        return n_state

    def get_possible_actions(self, state):
        actions = []
        pos = state[:2]
        if pos[0] != 0:
            actions.append(self.action.NORTH)
        if pos[0] != self.shape[0] - 1:
            actions.append(self.action.SOUTH)
        if pos[1] != 0:
            actions.append(self.action.WEST)
        if pos[1] != self.shape[1] - 1:
            actions.append(self.action.EAST)
        return actions

    def get_reward(self, state):
        if state[:2] == self.start_point:
            if state[2] == 1 and state[3] == 1:
                return 100, True
        if state[2] == 0:
            return -10, True
        return 0, False

    def qtable_update(self, state, action, n_state):
        reward, finish = self.get_reward(state)
        #for action in self._qtable[state].keys():
        old = self._qtable[state][action]
#        print("This is the old action value for " + str(action) + ": " + str(self._qtable[state][action]))
        if finish:
            self._qtable[state][action] = reward
            return 
        else:
            if n_state not in self._qtable.keys():
                return 
            max_next_value = max(self._qtable[n_state].items(), key=lambda x: x[1])[1]
            self._qtable[state][action] = self._qtable[state][action] + self.alpha * (reward + self.gamma * max_next_value - self._qtable[state][action])
#        print("This is the new action value for " + str(action) + ": " + str(self._qtable[state][action]))
        new = self._qtable[state][action]
#        if self.error < max (self.error, abs(new - old)):
#            print(state)
        self.error = max (self.error, abs(new - old))

    def init_qtable(self):
        #  Create state list
        self._state_list = list(self.state_set)
        # print(len(self._state_list))
        """
        self._state_list = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for life in range(2):
                    for treasure in range(2):
                        for key in range(2):
                            for sword in range(2):
                                self._state_list.append((i, j, life, treasure, key, sword))
        """
        #  Create action list for states
        for state in self._state_list:
            self._possible_actions = {}
            actions = self.get_possible_actions(state)  # this is a list of enumerated actions
            for move in actions:  # for each intenum member assign a random initial q value
                self._possible_actions[move] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self._qtable[state] = self._possible_actions

    # def qlearn(self):
    #     iterations = 5
    #     while iterations > 0:
    #         iterations -= 1
    #         state = self.start_state
    #         n_move = max(self._qtable[state].items(), key=lambda x: x[1])[0]
    #         while self.game.game_won is not True:
    #             new_state = self.game.get_next_state(state=state, action=n_move)
    #             self.qtable_update(state)
    #             state = new_state
    #             n_move = max(self._qtable[state].items(), key=lambda x: x[1])[0]
    #             if self.game.isfinished is True:
    #                 self.game.restart_game()
    #                 break

    def qlearn(self):
        iterations = 1
        max_step = 1000
        while iterations > 0:
            iterations -= 1
#            state = self.start_state
#            n_move = max(self._qtable[state].items(), key=lambda x: x[1])[0]
            steps = 0
            print("Iteration #: " + str(iterations))
            self.error = 0
            self.alpha *= 0.9
            while steps < max_step:
                # self.game.dungeon.print_carte()
                state = self._state_list[np.random.randint(len(self._state_list))]
                while self._transitions[state] is None or _DECISION_  not in self._transitions[state].keys(): 
                    state = self._state_list[np.random.randint(len(self._state_list))]
                old_state = state
                actions = list(self._qtable[state].keys())
                n_move = actions[np.random.randint(len(actions))]

                # print("The previous state was: " + str(old_state))
                #self.game.print_game_info()
                # input()
                # if _LINUX_:
                #     os.system('clear')
                # else:
                #     os.system('cls')
                steps += 1
                #print("Step #: " + str(steps))
                new_state = self.game.get_next_state(state=state, action=n_move)
                # state = new_state
                # print("The new state is: " + str(state))
                self.qtable_update(old_state, n_move, new_state)
                # n_move = max(self._qtable[state].items(), key=lambda x: x[1])[0]
                
#                if self.game.isfinished is True:
#                    self.game.restart_game()
#                    break

            print("max_error", self.error)

    def nested_dict_to_matrix(self, dictionary, shape, pos=None, life=None, treasure=None, key=None, sword=None):
        self.mat = np.zeros(shape)
        for s in dictionary.keys():
            if pos is not None and s[0]!=pos[0] and s[1]!=pos[1]:
                continue
            if life is not None and s[2]!=life:
                continue
            if treasure is not None and s[3]!=treasure:
                continue
            if key is not None and s[4]!=key:
                continue
            if sword is not None and s[5]!=sword:
                continue
            i, j = s[:2]
            print(str(s))
            self.mat[i][j] = max(dictionary[s].items(), key=lambda x: x[1])[1]
        return self.mat, print(str(self.mat))

    # def nested_dict_to_matrix(self, dictionary, shape):
    #     mat = np.zeros(shape)
    #     for s in dictionary.keys():
    #         i, j = s[:2]
    #         mat[i][j] = max(dictionary[s].items(), key=lambda x: x[1])[1]
    #     return mat

    def print_heat_map(self):
        matrix = self.mat
        fig, ax = plt.subplots()
#        im = ax.imshow(matrix)
        sns.set(font_scale=2.8)
#        carte = self.game.dungeon.carte
        texts = np.zeros(self.shape, dtype=str)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                room = self.game.dungeon.carte[i, j]
                texts[i][j] = _ROOMS_LETTRE_[room]

        ax = sns.heatmap(matrix, annot=texts, cmap="RdYlGn", fmt='', vmin=-10, vmax=100)
        ax.set_title('lalala')
        plt.show()

    def init_experience(self, size):
        self.__init__(shape=[size, size])

    def run(self):
        self.init_qtable()
        self.qlearn()

file_name = "instance.txt"
mat = read_instance_from(file_name)
print(mat)

d = Dungeon()
d.init_from(mat)
d.print_carte()

ql = Qlearning(dungeon=d)
ql.game.dungeon.print_carte()
ql.init_qtable()
ql.qlearn()
ql.nested_dict_to_matrix(ql._qtable, [8, 8],pos=None,life=1,
                          treasure=1,key=None,sword=None)
print(ql.print_heat_map())
"""
 def heat_maps(name):
     for life in range:
         for treasure in range:
             for key in range:
                 for sword in range:
                     ql.nested_dict_to_matrix(life. treasure, key, sword)
                     ql.print_heat_map()

"""