# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:47:57 2018

@author: you
"""

import numpy as np
from basic import Game, _ACTION_VECTOR_, _ROOMS_LETTRE_, read_instance_from, Dungeon
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

_CONSEQUENCE_ = -1
_DECISION_ = -2

class MDPGame():
    def __init__(self, dungeon=None, gamma=0.95, shape=[8,8]): 
        self.gamma = gamma
        if dungeon is None:
            self.game = Game(shape)
        else:
            self.game = Game(dungeon=dungeon)
            
        self.actions = self.game.action
        self.shape = self.game.dungeon.shape
        self.start_point = (self.shape[0]-1, self.shape[1]-1)
        
        #State s: [player.get_state, map.get_state]
        self.start_state = (self.start_point[0],self.start_point[1], 1, 0,0,0)
    
        self.portal_transfer_pos = self.create_portal_transfer_pos()
        
        self._vtable = None #state & values
        self._dtable = None #state & decisions(actions)
        self._transitions = None
        self._consequences = None
        self.state_set = None
        self.is_winnable = False;
        self.init()
        while not self.is_winnable:
            self.game.new_game(shape)
            self.portal_transfer_pos = self.create_portal_transfer_pos()
            self._vtable = None #state
            self._transitions = None
            self._consequences = None
            self.init()
    
    def create_portal_transfer_pos(self):
        positions = []
        carte = self.game.dungeon.carte
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if carte[i, j] != self.game.dungeon.room.WALL:
                    positions.append((i,j))
        return positions
    
    def init(self):
        if self._transitions is None:
            self._transitions = dict()
        if self._consequences is None:
            self._consequences = dict()
        if self._dtable is None:
            self._dtable = dict()
            
        queue = [self.start_state]
        done_set = set()
        while len(queue)>0:
            state = queue.pop()
            if state[:4] == self.start_point + (1,1):
                self.is_winnable = True
            done_set.add(state)
            has, proba, n_state_proba = self.check_consequence(state)
            
            n_state = []
            if proba == 1.0:
                self._transitions.update({state:dict({_CONSEQUENCE_:proba})})
                self._consequences.update({state : n_state_proba})
                n_state.extend( n_state_proba.keys())
            elif proba == 0.0:
                self._transitions.update({state:dict({_DECISION_:(1.0)})})
                actions = self.get_possible_actions(state)
                n_state.extend(self.get_next_state(state, actions))
            else:
                self._transitions.update({state:dict({_CONSEQUENCE_:proba, _DECISION_:(1.0-proba)})})
                self._consequences.update({state : n_state_proba})
                n_state.extend(n_state_proba.keys())
                actions = self.get_possible_actions(state)
                n_state.extend(self.get_next_state(state, actions))
                
            for s in n_state:
                if s not in done_set:
                    queue.append(s)
                    
        if self._vtable is None:
            self._vtable = dict()
            
        for s in done_set:
            r, isfinish = self.get_reward(s)
            if isfinish: 
                self._transitions[s] = None
            self._vtable.update({s:r})
        self.state_set = done_set
            
    def init_tables(self):
        for s in self.state_set:
            r, isfinish = self.get_reward(s)
            self._vtable.update({s:r})
        self._dtable = dict()
         
        
        
    """        
    def is_winnable(self):
        res = self.filter_v_table(pos=self.start_point, life=1, treasure=1, key=None, sword=None)
        if len(res) == 0:
            return False
        else:
            return True
    """   
    #pos, pos, life, treasure, key sword
    def check_consequence(self,state):
#        while game is not finished
#        state is (pos_x, pos_y, life, treasure, key, sword) 0-5
        rm = self.game.dungeon.carte[state[:2]]
        room = self.game.dungeon.room
        
        #return has, proba, n_state_proba
        
        if rm == room.WALL:
            return True, 1.0, dict({self.start_point + state[2:] : 1.0}) #concatenation
        
        elif rm == room.ENEMY and state[5] == 0:
            return True, 0.3, dict({state[:2]+(0,)+state[3:] : 1.0})
        elif rm == room.TRAP:
            return True, 0.4, dict({state[:2]+(0,)+state[3:] : 0.25, 
                               self.start_point + state[2:] : 0.75})
        elif rm == room.CRACK:
            return True, 1.0, dict({state[:2]+(0,)+state[3:] : 1.0})
        elif rm == room.TREASURE and state[3]==0:
            return True, 1.0, dict({state[:3]+(1,)+state[4:] : 1.0})
        elif rm == room.KEY and state[4]==0:
            return True, 1.0, dict({state[:4]+(1,)+state[5:] : 1.0})
        elif rm == room.SWORD and state[5]==0:
            return True, 1.0, dict({state[:5]+(1,) : 1.0})
        elif rm == room.PORTAL:
            p = 1.0/len(self.portal_transfer_pos)
            rest_state = state[2:]
            return True, 1.0, dict([(x+rest_state, p) for x in self.portal_transfer_pos])
        elif rm == room.PLATFORM:
            actions = self.get_possible_actions(state)
            n_state = self.get_next_state(state, actions)
            p = 1.0/len(n_state)
            return True, 1.0, dict([(x,p) for x in n_state])
        return False, 0.0, None
        
    
    def get_next_state(self, state, actions:list):
        n_state =[]
        pos = np.array(state[:2])
        for a in actions:
            next_pos = _ACTION_VECTOR_[a] + pos
            n_state.append(tuple(next_pos)+state[2:])
        return n_state
        
    def get_reward(self, state):
        if state[:2] == self.start_point:
            if state[2]==1 and state[3]==1:
#                self.finished=True
                return 100, True
        if state[2]==0:
#            self.finished=True
            return -10, True
        return 0, False
    
    def get_possible_actions(self, state):
        actions = []
        pos = state[:2]
        if pos[0] != 0:
            actions.append(self.actions.NORTH)
        if pos[0] != self.shape[0]-1:
            actions.append(self.actions.SOUTH)
        if pos[1] != 0:
            actions.append(self.actions.WEST)
        if pos[1] != self.shape[1]-1:
            actions.append(self.actions.EAST)
        return actions
    
    def politic_iteration(self, iterations=200, epsilon=0.01):
        self.init_tables()
        changed = True
        while iterations > 0 and changed:
            n_vtable=dict()
            changed = False
            counter = 0
            for s in self._vtable.keys(): # for each state
                transition = self._transitions[s] # get transition on state
                if transition is None:
                    n_vtable.update({s : self._vtable[s]})
                    continue
                v_cons = 0
                if _CONSEQUENCE_ in transition.keys(): # it has no actions but only consequences
                    consqcs = self._consequences[s]# get all consqcs
                    for n_s in consqcs.keys(): 
                        v_cons += consqcs[n_s] * self._vtable[n_s]
                    v_cons = v_cons * transition[_CONSEQUENCE_]
                v_decis = 0 # negative infinit
                if _DECISION_ in transition.keys():
                    v_decis = -np.Inf # negative infinit
                    d = None # negative infinit
                    actions = self.get_possible_actions(s)
                    n_states = self.get_next_state(s,actions) # next state
                    for n_s in n_states: 
                        if self._vtable[n_s] > v_decis:
                            v_decis = self._vtable[n_s]
                            d = actions[n_states.index(n_s)]
                    v_decis = v_decis * transition[_DECISION_]
                    if s not in self._dtable.keys() or d!=self._dtable[s]:
                        changed = True
                        counter+=1
                        self._dtable.update({s:d})
                n_vtable.update({s : self.gamma*(v_cons + v_decis)})
                
            #print("iterations",iterations, "changed", changed, "counter", counter) 
            self._vtable = n_vtable
            iterations -= 1
        
    def value_iteration(self, iterations=200, epsilon=0.1):
        self.init_tables()
        error = np.Inf
        while iterations > 0 and error > epsilon:
            #print("iterations",iterations, "error", error) 
            n_vtable=dict()
            max_err = 0
            for s in self._vtable.keys(): # for each state
                transition = self._transitions[s] # get transition on state
                if transition is None:
                    n_vtable.update({s : self._vtable[s]})
                    continue
                v_cons = 0
                if _CONSEQUENCE_ in transition.keys(): # it has no actions but only consequences
                    consqcs = self._consequences[s]# get all consqcs
                    for n_s in consqcs.keys(): 
                        v_cons += consqcs[n_s] * self._vtable[n_s]
                    v_cons = v_cons * transition[_CONSEQUENCE_]
                v_decis = 0 # negative infinit
                if _DECISION_ in transition.keys():
                    v_decis = -np.Inf # negative infinit
                    actions = self.get_possible_actions(s)
                    n_states = self.get_next_state(s,actions) # next state
                    for n_s in n_states: 
                        if self._vtable[n_s] > v_decis:
                            v_decis = self._vtable[n_s]
                    v_decis = v_decis * transition[_DECISION_]
                n_vtable.update({s : self.gamma*(v_cons + v_decis)})
                
                if max_err < abs(n_vtable[s] - self._vtable[s]):
                    max_err = abs(n_vtable[s] - self._vtable[s])
#                print("s",s,"n_vtable", n_vtable[s], "self._vtable[s]",  self._vtable[s])

            if max_err < error:
                error = max_err
#            print("iterations",iterations, "error", error, "max_err", max_err) 
#            print(iterations > 0 and error > epsilon)
            
            self._vtable = n_vtable
            iterations -= 1
            
        #update the decistions
        for s in self._vtable.keys():
            transition = self._transitions[s] # get transition on state
            if transition is None:
                self._dtable.update({s:None})
                continue
            if _DECISION_ in transition.keys():
                actions = self.get_possible_actions(s)
                n_states = self.get_next_state(s,actions) # next state
                values = [self._vtable[n_s] for n_s in n_states]
                self._dtable.update({s:actions[np.argmax(values)]})
                
    def print_opt_path(self):
        state = self.start_state
        reward, finish = self.get_reward(state)
        while not finish:
            print(state, self._vtable[state])
            reward, finish = self.get_reward(state)
            actions = self.get_possible_actions(state)
            n_states = self.get_next_state(state, actions)
            v_n_states = [self._vtable[s] for s in n_states]
            n_state = n_states[np.argmax(v_n_states)]
            state = n_state
        
    def filter_v_table(self, pos=None, life=None, treasure=None, key=None, sword=None):
        filtered_vtable = dict()
        for s in self._vtable.keys():
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
            filtered_vtable[s] = self._vtable[s] 
        return filtered_vtable
    
    def dict_to_matrix(self, dictionary, shape):
        mat = np.zeros(shape)
        for s in dictionary.keys():
            i,j = s[:2]
            mat[i][j] = dictionary[s]
        return mat
    
    def print_heat_map(self, dictionary, title=""):
        matrix = self.dict_to_matrix(dictionary, self.shape)
        fig, ax = plt.subplots()
#        im = ax.imshow(matrix)
        sns.set(font_scale=2.6)
#        carte = self.game.dungeon.carte
        texts = np.zeros(self.shape, dtype=str)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                room = self.game.dungeon.carte[i, j]
                texts[i][j] = _ROOMS_LETTRE_[room]
        
        ax = sns.heatmap(matrix, annot = texts, cmap="RdYlGn", fmt = '',vmin=-10, vmax=100)        
        ax.set_title(title)

        plt.show()

        """
        # Loop over data dimensions and create text annotations.
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                room = self.game.dungeon.carte[i, j]
                text = ax.text(j, i, _ROOMS_LETTRE_[room],
                               ha="center", va="center", color="w")

        ax.set_title("The values of state")
        fig.tight_layout()
        plt.show()
        """
    def init_experience(self, size):
        self.__init__(shape=[size, size])
        
    def run(self): 
        if self.solver == "value_iteration":
            self.value_iteration()
        elif self.solver == "politic_iteration":
            self.politic_iteration()
        else:
            return
    
    def set_solver(self, solver=None):
        if solver is None:
            self.solver = "politic_iteration"
        else:
            self.solver = solver
        #self.politic_iteration()
        
"""
game = MDPGame(shape=[5,5])
#game.game.dungeon.print_carte()
print(len(game._vtable.keys()))
#print(len(game._transitions.keys()))
game.value_iteration()
game.game.dungeon.print_carte()

filterd_dict = game.filter_v_table(None, 1,1,1,1)
game.print_heat_map(filterd_dict,"t:1, k:1, s:1")

filterd_dict = game.filter_v_table(None, 1,0,0,0)
game.print_heat_map(filterd_dict,"t:0, k:0, s:0")

#pos, life, treasure, key sword

filterd_dict = game.filter_v_table(None, 1,0,0,0)
game.print_heat_map(filterd_dict,"t:0, k:0, s:0")

filterd_dict = game.filter_v_table(None, 1,0,1,0)
game.print_heat_map(filterd_dict,"t:0, k:1, s:0")

filterd_dict = game.filter_v_table(None, 1,0,0,1)
game.print_heat_map(filterd_dict,"t:0, k:0, s:1")

filterd_dict = game.filter_v_table(None, 1,0,1,1)
game.print_heat_map(filterd_dict,"t:0, k:1, s:1")

filterd_dict = game.filter_v_table(None, 1,1,1,1)
game.print_heat_map(filterd_dict,"t:1, k:1, s:1")

game.politic_iteration(iterations=50, epsilon=0.01)
filterd_dict = game.filter_v_table(None, 1,0,None,0)
game.print_heat_map(filterd_dict)
"""
#for t in game.filter_v_table(None, 1,1).items():
#    print(t)
    
#game.print_opt_path()
#for t in game._vtable.items():
#    print(t)

"""
#game.simulation(10,0.1)
print('transitions')
for t in game._transitions.items():
    print(t)
print('consequences')
for t in game._consequences.items():
    print(t)
#print([i  for i in game._vtable.items() if i[1] >0])
"""

"""
simulations and visulization of optimum policies
room that loose objects. 
"""

"""
file_name = "instance.txt"
mat = read_instance_from(file_name)
print(mat)

d = Dungeon()
d.init_from(mat)
d.print_carte()
game = MDPGame(d)
game.value_iteration()
game.politic_iteration()
game._vtable[(0, 0, 1, 0, 1, 0)]
game._vtable[(0, 0, 1, 0, 0, 0)]
filterd_dict = game.filter_v_table(None, 1,0,0,0)
game.print_heat_map(filterd_dict,"mdp_t:0, k:0, s:0")

filterd_dict = game.filter_v_table(None, 1,0,1,0)
game.print_heat_map(filterd_dict,"mdp_t:0, k:1, s:0")

filterd_dict = game.filter_v_table(None, 1,0,0,1)
game.print_heat_map(filterd_dict,"mdp_t:0, k:0, s:1")

filterd_dict = game.filter_v_table(None, 1,1,1,0)
game.print_heat_map(filterd_dict,"mdp_t:1, k:1, s:0")

filterd_dict = game.filter_v_table(None, 1,1,1,1)
game.print_heat_map(filterd_dict,"mdp_t:1, k:1, s:1")
"""


