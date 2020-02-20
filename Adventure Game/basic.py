# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:18:28 2018

@author: you
"""

from enum import IntEnum
import numpy as np 

_DEBUG_ = False
_GAME_INFO_ = False


_ROOMS_LETTRE_ = dict({
                #   the type of the caracters in the dungeon.
        0:"o",  # starting position
        1:" ",  # empty cell nothing
        2:"■",  # wall (moving to a wall will bounce back to the starting position)
        
        3:"E",  # enemy a malicious foe is attacking the adventurer
                # (the adventurer is victorious with penemy probability, 
                # fixed to 0.7; otherwise the adventurer is dead).
                
        4:"R",  # trap a trap that either kill the adventurer (with probability 0.1), 
                # bring the adventurer to the starting cell (probability 0.3)
                # or nothing happens (probability 0.6)
                
        5:"C",  # cracks immediate death
        6:"T",  # treasure to open the door of the treasure’s room 
                # it is necessary to have the golden key
        
        7:"S",  # magic sword the adventurer can collect the sword and use it in combat;
                # its powers allow him to win any combat without fighting
                
        8:"K",   # golden key necessary to open the treasure’s room
        9:"P",  # magic portal a magic portal teleports the agent 
                # to a random (non-wall) cell of the dungeon
                
        10:"-",  # moving platform the pavement is moving! 
                 # the adventurer is forced to take refuge in one of the neighbouring cells (at random)
        
        -1:"@"   #player is here
        })

class Room(IntEnum):
    def __str__(self):
        return str(_ROOMS_LETTRE_[self.value])
    START=0     # starting position
    EMPTY=1     # empty cell nothing
    WALL=2      # wall (moving to a wall will bounce back to the starting position)
    ENEMY=3     # enemy a malicious foe is attacking the adventurer
                # (the adventurer is victorious with penemy probability, 
                # fixed to 0.7; otherwise the adventurer is dead).
                
    TRAP=4      # trap a trap that either kill the adventurer (with probability 0.1), 
                # bring the adventurer to the starting cell (probability 0.3)
                # or nothing happens (probability 0.6)
                
    CRACK=5     # cracks immediate death
    TREASURE=6  # treasure to open the door of the treasure’s room 
                # it is necessary to have the golden key
        
    SWORD=7     # magic sword the adventurer can collect the sword and use it in combat;
                # its powers allow him to win any combat without fighting
                
    KEY=8       # golden key necessary to open the treasure’s room
    PORTAL=9    # magic portal a magic portal teleports the agent 
                # to a random (non-wall) cell of the dungeon
                
    PLATFORM=10 # moving platform the pavement is moving! 
                # the adventurer is forced to take refuge in one of the neighbouring cells (at random)

class Action(IntEnum):
        NORTH=0
        SOUTH=1
        EAST=2
        WEST=3
        
_ACTION_VECTOR_ = dict({
        Action.NORTH:np.array([-1,0]),
        Action.SOUTH:np.array([1,0]),
        Action.EAST:np.array([0,1]),
        Action.WEST:np.array([0,-1])})
    
"""   
str(Room(10))
Room.PLATFORM==10
print(Action.NORTH.name)
Action.NORTH==0
len(Room)
"""

class Dungeon():
    def __init__(self,shape=[8,8], room=Room):
        self.carte = np.zeros(shape, dtype=np.int)
        self.room = room
        self.shape = shape
        self.init_distribution()
        #self._state = self.init_state()
        
    def init_from(self, mat):
        self.shape = mat.shape
        self.carte = mat
        
    def init_random(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                item = np.random.randint(0,len(self.room))
                while item==self.room.START or item==self.room.TREASURE :
                    item = np.random.randint(0,len(self.room))
                self.carte[i,j] = item;
        self.carte[0,0] = self.room.TREASURE
        self.carte[self.shape[0]-1,self.shape[1]-1] = self.room.START
    
    def init_distribution(self, distribution=[0, 0.2, 0.1, 0.1, 0.1,
                                              0.1, 0, 0.1, 0.1, 0.1, 0.1]):
        for i in range(self.shape[0]): 
            line = np.random.choice(len(self.room), self.shape[1], p=distribution)
            while self.room.START in line or self.room.TREASURE in line:
                line = np.random.choice(len(self.room), self.shape[1], p=distribution)
            self.carte[i] = line;
        self.carte[0,0] = self.room.TREASURE
        self.carte[self.shape[0]-1,self.shape[1]-1] = self.room.START
        
        self._state = self.init_state()
        
    def print_carte(self, carte=None):
        if carte is None:
            carte=self.carte
        print(np.array2string(carte, formatter={'int':lambda x: _ROOMS_LETTRE_[x]}))
    
    def clear_room(self, i,j):
        self.carte[i,j] = self.room.EMPTY
        self._state[i,j] = 0
        
    def has_key_sword(self):# check if the room is available
        if self.room.KEY not in self.carte:
            return False
        if self.room.SWORD not in self.carte:
            return False
        return True
    
    def init_state(self):
        state = dict()
        rm = self.room
        rms = [rm.TREASURE, rm.KEY, rm.SWORD, rm.ENEMY]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.carte[i, j] in rms:
                     state.update({(i, j):1})
        return state        
    @property 
    def state(self):
        #values not in order !! return tuple(self._state.values())
        pass
    def get_state_dict(self):
        return self._state
    
    
"""
dg = Dungeon()
dg.init_random()
dg.print_carte()
dg.clear_room(0,1)
dg.print_carte()

dg = Dungeon()
distribution=[0, 0.2, 0.1, 0.1, 0.1,
              0, 0.1, 0.1, 0.1, 0.1, 0.1]
dg.init_distribution(distribution) 
dg.print_carte()
dg.has_key_sword()
dg.carte[(0,0)]
"""


class Player():
    def __init__(self, pos=np.array([0,0]), life=1, treasure=0, key=0, sword=0):
        self.pos=pos
        self._state = self.init_state( pos, life, treasure, key, sword)
        
    def get(self, item):
        self._state[item]=1;
        if _DEBUG_:
            print("-player get "+item+" at "+str(self.pos))
            print("player state: "+ str(self.state))
        
    def has(self,item):
        return self._state[item]>0
    
    def use(self, item):
        if self.has(item):
            self._state[item]=0;
        if _DEBUG_:
            print("-player use "+item+" at "+str(self.pos))
            print("player state: "+ str(self.state))
            
    def lose(self, item):
        if self.has(item):
            self._state[item]=0;
        if _DEBUG_:
            print("-player lose "+item+" at "+str(self.pos))
            print("player state: "+ str(self.state))
        
    def go_to(self, pos):
        self.pos=pos
        self._state["pos_x"]=pos[0];
        self._state["pos_y"]=pos[1];
        if _DEBUG_:
            print("-player go to "+str(pos))
            print("player state: "+ str(self.state))
        
    def init_state(self, pos, life, treasure, key, sword):
        return dict({
                "pos_x":pos[0],
                "pos_y":pos[1],
                "life":life,
                "treasure":treasure,
                "key":key,
                "sword":sword
                })
    @property
    def state(self):
        d = self._state
        return tuple([d["pos_x"], d["pos_y"], d["life"], d["treasure"],d["key"],d["sword"] ])
    @state.setter
    def state(self, state):
        if _DEBUG_:
            print("-player set state from: "+str(self.state)+" to: "+str(state))
        pos = state[:2]
        self.pos=pos
        self._state = self.init_state(pos, state[2], state[3], state[4], state[5])
    
    def get_state_dict(self):
        return self._state
    
    
#State s: [player.get_state, map.get_state]
class Game():
    def __init__(self, shape=[8,8], dungeon=None, player=None, action=Action):
        if dungeon is None:
            self.dungeon = Dungeon(shape)
        else:
            self.dungeon = dungeon
        if player is None:
            self.player = Player()
        else:
            self.player = player
        
        self.backup_map = self.dungeon.carte.copy()
        
        self.action = action
        self.evenemets = None
        self.game_won=False
        self.isfinished = False
        self.has_moved = False
        
        self.game_info = []
        
        self.init_player_pos()
        self.init_evenements()
        
    def get_possible_actions(self):
        act = [a for a in self.action]
        if self.player.pos[0]==0:
            act.remove(self.action.NORTH)
        if self.player.pos[0]==self.dungeon.shape[0]-1:
            act.remove(self.action.SOUTH)
        if self.player.pos[1]==0:
            act.remove(self.action.WEST)
        if self.player.pos[1]==self.dungeon.shape[1]-1:
            act.remove(self.action.EAST)
        return act
    
    def move(self,a=None):
        if a is None:
            return
        self.player.go_to(self.player.pos + _ACTION_VECTOR_[a])
        self.has_moved = True
        if _GAME_INFO_:
            self.game_info.append("YOU MOVE TO "+str(self.player.pos)+".")
        
    def move_rand(self):
        if _DEBUG_:
            print("move_rand")
        act =  self.get_possible_actions()
        a = act[np.random.randint(len(act))]
        self.move(a)
        
    def clear_current_room(self):
        i,j = self.player.pos
        self.dungeon.clear_room(i,j)
        
    @property
    def current_room(self):
        i,j = self.player.pos
        return self.dungeon.carte[i,j]
    
    def finished(self):
        return self.isfinished
        if _DEBUG_:
            print("finished")
    
    def game_over(self):
        self.isfinished=True
        if _DEBUG_:
            print("=====game_over=====")
        if _GAME_INFO_:
            self.game_info.append("GAME OVER. PLEASE TRY AGAIN!")
        
    def game_win(self):
        self.isfinished=True
        self.game_won=True
        if _DEBUG_:
            print("=====game_win=====")
        if _GAME_INFO_:
            self.game_info.append("GAME FINISH. YOU WIN!")
#        self.clear_current_room()
        
    def get_in_trap(self):
        if _GAME_INFO_:
            self.game_info.append("YOU GET INTO A TRAP!")
        if _DEBUG_:
            print("get_in_trap")
        v = np.random.random()
        if v < 0.6:
            if _DEBUG_:
                print("safe")
            if _GAME_INFO_:
                self.game_info.append("--ESCAPED. YOU ARE SAFE.")
#            self.clear_current_room()
        elif v < 0.9:
            self.init_player_pos()
            if _GAME_INFO_:
                self.game_info.append("--YOU ARE TRANSFERED TO START POINT.")
        else: 
            self.lose_item("life")
            if _GAME_INFO_:
                self.game_info.append("--YOU ARE TRAPPED.")
       
    def get_in_portal(self):
        if _GAME_INFO_:
            self.game_info.append("YOU GET INTO A PORTAL.")
        if _DEBUG_:
            print("get_in_portal")
        cell = (np.random.randint(self.dungeon.shape[0]), 
                         np.random.randint(self.dungeon.shape[1]))
        while(self.dungeon.room.WALL == self.dungeon.carte[cell]):
             cell = (np.random.randint(self.dungeon.shape[0]), 
                         np.random.randint(self.dungeon.shape[1]))
        self.player.go_to(np.array(cell))
        self.has_moved = True
        if _DEBUG_:
            print("player go to "+str(cell))
        if _GAME_INFO_:
            self.game_info.append("YOU ARE TRANSFERED TO "+str(np.array(cell))+".")
        
    def get_in_platform(self):
        if _GAME_INFO_:
            self.game_info.append("YOU GET INTO A PLATEFORM.")
        if _DEBUG_:
            print("get_in_platform")
        act = [a for a in self.action]
        if self.player.pos[0]==0:
            act.remove(self.action.NORTH)
        if self.player.pos[0]==self.dungeon.shape[0]-1:
            act.remove(self.action.SOUTH)
        if self.player.pos[1]==0:
            act.remove(self.action.WEST)
        if self.player.pos[1]==self.dungeon.shape[1]-1:
            act.remove(self.action.EAST)
        a = act[np.random.randint(len(act))]
        if _DEBUG_:
            print("take action "+a.name)
        self.player.go_to( self.player.pos + _ACTION_VECTOR_[a])
        self.has_moved = True
        if _GAME_INFO_:
            self.game_info.append("YOU ARE TRANSFERED TO "+str(self.player.pos)+".")
        
    def get_in_wall(self):
        if _GAME_INFO_:
            self.game_info.append("YOU GET INTO A WALL.")
            self.game_info.append("YOU ARE TRANSFERED TO START POINT.")
        if _DEBUG_:
            print("get_in_wall")
        self.init_player_pos()
        
    def get_in_crack(self):
        if _GAME_INFO_:
            self.game_info.append("YOU CRACK IN THE ROOM.")
        self.lose_item("life")
        
    def beat_enemy(self):
        if _GAME_INFO_:
            self.game_info.append("YOU MEET A MONSTER.")
        if _DEBUG_:
            print("beat_enemy")
        if self.player.has("sword"):
            if _DEBUG_:
                print("player has_sword")
            if _GAME_INFO_:
                self.game_info.append("--YOU HAVE A SWORD. MONSTRE ESCAPED.")
#            self.clear_current_room()
        else:
            if np.random.random() < 0.7:
                if _DEBUG_:
                    print("won battle")
                if _GAME_INFO_:
                    self.game_info.append("--YOU WON THE BATTLE.")
#                self.clear_current_room()
            else:
                if _GAME_INFO_:
                    self.game_info.append("--THE MONSTER WON THE BATTLE.")
                
                self.lose_item("life")
#                self.clear_current_room()
            
    def get_item(self,item):
        if _DEBUG_:
            print("get_item: "+item)
        
        if item=="sword":
            if _GAME_INFO_:
                self.game_info.append("YOU FIND A SWORD.")
            self.player.get("sword")
#            self.clear_current_room()
        elif item=="key":
            if _GAME_INFO_:
                self.game_info.append("YOU FIND A KEY.")
            self.player.get("key")
#            self.clear_current_room()
        elif item=="treasure":
            if _GAME_INFO_:
                self.game_info.append("YOU MEET THE TREASURE.")
            if self.player.has("key"):
                self.player.get("treasure")
                if _GAME_INFO_:
                    self.game_info.append("YOU TAKE THE TREASURE.")
#                self.clear_current_room()
            
    def lose_item(self,item):
        if _DEBUG_:
            print("lose_item: "+item)
        
        if item=="sword":
            if _GAME_INFO_:
                self.game_info.append("YOU LOSE A SWORD.")
            self.player.lose("sword")
        elif item=="key":
            if _GAME_INFO_:
                self.game_info.append("YOU LOSE A KEY.")
            self.player.get("key")
        elif item=="treasure":
            if _GAME_INFO_:
                self.game_info.append("YOU LOSE THE TREASURE.") 
            self.player.lose("treasure")
        elif item=="life":
            if _GAME_INFO_:
                self.game_info.append("YOU LOSE ONE LIFE.") 
            self.player.lose("life")
            self.game_over()
                    
    def init_player_pos(self):
        if _DEBUG_:
            print("init_player_pos")
        self.player.go_to(np.array([self.dungeon.shape[0]-1,self.dungeon.shape[1]-1]))
        self.has_moved = True
        if _GAME_INFO_:
            self.game_info.append("YOU ARE AT THE START POINT.")

        
    def nothing(self):
        if _DEBUG_:
            print("nothing")
        pass
    
    def get_into(self, room):
        if _DEBUG_:
            print("get into Room: "+Room(room).name)
        self.evenemets[room](self)
        
    def has_envenement(self):
        if _DEBUG_:
            print("has_envenement")
        if self.current_room == self.dungeon.room.EMPTY:
            return False
        if self.current_room == self.dungeon.room.TREASURE:
            if self.player.has("key"):
                return True
            else:
                return False
        if self.current_room == self.dungeon.room.START:
            if self.player.has("treasure"):
                return True
            else:
                return False
        return True
    
    def check_envenement(self):
        if _DEBUG_:
            print("check_envenement")
        self.has_moved = False
        self.get_into(self.current_room)
            
    def init_evenements(self):
        self.evenemets = dict({    #   the type of the caracters in the dungeon.
            # starting position
            Room.START : lambda x:x.game_win() if x.player.has("treasure") else x.nothing(),
            # empty cell nothing
            Room.EMPTY : lambda x:x.nothing(),
            # wall (moving to a wall will bounce back to the starting position)
            Room.WALL  : lambda x:x.get_in_wall(),
            # enemy a malicious foe is attacking the adventurer
            # (the adventurer is victorious with p_enemy probability, filambda x:xed to 0.7;
            # otherwise the adventurer is dead).
            Room.ENEMY : lambda x:x.beat_enemy(),
            # trap a trap that either kill the adventurer (with probability 0.1), 
            # bring the adventurer to the starting cell (probability 0.3)
            # or nothing happens (probability 0.6)            
            Room.TRAP  : lambda x:x.get_in_trap(),
            # cracks immediate death            
            Room.CRACK : lambda x:x.get_in_crack(),
            # treasure to open the door of the treasure’s room 
            # it is necessary to have the golden key    
            Room.TREASURE : lambda x:x.get_item("treasure"),
            # magic sword the adventurer can collect the sword and use it in combat;
            # its powers allow him to win any combat without fighting
            Room.SWORD : lambda x:x.get_item("sword"),
             # golden key necessary to open the treasure’s room            
            Room.KEY : lambda x:x.get_item("key"),
            # magic portal a magic portal teleports the agent 
            # to a random (non-wall) cell of the dungeon
            Room.PORTAL : lambda x:x.get_in_portal(),
            # moving platform the pavement is moving! 
            # the adventurer is forced to take refuge in one of the neighbouring cells (at random)            
            Room.PLATFORM: lambda x:x.get_in_platform()
        })
    def new_game(self, shape=[8,8]):
        self.game_info = []
        self.dungeon=Dungeon(shape)
        self.dungeon.init_distribution()
        self.backup_map = self.dungeon.carte.copy()
        
        self.player=Player()
        self.init_player_pos()
         
        self.isfinished=False
        
        self.game_won=False
        
    def restart_game(self):
        self.game_info = []
        self.dungeon.carte = self.backup_map.copy()
        self.player=Player()
        self.init_player_pos()
         
        self.isfinished=False
        
        self.game_won=False
        
        self.evenemets = None
        self.init_evenements()
    
    @property
    def state(self):
        #state = []
        #state.extend(self.player.state)
        #state.extend(self.dungeon.state)
        #return tuple(state)
        return self.player.state
    
    def get_state_dict(self):
        state = []
        state.append(self.player.get_state())
        state.append(self.dungeon.get_state())
        return state
    
    def get_next_state(self, state, action):
        self.player.state = state
        n_states = []
        self.move(action)
        while self.has_moved:
            n_states.append(self.player.state)
            self.check_envenement()
        #n_state = self.get_state()
        #return n_state
        
        if _DEBUG_:
            print("\n-->player state "+str(state)+" go "+str(action))
            print("plyer next state : "+ str(self.player.state))
        return n_states
    
    def print_game_info(self):
        while(len(self.game_info)<10):
            self.game_info.append("")
        print("\n".join(self.game_info))
        self.game_info = []


def read_instance_from(file_name, skip="#"):
    f=open(file_name)
    mat=[] 
    name=[] 
    rev_dict = { v:k for k, v in _ROOMS_LETTRE_.items()}
    for line in f.readlines(): 
        if line.startswith(skip):
            continue
        line = line.replace('\n','')  
        line = line.replace('â–\xa0','■')  
        line = line.replace('W','■')  
        a=line.split(", ")
        #print(a)
        mat.append([rev_dict[v] for v in a])
    mat = np.array(mat, dtype=np.int)
    return mat
    
"""
Evenements happens in the room, it takes game object as variable 
ex: 
    
_DEBUG_EBUG_=True
    
dg = Dungeon()
distribution=[0, 0.2, 0.1, 0.1, 0.1,
              0, 0.1, 0.1, 0.1, 0.1, 0.1]
dg.init_distribution(distribution) 
dg.print_carte()
dg.has_key_sword()

dg.carte[(0,0)]
pl = Player()
pl.state

game = Game(dungeon=dg)
game.player.state

eve = Evenement(game)
eve.game.beat_enemy()
eve.game.player.state

eve.game.dungeon.print_carte()
eve.get_into(Room.ENEMY) 
eve.game.player.state
eve.get_into(Room.KEY)
eve.game.player.state

for i in range(len(Room)): 
    eve.get_into(i)
    eve.game.player.state 
    
    
def init_game(): 
    dg = Dungeon()
    dg.init_distribution()  
    dg.print_carte()
    game = Game(dungeon=dg)
    return game

game = init_game()

def play_a_turn(game):
    #game.player.state
    game.dungeon.print_carte()
    game.move_rand()
    game.check_envenement()
    
play_a_turn(game)


game = Game()
state = game.state
game.dungeon.print_carte()

actions = game.get_possible_actions()
a = actions[np.random.randint(len(actions))]
state = game.get_next_state(state, a)
state

""" 

"""
file_name = "instance.txt"
mat = read_instance_from(file_name)
print(mat)

d = Dungeon()
d.init_from(mat)
d.print_carte()
"""


        