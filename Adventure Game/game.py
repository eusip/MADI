# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:40:58 2018

@author: you
"""

_LINUX_=False


import msvcrt
if _LINUX_:
    import getch
    
import time
import numpy as np
from basic import Game, Action
import os
class InteractiveGame():
    def __init__(self):
        self.game = Game(); 
        self.game.new_game()
        self.keys=[b'x',b'q',b's',b'd',b'z']
        self.keys_linux=['x','a','s','d','w']
        
   
    def get_input(self):
        while True:
            time.sleep(0.1)
            if _LINUX_ :
                pressedKey = getch.getch()
                if pressedKey in self.keys_linux:  
                    #print("press ", pressedKey)
                    return pressedKey
            else:
                pressedKey = msvcrt.getch()
                if pressedKey in self.keys:  
                    #print("press ", pressedKey)
                    return pressedKey
                
    def print_game(self):
        pass
    def show_help(self):
        print("Game controlls: ")
        print(dict({
                "q":"LEFT",
                "d":"RIGHT",
                "z":"UP",
                "s":"DOWN",
                "x":"QUIT",
                }))
        print("Good Luck!\n\n")
        pass
    
    def start_game(self):
        self.show_help() 
        while self.game.finished() is False:
            time.sleep(0.01)
            m = self.game.dungeon.carte.copy()
            i,j = self.game.player.pos
            m[i,j]=-1
            
            if _LINUX_:
                os.symtem('clear')
            else:
                os.system('cls')
                
            self.game.dungeon.print_carte(m)
            self.game.print_game_info()
            
            print("\nPlayer: ", self.game.player.state)
            print("\nDungeon: ", self.game.dungeon.state)
            print("\nGame: ", self.game.state)
            
            possible_actions = self.game.get_possible_actions()
            if _LINUX_:
                keys = self.keys_linux
            else:
                keys = self.keys
                
            ch = self.get_input()
            if ch==keys[0] : #exit
                break  
            elif ch==keys[1] : #left
                if Action.WEST in possible_actions:
                     self.game.move(Action.WEST)
                else:
                    continue
            elif ch==keys[3] : #right
                if Action.EAST in possible_actions:
                     self.game.move(Action.EAST)
                else:
                    continue
            elif ch==keys[4] : #up
                if Action.NORTH in possible_actions:
                     self.game.move(Action.NORTH)
                else:
                    continue
            elif ch==keys[2] : #down
                if Action.SOUTH in possible_actions:
                     self.game.move(Action.SOUTH)
                else:
                    continue
                
            while self.game.has_moved:
                self.game.check_envenement()
                
            if _LINUX_:
                os.symtem('clear')
            else:
                os.system('cls')
        self.game.dungeon.print_carte(self.game.dungeon.carte.copy())
        self.game.print_game_info()
            
game = InteractiveGame()
game.start_game()