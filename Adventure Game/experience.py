# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:43:52 2019

@author: you
"""

#experience: 
#definition of experience: 
# play a round, get the runtime 
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from mdp import MDPGame
from qlearning import Qlearning

_DEBUG_ = False
_GAME_INFO_ = False

class Experience:
    def __init__(self, experience, dg_size_range=[4,16], dg_size_step=2, iterations=20):
        self.experience = experience
        self.dg_size_range = dg_size_range
        self.dg_size_step = 1
        self.iterations = iterations
        self.dg_sizes = [i for i in range(self.dg_size_range[0], self.dg_size_range[1], self.dg_size_step)]
        self.time_excution = np.zeros(len(self.dg_sizes))
    def run(self):
        for i in range(len(self.dg_sizes)):
            size = self.dg_sizes[i]
            start_time = time.clock()
            for j in range(self.iterations):
                self.experience.init_experience(size)
                self.experience.run()
            end_time = time.clock()-start_time 
            self.time_excution[i] = end_time / self.iterations
    
    def crate_image(self, name):
        xs = self.dg_sizes
        ys = self.time_excution
        plt.figure(figsize=(10,10))
        plt.plot(xs,ys)
        plt.title('The execution time over size of instance: '+name)
        plt.xlabel('Size of dungeon')
        plt.ylabel('Time of execution')
        plt.savefig('exp_'+name+'.png')

if len(sys.argv)>1 and sys.argv[1] == "mdp":
    game = MDPGame()
    name = ""
    if len(sys.argv)>2:
        if sys.argv[2] == "v":
            game.set_solver("value_iteration")
            name='mdp_value_iter'
        elif sys.argv[2] == "p":
            game.set_solver("politic_iteration")
            name='mdp_politic_iter'
    else:
        game.set_solver("politic_iteration")
        name='mdp_politic_iter'
    exps = Experience(game)
    exps.run()
    exps.crate_image(name)
    
elif len(sys.argv)>1 and sys.argv[1] == "qlearning": 
    #Qlearning should implement:
    #Qlearning.init_experience(size) for initialization
    #Qlearning.run() for execution of solving
    game = Qlearning()
    exps = Experience(game)
    exps.run()
    exps.crate_image(name)
    
    
    
        
        
            
            