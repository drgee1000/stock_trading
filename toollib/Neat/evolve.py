from __future__ import print_function
import os
import neat
from numpy import median, std
from random import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler#Scaling
from toollib.Neat.fitness import Fitness
import sys
from collections import Counter

class Neat:
    def __init__(self,fitness_index,intervals,arg1,arg2):
        self.inputs = []
        self.outputs = []
        self.fitness_index = fitness_index
        self.intervals = intervals
        self.arg = []
        self.arg.append(arg1)
        self.arg.append(arg2)
        self.winner_net = None

    def Feature_Scaling(self,data,label):
        X=data.drop(labels=label, axis=1, inplace=False)
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X[label]=data[label]
        return X



    def eval_genomes(self,genomes, config):
        for genome_id, genome in genomes:
            AUX = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for i,o in zip(self.inputs,self.outputs):
                output = net.activate(i)
                #print(type(output),output)
                aux = [0 if randint(0,100)/100 <= 1-i else 1 for i in output]
                #print(type(aux),output,aux,o)
                #aux = tuple(aux)
                if(aux[0] == o): AUX+=1
            genome.fitness = AUX/len(self.inputs)



    def run(self,config_file, X_train, X_test, y_train, y_test):
        # Load configuration.
        self.outputs=y_train
        self.inputs=X_train

        outputs = y_train
        inputs = X_train
        f = Fitness(inputs, outputs,self.intervals,self.arg,len(outputs))

        fitnessways = ['accuracy', 'badaccgoodacc', 'profit', 'profit_history']
        eval_fitness = [f.eval_genomes1, f.eval_genomes2, f.eval_genomes3, f.eval_genomes4]
        print(fitnessways[self.fitness_index] + '_interval' + str(self.intervals) + '_args' + str(
            self.arg) + '_data20172018')

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run for up to 300 generations.
        winner = p.run(eval_fitness[self.fitness_index],n = 300)
        self.winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        f.eval_test_all(self.winner_net, 0, self.intervals, config, X_test, y_test)
        #print(stats.get_fitness_mean())


    def predict(self,X_test):
        output_ = []
        for i in X_test:
            output = self.winner_net.activate(i)
            output_.append(output[0])
        print(output_)
        return output_

