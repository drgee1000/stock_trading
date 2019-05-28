from __future__ import print_function
import os
import neat
from numpy import median, std
from random import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler#Scaling
from collections import Counter

class Neat:
    def __init__(self):
        self.inputs = []
        self.outputs = []
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
        winner = p.run(self.eval_genomes,10)
        print(stats.get_fitness_mean())
        #print(stats.get_fitness_best())
        # Display the winning genome.
        #print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        #print('\nOutput:')
        test_inputs = []
        test_outputs = y_test
        test_inputs=X_test
        self.winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        accuracy = 0
        for i,o in zip(test_inputs,test_outputs):
            output = self.winner_net.activate(i)
            aux = np.argmax(output)
            if(aux == np.argmax(o)): accuracy+=1
            #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
        accuracy = accuracy/len(test_inputs)
        print('Final accuracy {!r}'.format(accuracy))

    def predict(self,X_test):
        output_ = []
        for i in X_test:
            output = self.winner_net.activate(i)
            output_.append(output[0])
        print(output_)
        return output_

