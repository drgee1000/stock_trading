from __future__ import print_function
import os
import neat
from numpy import median, std
from random import *
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler#Scaling
import visualize
from collections import Counter

def Feature_Scaling(data,label):
    X=data.drop(labels=label, axis=1, inplace=False)
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X[label]=data[label]
    return X

def code_answer(attributer):
    ans = np.zeros((attributer.shape[0],len(Counter(attributer))), dtype="int")   #len(Counter(attributer))属性的长度
    attributer=list(attributer)
    first=attributer[0]
    for i in range(len(attributer)):
        if attributer[i] == first: 
            ans[i][1] = 1
        else:
            ans[i][0] = 1
    return ans

label='Creditability'
global inputs,outputs,tttt
inputs = []
outputs = []
tttt=0
data=Feature_Scaling(pd.read_csv('~//data//German_v.csv'),label)
kf=KFold(n_splits=4,random_state=22).split(data)

def eval_genomes(genomes,config):
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,o in zip(inputs,outputs):
            output = net.activate(i)
            aux = np.argmax(output)
            if(aux == np.argmax(o)): 
                AUX+=(output[0] - output[1]) ** 2
            else:
                AUX-=(output[0] - output[1]) ** 2
        genome.fitness = AUX


def run(config_file):
    # Load configuration.
    acc=[]
    for _, (train_index, test_index) in enumerate(kf):
        train=data.iloc[train_index, :]
        test=data.iloc[train_index, :]
        global inputs,outputs
        outputs=code_answer(train[label])
        X=train.drop(labels=label, axis=1, inplace=False)
        inputs=X.values.tolist()
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
        winner = p.run(eval_genomes, 100)
        print(stats.get_fitness_mean())
        print(stats.get_fitness_best())
        # Display the winning genome.
        #print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        #print('\nOutput:')
        test_inputs = []
        test_outputs = code_answer(test[label])
        X_=test.drop(labels=label, axis=1, inplace=False)
        test_inputs=X_.values.tolist()
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        accuracy = 0
        for i,o in zip(test_inputs,test_outputs):
            output = winner_net.activate(i)
            aux = np.argmax(output)
            if(aux == np.argmax(o)): accuracy+=1
            #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
        accuracy = accuracy/len(test_inputs)
        acc.append(accuracy)
        print('Final accuracy {!r}'.format(accuracy))
    print(acc)
    print('---average,%f'%np.mean(acc))



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
