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
print('ssssss')

def Feature_Scaling(data,label):
    X=data.drop(labels=label, axis=1, inplace=False)
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X[label]=data[label]
    return X

label='y'
inputs = []
outputs = []
data=Feature_Scaling(pd.read_csv('//home//cs-liuy//data//China_v.csv'),'y')
kf=KFold(n_splits=4,random_state=22).split(data)

index=[[train_index, test_index] for _, (train_index, test_index) in enumerate(kf)]
train=data.iloc[index[0][0], :]
test=data.iloc[index[0][1], :]
outputs=list(train['y'])
X=train.drop(labels=label, axis=1, inplace=False)
inputs=X.values.tolist()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i,o in zip(inputs,outputs):
            output = net.activate(i)
            #print(type(output),output)
            aux = [0 if randint(0,100)/100 <= 1-i else 1 for i in output]
            #print(type(aux),output,aux,o)
            #aux = tuple(aux)
            if(aux[0] == o): AUX+=1
        genome.fitness = AUX/len(inputs)

test_inputs = []
test_outputs = list(test['y'])
X_=test.drop(labels=label, axis=1, inplace=False)
test_inputs=X_.values.tolist()

def run(config_file):
    # Load configuration.
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
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    accuracy = 0
    for i,o in zip(test_inputs,test_outputs):
        output = winner_net.activate(i)
        aux = [0 if randint(0,100)/100 <= 1-i else 1 for i in output]
        aux = tuple(aux)
        print("input {!r}, expected output , got {!r}".format(o, aux))
        if(aux[0] == o): accuracy+=1
    accuracy = accuracy/len(test_inputs)
    print('Final accuracy {!r}'.format(accuracy))
    
    #node_names = {-1:'A', -2: 'B', -3: 'C',-4: 'D',0:'a',1: 'b',2: 'c'}
    visualize.draw_net(config, winner, False)
    #visualize.plot_stats(stats, ylog=False, view=False)
    #visualize.plot_species(stats, view=False)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)

#Depois Tentar entender tanto essa parte do __main__ quando o modulo visualize, 
#e a nn dessa neat
