from numpy import median, std
from random import *
import numpy as np
import pandas as pd
import neat

global inputs, outputs
def deal_data(input,out):
    global inputs,outputs
    inputs=input
    outputs=out

#acc
def eval_genomes1(genomes,data_idx,intervals,arg,config):
    #print('The data index is %d',data_idx)
    global inputs,outputs
    eval_input=inputs[data_idx:data_idx+intervals]
    eval_output=outputs[data_idx:data_idx+intervals]
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
        result=accuracy_function(eval,pred)
        genome.fitness = result[0]

#badacc+goodacc
def eval_genomes2(genomes,data_idx,intervals,arg,config):
    #print('The data index is %d',data_idx)
    global inputs,outputs
    eval_input=inputs[data_idx:data_idx+intervals]
    eval_output=outputs[data_idx:data_idx+intervals]
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
        result=accuracy_function(eval,pred)
        genome.fitness = result[1]+result[2]
        #genome.fitness = result[0]

#profit
def eval_genomes3(genomes,data_idx,intervals,arg,config):
    #print('The data index is %d',data_idx)
    global inputs,outputs
    eval_input=inputs[data_idx:data_idx+intervals]
    eval_output=outputs[data_idx:data_idx+intervals]
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
        
        profit=0
        for i in range(len(eval)):
            result=eval[i]
            #print(eval_input[i],eval_output[i],result)
            apply_money=eval_input[i][0]
            return_money=eval_input[i][1]*eval_input[i][2]
            if result==pred[i]:
                if result==0:
                    profit=profit+apply_money
                else:
                    profit=profit+return_money-apply_money
            else:
                if result==0:
                    profit=profit-apply_money
                else:
                    profit=profit-(return_money-apply_money)

        genome.fitness = profit

#proft+history
def eval_genomes4(genomes,data_idx,intervals,arg,config):
    #print('The data index is %d',data_idx)
    global inputs,outputs
    eval_input=inputs[data_idx:data_idx+intervals]
    eval_output=outputs[data_idx:data_idx+intervals]
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
        
        profit=0
        for i in range(len(eval)):
            result=eval[i]
            #print(eval_input[i],eval_output[i],result)
            apply_money=eval_input[i][0]
            return_money=eval_input[i][1]*eval_input[i][2]
            if result==pred[i]:
                if result==0:
                    profit=profit+apply_money
                else:
                    profit=profit+return_money-apply_money
            else:
                if result==0:
                    profit=profit-apply_money
                else:
                    profit=profit-(return_money-apply_money)
        genome.pastfitness=genome.fitness if genome.fitness !=None else 0
        genome.fitness = arg[0]*profit+arg[1]*genome.pastfitness

def eval_test_next(genome,idx,intervals,config):
    global inputs,outputs
    acc=[]
    eval_input=inputs[idx:idx+intervals]
    eval_output=outputs[idx:idx+intervals]
    accuracy = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    pred=[]
    eval=[]
    for i,o in zip(eval_input,eval_output):
        output = net.activate(i)
        aux = np.argmax(output)
        pred.append(aux)
        eval.append(np.argmax(o))
        #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
    return accuracy_function(eval,pred)

def eval_test_all(genome,idx,intervals,config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    global inputs,outputs
    index=idx
    while index+intervals< len(outputs):
        eval_input=inputs[index:index+intervals]
        eval_output=outputs[index:index+intervals]
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
            #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
        results=accuracy_function(eval,pred)
        print(results[0],results[1],results[2])
        index=index+intervals

def accuracy_function(eval,pred):
    
    good_index= np.where(np.array(eval)==1)
    bad_index=np.where(np.array(eval)==0)
    accuracy=0
    good_acc=0
    bad_acc=0

    for i in range(len(eval)):
        result=eval[i]

        if result==pred[i]:

            accuracy=accuracy+1

            if result==0:
                bad_acc=bad_acc+1
            else:
                good_acc=good_acc+1
    #print(accuracy,good_acc,bad_acc)
    good_acc=good_acc/len(list(good_index[0])) if len(list(good_index[0])) != 0 else 0
    bad_acc=bad_acc/len(list(bad_index[0])) if len(list(bad_index[0])) != 0 else 0
    return accuracy/len(pred),good_acc,bad_acc

# def eval_genomes1(genomes,data_idx,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+5000]
#     eval_output=outputs[data_idx:data_idx+5000]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             if(aux == np.argmax(o)): 
#                 AUX+=(output[0] - output[1]) ** 2
#             else:
#                 AUX-=(output[0] - output[1]) ** 2
#         genome.fitness = AUX

# def eval_genomes(genomes,data_idx,intervals,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+intervals]
#     eval_output=outputs[data_idx:data_idx+intervals]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             if(aux == np.argmax(o)): 
#                 AUX+=1
#         genome.fitness = AUX/len(eval_input)
