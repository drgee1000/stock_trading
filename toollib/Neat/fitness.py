from numpy import median, std
from random import *
import numpy as np
import pandas as pd
import neat

class Fitness:
    def __init__(self, inputs, outputs,intervals,arg,size):
        self.inputs=inputs
        self.outputs=outputs
        self.intervals = intervals
        self.arg = arg
        self.size = size
        self.data_idx = 0
        print("size of output: ",self.size)
    #acc
    def eval_genomes1(self,genomes,config):
        #print('The data index is %d',data_idx)

        if((self.data_idx + self.intervals) < self.size):
            eval_input = self.inputs[self.data_idx:self.data_idx + self.intervals]
            eval_output = self.outputs[self.data_idx:self.data_idx + self.intervals]
            self.data_idx = self.data_idx + self.intervals
        else:
            eval_input = self.inputs[self.data_idx:]
            eval_output = self.outputs[self.data_idx:]
            self.data_idx = 0

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
            result=self.accuracy_function(eval,pred)
            genome.fitness = result[0]


    #badacc+goodacc
    def eval_genomes2(self,genomes,config):
        #print('The data index is %d',data_idx)

        if((self.data_idx + self.intervals) < self.size):
            eval_input = self.inputs[self.data_idx:self.data_idx + self.intervals]
            eval_output = self.outputs[self.data_idx:self.data_idx + self.intervals]
            self.data_idx = self.data_idx + self.intervals
        else:
            eval_input = self.inputs[self.data_idx:]
            eval_output = self.outputs[self.data_idx:]
            self.data_idx = 0

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
            result=self.accuracy_function(eval,pred)
            genome.fitness = result[1]+result[2]
            #genome.fitness = result[0]

    #profit
    def eval_genomes3(self,genomes,config):
        #print('The data index is %d',data_idx)

        if((self.data_idx + self.intervals) < self.size):
            eval_input = self.inputs[self.data_idx:self.data_idx + self.intervals]
            eval_output = self.outputs[self.data_idx:self.data_idx + self.intervals]
            self.data_idx = self.data_idx + self.intervals
        else:
            eval_input = self.inputs[self.data_idx:]
            eval_output = self.outputs[self.data_idx:]
            self.data_idx = 0

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
    def eval_genomes4(self,genomes,config):
        #print('The data index is %d',data_idx)

        if((self.data_idx + self.intervals) < self.size):
            eval_input = self.inputs[self.data_idx:self.data_idx + self.intervals]
            eval_output = self.outputs[self.data_idx:self.data_idx + self.intervals]
            self.data_idx = self.data_idx + self.intervals
        else:
            eval_input = self.inputs[self.data_idx:]
            eval_output = self.outputs[self.data_idx:]
            self.data_idx = 0

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
            genome.fitness = self.arg[0]*profit+self.arg[1]*genome.pastfitness

    def eval_test_next(self,genome,idx,intervals,config):
        acc=[]
        eval_input=self.inputs[idx:idx+intervals]
        eval_output=self.outputs[idx:idx+intervals]
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
        return self.accuracy_function(eval,pred)

    def eval_test_all(self,genome,idx,intervals,config, X_test, y_test):
        net = genome
        index=idx
        while index+intervals< len(y_test):
            eval_input=X_test[index:index+intervals]
            eval_output=y_test[index:index+intervals]
            pred=[]
            eval=[]
            for i,o in zip(eval_input,eval_output):
                output = net.activate(i)
                aux = np.argmax(output)
                pred.append(aux)
                eval.append(np.argmax(o))
                #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
            results=self.accuracy_function(eval,pred)
            print(results[0],results[1],results[2])
            index=index+intervals

    def accuracy_function(self,eval,pred):

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
