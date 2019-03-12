"""
Date: 03/2019

Author: XD

Target: Learn to use Genetic Algorithm to find a peak Relative Humidity in a year.

Dataset: Air Quality Data Set https://archive.ics.uci.edu/ml/datasets/Air+Quality

Reference: 
1.S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario, Sensors and Actuators B: Chemical, Volume 129, Issue 2, 22 February 2008, Pages 750-757, ISSN 0925-4005,
2. https://morvanzhou.github.io/tutorials/

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('AirQualityUCI.xlsx', sheet_name='AirQualityUCI')
RH = df['RH']


DNA_length = 30           
Population_size = 100          
Cross_rate = 0.8         
Mutation_rate = 0.005   
num_epoch = 100    
X_range = [0, 365]
start_point = 0# np.random.randint(0, RH.size-364) 
data_range = [start_point, start_point+365]



def get_fitness(x): 
	return np.squeeze(np.where(RH[x]>=0, RH[x], 0)) # delete invalid negative data


def translateDNA(pop): # convert binary 'pop' to decimal value within range X_range
	DNA = pop.dot(2 ** np.arange(DNA_length)[::-1]) / float(2**DNA_length-1) * X_range[1]
	return [int(round(dna)) for dna in DNA]


def select(pop, fitness):    
	idx = np.random.choice(np.arange(Population_size), size=Population_size, replace=True,
						   p=fitness/fitness.sum())
	return pop[idx]


def crossover(parent, pop):    
	if np.random.rand() < Cross_rate:
		# select another individual from pop
		i_ = np.random.randint(0, Population_size, size=1)   
		# choose crossover points
		cross_points = np.random.randint(0, 2, size=DNA_length).astype(np.bool)
		# mating and produce one child
		# parent[bollean, boollean...boolean] = pop[i_, [bollean, boollean...boolean]]
		parent[cross_points] = pop[i_, cross_points]
	return parent


def mutate(child):
	for point in range(DNA_length):
		if np.random.rand() < Mutation_rate:
			child[point] = 1 if child[point] == 0 else 0
	return child


pop = np.random.randint(2, size=(Population_size, DNA_length))   # initialize the pop DNA

plt.ion()       
x = np.linspace(*data_range, num=366)   # Unpacking Argument Lists
plt.plot(np.linspace(*X_range, num=366) , get_fitness(x))

for _ in range(num_epoch):
	fitness = get_fitness(translateDNA(pop))    # compute function value by extracting DNA

	if 'sca' in globals(): sca.remove()
	sca = plt.scatter(translateDNA(pop), fitness, s=100, lw=0, c='red', alpha=0.5); plt.pause(0.05)

	print("iteration {} Most fitted DNA:{} ".format(_, pop[np.argmax(fitness), :]))
	pop = select(pop, fitness)

	pop_copy = pop.copy()

	for parent in pop:
		child = crossover(parent, pop_copy)
		child = mutate(child)
		parent[:] = child       # parent is replaced by its child

plt.ioff()
plt.show()
