"""
Date: 03/2019

Author: XD

Target: Travelling Salesman Problem (TSP): Given a set of cities and distance between every pair of cities, the problem is to find the shortest possible route that visits every city exactly once and returns to the starting point.

Ref: https://morvanzhou.github.io/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np


N_cities = 12
cross_rate =0.5
mutate_rate = 0.02
Population_size = 100
num_epoch = 100


class GeneticAlgo(object):
	def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
		self.DNA_size = DNA_size
		self.cross_rate = cross_rate
		self.mutate_rate = mutation_rate
		self.pop_size = pop_size

		self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])


	def translateDNA(self, pop, city_position):    
		# coordinate of cities
		# The DNA here is the permutation of the order of cities
		city_x = np.empty_like(pop, dtype=np.float64)
		city_y = np.empty_like(pop, dtype=np.float64)
		for i, d in enumerate(pop):
			city_coordinate = city_position[d]
			city_x[i, :] = city_coordinate[:, 0]
			city_y[i, :] = city_coordinate[:, 1]            
		return city_x, city_y



	def get_fitness(self, city_x, city_y):
		total_distance = np.empty((city_x.shape[0],), dtype=np.float64)

		for i, (xs, ys) in enumerate(zip(city_x, city_y)):
			total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
		
		# print(city_x)
		# use exponential to enlarge the difference
		fitness = np.exp(self.DNA_size * 2 / total_distance)

		return fitness, total_distance


	def select(self, fitness):
		idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
		return self.pop[idx]


	def crossover(self, parent, pop):
		if np.random.rand() < self.cross_rate:
			i_ = np.random.randint(0, self.pop_size, size=1)     
			# print('i_',i_)                   
			# select another individual from pop
			cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool) 
			# print(cross_points) 
			# choose crossover points
			keep_city = parent[~cross_points]                                       
			# find the city number
			# print(~cross_points)
			# print('parent',parent)
			# print(keep_city)
			complement_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
			# print('rest of cities',complement_city)
			parent[:] = np.concatenate((keep_city, complement_city))
			# print(parent)
		return parent


	def mutate(self, child):
		for point in range(self.DNA_size):
			if np.random.rand() <  self.mutate_rate:
				swap_point = np.random.randint(0, self.DNA_size)
				swapA, swapB = child[point], child[swap_point]
				child[point], child[swap_point] = swapB, swapA
		return child


	def evolve(self, fitness):
		pop = self.select(fitness)
		pop_copy = pop.copy()
		# print('pop',pop)
		for parent in pop:  # for every parent
			child = self.crossover(parent, pop_copy)
			child = self.mutate(child)
			parent[:] = child
		self.pop = pop


class TSP(object):
	def __init__(self, n_cities):
		self.city_position = np.random.rand(n_cities, 2)
		plt.ion()
		self.n_cities = n_cities

	def plotting(self, lx, ly, total_d):
		plt.cla()
		plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=20, c='k')
		for i, txt in enumerate(np.arange(self.n_cities)):
			plt.annotate(i, (self.city_position[:, 0][i].T, self.city_position[:, 1][i].T))
		
		plt.plot(lx.T, ly.T, 'g-')
		plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'green'})
		# print(self.city_position[:, 0])

		plt.xlim((-0.1, 1.1))
		plt.ylim((-0.1, 1.1))
		plt.pause(0.01)


tsp = GeneticAlgo(DNA_size=N_cities, cross_rate=cross_rate, mutation_rate=mutate_rate, pop_size=Population_size)

env = TSP(N_cities)

for i_ in range(num_epoch):
	lx, ly = tsp.translateDNA(tsp.pop, env.city_position)
	# print('lx',lx)
	# print('ly',ly)
	# print(env.city_position)

	fitness, total_distance = tsp.get_fitness(lx, ly)
	tsp.evolve(fitness)
	best_idx = np.argmax(fitness)
	print('Epoch: {} | route: {} | distance: {} ' .format(i_, tsp.pop[best_idx], total_distance[best_idx]))
	
	env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()

