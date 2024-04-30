import model
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle
import copy

if __name__ == "__main__":
	POP_SIZE = 20
	NEIGHBORHOOD_SIZE = 3
	NUM_SENSORS = 100
	NUM_SINK_NODES = 1
	NUM_GENERATION = 50000
	LENGTH, WIDTH = 10000, 50

	with open(f'Datasets/uniform/{WIDTH}x{LENGTH}units/{NUM_SENSORS}sensors/sensor_positions_0.pickle','rb') as file:
		sensors_positions = pickle.load(file)
	with open('Datasets/sink_nodes_positions.pickle','rb') as file:
		sink_nodes_positions = pickle.load(file)
		
	pop = model.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sensors_positions,NUM_SINK_NODES,sink_nodes_positions)

	fitness = []
	objectives_by_generations = []
	for i in range(NUM_GENERATION):
		pop.reproduct()
		f = []
		for indi in pop.pop:
			f.append(copy.deepcopy(indi.f))
		objectives_by_generations.append(f)
		best = sorted(pop.pop,key= lambda x:x.fitness)[-1]
		fitness.append(best.fitness)
		if(i%100==0):
			print(i/NUM_GENERATION*100,'%')
	
	plt.plot(fitness)
	plt.show()
	
	# Change file name everytime!
	with open(f'Result/uniform/{WIDTH}x{LENGTH}units/{NUM_SENSORS}sensors/objectives_by_generations_0.pickle','wb') as file:
		pickle.dump(objectives_by_generations,file)
	with open(f'Result/uniform/{WIDTH}x{LENGTH}units/{NUM_SENSORS}sensors/fitness_0.pickle','wb') as file:
		pickle.dump(fitness,file)