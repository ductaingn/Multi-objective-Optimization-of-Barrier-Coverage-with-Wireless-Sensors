import csv
import model
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle
import copy

# def write_to_csv(i, epoch, best, best_indi_fitness, pop_avg_fitness, last_id):
def write_to_csv(i, epoch, indi, pop_avg_fitness, last_id):
	with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/results_{epoch}.csv', 'a', newline='') as csv_file:
		fieldnames = ['gen', 'energy', 'active_sensors', 'd_sink_node', 'weights', 'fitness', 'avg_fitness']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writerow({
			'gen': i,
			'energy': indi.f_raw[0],
			'active_sensors': indi.f_raw[1],
			'd_sink_node': indi.f_raw[2],
			'weights': indi.lambdas,
			'fitness': indi.fitness,
			'avg_fitness': pop_avg_fitness[i]
		})
		if i == last_id:
			writer.writerow({
				'gen': 'last',
				'energy': indi.f_raw[0],
				'active_sensors': indi.f_raw[1],
				'd_sink_node': indi.f_raw[2],
				'weights': indi.lambdas,
				'fitness': indi.fitness,
				'avg_fitness': pop_avg_fitness[last_id]
			})


if __name__ == "__main__":
	POP_SIZE = 20
	NEIGHBORHOOD_SIZE = 3
	NUM_SENSORS = 300
	NUM_SINK_NODES = 1
	NUM_GENERATION = 30000
	LENGTH, WIDTH = 1000, 50
	NUM_EPOCH = 5

	# Load positions
	with open(f'Datasets/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/sensors_positions_1.pickle','rb') as file:
		sensors_positions = pickle.load(file)
	with open('Datasets/sink_nodes_positions.pickle','rb') as file:
		sink_nodes_positions = pickle.load(file)
	
	# Run
	for epoch in range(NUM_EPOCH):

		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/results_{epoch}.csv', 'w', newline='') as csv_file:
			fieldnames = ['gen', 'energy', 'active_sensors', 'd_sink_node', 'weights', 'fitness', 'avg_fitness']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
	
		population = model.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sensors_positions,NUM_SINK_NODES,sink_nodes_positions)

		best_indi_fitness = []
		pop_avg_fitness = []
		objectives_by_generations = []
		first_solutions = [indi.solution for indi in population.pop]
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/first_solutions_{epoch}.pickle','wb') as file:
			pickle.dump(first_solutions,file)

		for i in range(NUM_GENERATION):
			pop_fitness = [indi.fitness for indi in population.pop]	
			population.reproduct()
			f = []
			for indi in population.pop:
				f.append(copy.deepcopy(indi.f))
			objectives_by_generations.append(f)
			best = sorted(population.pop,key= lambda x:x.fitness)[-1]
			best_indi_fitness.append(best.fitness)
			pop_avg_fitness.append(np.mean(pop_fitness))
			external_pop = population.EP

			if(i%100==0):
				print(i/NUM_GENERATION*100,'%')
				for indi in population.EP:
					# print(indi)
					write_to_csv(i, epoch, indi, pop_avg_fitness, NUM_GENERATION-1)
		
		last_solutions = [indi.solution for indi in population.pop]		
		# Change file name everytime!
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/last_solutions_{epoch}.pickle','wb') as file:
			pickle.dump(last_solutions,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/objectives_by_generations_{epoch}.pickle','wb') as file:
			pickle.dump(objectives_by_generations,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/pop_avg_fitness_{epoch}.pickle','wb') as file:
			pickle.dump(pop_avg_fitness,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/best_indi_fitness_{epoch}.pickle','wb') as file:
			pickle.dump(best_indi_fitness,file)