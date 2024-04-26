import model
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle

if __name__ == "__main__":
	POP_SIZE = 20
	NEIGHBORHOOD_SIZE = 3
	NUM_SENSORS = 100
	NUM_SINK_NODES = 1
	NUM_EVALUATION = 100
	# sensors_x = np.random.uniform(low=0,high=1000,size=(NUM_SENSORS))
	# sensors_y = np.random.uniform(low=0,high=10,size=(NUM_SENSORS))
	# sensors_positions = np.array([[sensors_x[i],sensors_y[i]] for i in range(NUM_SENSORS)])
	# sensors_positions.sort(axis=0)

	# sink_nodes_x = np.random.uniform(low=0, high=1000, size=(NUM_SINK_NODES))
	# sink_nodes_y = np.random.uniform(low=0, high=10,size=(NUM_SINK_NODES))
	# sink_nodes_positions = [[sink_nodes_x[i],sink_nodes_y[i]] for i in range(NUM_SINK_NODES)]
	with open('sensor_positions.pickle','rb') as file:
		sensors_positions = pickle.load(file)
	with open('sink_nodes_positions.pickle','rb') as file:
		sink_nodes_positions = pickle.load(file)
		
	pop = model.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sensors_positions,NUM_SINK_NODES,sink_nodes_positions)

	fitness = []
	for i in range(NUM_EVALUATION):
		pop.reproduct()
		best = sorted(pop.pop,key= lambda x:x.fitness)[-1]
		fitness.append(best.fitness)
		if(i%100==0):
			print(i/NUM_EVALUATION*100,'%')
	with open('result.pickle','wb') as file:
		pickle.dump(pop,file)
	# write population to txt file
	with open('dataset/result.txt','w') as file:
		for i in range(POP_SIZE):
			file.write(str(pop.pop[i])+'\n')
		
	with open('sensor_positions.pickle','wb') as file:
		pickle.dump(sensors_positions,file)
	sensors_positions_str = np.array2string(sensors_positions)
	with open('dataset/sensors_positions.txt','wb') as file:
		file.write(sensors_positions_str.encode())
  
	with open('sink_nodes_positions.pickle','wb') as file:
		pickle.dump(sink_nodes_positions,file)
	# # print(sink_nodes_positions)
	with open('dataset/sink_nodes_positions.txt','w') as file:
		file.write(str(sink_nodes_positions))

	Plot.Plot_solution(pop.pop[0].sensors_positions, pop.pop[0].solution)

	plt.plot(fitness)
	plt.show()
	print()