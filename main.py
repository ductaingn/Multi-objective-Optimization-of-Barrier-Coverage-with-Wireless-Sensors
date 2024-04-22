import model
import numpy as np

if __name__ == "__main__":
    POP_SIZE = 20
    NEIGHBORHOOD_SIZE = 20
    NUM_SENSORS = 100
    NUM_SINK_NODES = 1
    NUM_EVALUATION = 1000
    sensor_positions = np.random.randn(NUM_SENSORS,2)
    sink_nodes_positions = np.random.randn(NUM_SINK_NODES, 2)

    sensor_positions = sensor_positions[sensor_positions[:,0].argsort()]
    # print(sensor_positions)
    pop = model.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sensor_positions,NUM_SINK_NODES,sink_nodes_positions)
    
    for _ in range(NUM_EVALUATION):
        pop.reproduct()
