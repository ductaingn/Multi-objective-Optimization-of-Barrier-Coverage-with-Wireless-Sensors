import model
import numpy as np

if __name__ == "__main__":
    POP_SIZE = 20
    NEIGHBORHOOD_SIZE = 20
    NUM_SENSORS = 1000
    NUM_SINK_NODES = 1
    sink_nodes_positions = np.random.randn(shape=(NUM_SENSORS,2))
    sink_nodes_positions = np.random.randn(2)

    pop = model.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sink_nodes_positions,NUM_SINK_NODES,sink_nodes_positions)