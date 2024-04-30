import numpy as np
import scipy.stats as stats
import pickle

def generate_positions(number_sensors, length=1000, width=50, distribution='uniform'):
    if(distribution=='uniform'):
        sensors_x = np.random.uniform(low=0,high=length,size=(number_sensors))
        sensors_y = np.random.uniform(low=-width/2,high=width/2,size=(number_sensors))
        sensors_positions = np.array([[sensors_x[i],sensors_y[i]] for i in range(number_sensors)])
        sensors_positions.sort(axis=0)

    if(distribution=='gauss'):
        mu = length/2
        sigma = mu/5
        lower, upper = 0, length
        sensors_x = np.array(stats.truncnorm((lower-mu)/sigma,(upper-mu)/sigma,loc=mu, scale=sigma).rvs(number_sensors))
        sensors_y = np.random.uniform(low=-width/2,high=width/2,size=(number_sensors))
        sensors_positions = np.array([[sensors_x[i],sensors_y[i]] for i in range(number_sensors)])
        sensors_positions.sort(axis=0)

    return sensors_positions

for i in range(5):
    length = 1000
    width = 50
    for num_sensors in [100,300,700]:
        sensors_positions = generate_positions(num_sensors, length, width, 'uniform')
        with open(f'Datasets/uniform/{width}x{length}unit/{num_sensors}sensors/sensors_positions_{i}.pickle','wb') as file:
            pickle.dump(sensors_positions,file)


with open('Datasets/sink_nodes_positions.pickle','wb') as file:
    sink_node_pos = [-25,500]
    pickle.dump(sink_node_pos,file)
        