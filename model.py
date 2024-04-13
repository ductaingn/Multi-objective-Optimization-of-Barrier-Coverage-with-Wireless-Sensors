import numpy as np
from sklearn.neighbors import NearestNeighbors

class Individual:
    def __init__(self,lambdas, num_sensors, num_sink_nodes, sensors_positions, sink_node_positions) -> None:
        self.num_sensors = num_sensors
        self.num_sink_nodes = num_sink_nodes
        self.lambdas = lambdas
        self.sensors_positions = sensors_positions
        self.sink_nodes_positions = sink_node_positions
        self.mu = 0.99
        
        # Random solution
        activate = np.random.choice([0,1],num_sensors)
        srange = np.random.rand(num_sensors)
        self.solution = [[activate[i], srange[i]] for i in range(num_sensors)]
        self.repair_solution()

        self.fitness = self.compute_fitness(self.solution)


    def compute_fitness(self, solution):
        z = [0,0,0]
        f = [0,0,0] 
        for i in range(self.num_sensors):
            if(solution[i][0]==1):
                f[0] += solution[i][1]**2
                f[1] += 1

                nearest_sink_node_distance = 1e9
                for j in range(self.num_sink_nodes):
                    # print(self.sensors_positions[i])
                    # print(self.sink_nodes_positions[j])
                    distance = np.abs((self.sensors_positions[i][0]-self.sink_nodes_positions[j][0])**2 + (self.sensors_positions[i][1]-self.sink_nodes_positions[j][1])**2)
                    nearest_sink_node_distance = min(nearest_sink_node_distance, distance)
            
                f[2] += nearest_sink_node_distance
        f[2]/=f[1]

        gte = max([self.lambdas[i]*abs(f[i]-z[i]) for i in range(3)])
        return gte
                   
    def repair_solution(self):
        
        return
    
    def update_utility(self, new_solution):
        delta_i = self.compute_fitness(new_solution) - self.fitness
        if(delta_i>0.001):
            return 1
        else:
            return 0.99 + 0.01*delta_i/0.001

    
class Population:
    def __init__(self, pop_size, neighborhood_size, num_sensors, sensors_positions,num_sink_node, sink_nodes_positions) -> None:
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.lambdas = self.generate_lambdas()
        self.pop:list[Individual] = []
        for i in range(self.pop_size):
            self.pop.append(Individual(self.lambdas[i], num_sensors, num_sink_node, sensors_positions, sink_nodes_positions))

        self.neighbor = {} # Use KNN/... to find neighbors of each sub-problem
        def find_neighbor(self):
            # max value for distance to neighbor
            X = np.array(self.lambdas)
            nbrs = NearestNeighbors(n_neighbors=self.neighborhood_size, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            for i in range(len(self.lambdas)):
                self.neighbor[i] =list( indices[i])

    def __repr__(self) -> str:
        # print every individual in population
        res = ""
        for i in range(self.pop_size):
            res += f"Solution to Individual {i}: {self.pop[i].solution}\n"
        return res

    # Genrate uniformly spread weighted vectors lambda 
    def generate_lambdas(self):
        sub_problem_lambdas = []
        for _ in range(3):
            sub_problem_lambdas.append(np.random.uniform(0,1,self.pop_size))

        res = []
        for i in range(self.pop_size):
            sum = 0
            for j in range(3):
                sum += sub_problem_lambdas[j][i]
            res.append([sub_problem_lambdas[j][i]/sum for j in range(3)])
        
        return res
    
    def selection(self):
        return
    
    def forward_local_search(self, k):
        # find one gene with max range in pop[k].solution
        max_range = 0
        max_range_index = 0
        for i in range(len(self.pop[k].solution)):
            if self.pop[k].solution[i][1]>max_range:
                max_range = self.pop[k].solution[i][1]
                max_range_index = i
        for i in range(len(self.pop[k].solution)):
            if i==max_range_index:
                self.pop[k].solution[i][0] = 1
            else:
                self.pop[k].solution[i][0] = 0

        # repair solution


    def local_search(self, k):       
        j = np.random.choice(self.pop_size)
        if (j < k):
            for _ in range (k-j):
                self.forward_local_search(k)
        else:
            for _ in range (j-k):
                self.backward_local_search(k)
                
    def mutation(self):
        return

    def update(self):
        return
    
    def update_utility(self):
        return
    
