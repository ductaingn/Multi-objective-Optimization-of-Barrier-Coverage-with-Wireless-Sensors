import numpy as np
from sklearn.neighbors import NearestNeighbors

# 1 Individual contains 1 sub-problem and 1 solution
class Individual:
    def __init__(self,lambdas, num_sensors, num_sink_nodes, sensors_positions, sink_node_positions, ideal_point) -> None:
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

        self.f = [1e9,1e9,1e9]
        self.fitness = self.compute_fitness(self.solution, ideal_point)


    def compute_fitness(self, solution,ideal_point):
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
        self.f = f
        gte = max([self.lambdas[i]*abs(f[i]-ideal_point[i]) for i in range(3)])
        return gte
    
    def mutation(self):

        return
                   
    def repair_solution(self):
        
        return
    
    def update_utility(self, new_solution):
        delta_i = self.compute_fitness(new_solution) - self.fitness
        if(delta_i>0.001):
            return 1
        else:
            return 0.99 + 0.01*delta_i/0.001

    
class Population:
    def __init__(self, pop_size, neighborhood_size, num_sensors, sensors_positions,num_sink_nodes, sink_nodes_positions) -> None:
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.num_sensors = num_sensors
        self.num_sink_nodes = num_sink_nodes
        self.lambdas = self.generate_lambdas()
        self.pop:list[Individual] = []
        self.ideal_point = [0,0,0]
        self.EP = []
        for i in range(self.pop_size):
            self.pop.append(Individual(self.lambdas[i], num_sensors, self.num_sink_nodes, sensors_positions, sink_nodes_positions, self.ideal_point))

        def find_neighbor():
            # max value for distance to neighbor
            X = np.array(self.lambdas)
            nbrs = NearestNeighbors(n_neighbors=self.neighborhood_size, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            neighbor = {}
            for i in range(len(self.lambdas)):
                neighbor[i] =list( indices[i])
            return neighbor
        
        self.neighbor = find_neighbor()

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
    
  
    def forward_local_search(self, k):
        # find one gene with max range in pop[k].solution
        max_range = 0
        max_range_index = 0
        for i in range(len(self.pop[k].solution)):
            if self.pop[k].solution[i][1] > max_range:
                max_range = self.pop[k].solution[i][1]
                max_range_index = i
        

        # repair solution

    def local_search(self, k):       
        j = np.random.choice(self.pop_size)
        if (j < k):
            for _ in range (k-j):
                self.forward_local_search(k)
        else:
            for _ in range (j-k):
                self.backward_local_search(k)

    def selection(self, k=16)->Individual:
        indi_index = list(np.random.choice(range[0,self.pop_size],k))
        # k is number of individuals in selection pool
        while(k>2):
            i = 0
            for i in range(0,k-1,2):
                if(self.pop[indi_index[i]].mu>self.pop[indi_index[i+1]].mu):
                    indi_index.pop(i+1)
                else:
                    indi_index.pop(i)
            k/=2
        
        return sorted([self.pop[i] for i in indi_index])[-1]

    def update_utility(self, individuals:list[Individual]):
        for indi in individuals:
            indi.update_utility()
        return
    
    def update_neighbor_solution(self):
        return
    
    def update_EP(self):
        return
    
    def reproduct(self):
        # Select 1 sub-problem

        # Offspring generation 

        # Mutation

        # Local search

        # Repair solution

        # Update current and neighboring solution

        # Update Ultility

        # Update EP
                
        return
