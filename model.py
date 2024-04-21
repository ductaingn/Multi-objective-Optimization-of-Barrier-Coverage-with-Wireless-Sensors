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
        # first element of mem_FLS is max range 
        self.mem_FLS = []
        self.mem_BLS = []
        
        # Random solution
        activate = np.random.choice([0,1],num_sensors)
        srange = np.random.rand(num_sensors)
        self.solution = [[activate[i], srange[i]] for i in range(num_sensors)]
        self.repair_solution()

        self.f = [1e9,1e9,1e9]
        self.fitness = self.compute_fitness(self.solution, ideal_point)
        self.neighbor = []
        self.preprocess_for_LS()


    def compute_fitness(self, solution, ideal_point):
        f = [0,0,0] 
        for i in range(self.num_sensors):
            if(solution[i][0]==1):
                f[0] += solution[i][1]**2
                f[1] += 1

                nearest_sink_node_distance = 1e9
                for j in range(self.num_sink_nodes):
                    distance = np.abs((self.sensors_positions[i][0]-self.sink_nodes_positions[j][0])**2 + (self.sensors_positions[i][1]-self.sink_nodes_positions[j][1])**2)
                    nearest_sink_node_distance = min(nearest_sink_node_distance, distance)
            
                f[2] += nearest_sink_node_distance
              
        f[2]/=f[1]
        self.f = f
        gte = max([self.lambdas[i]*abs(f[i]-ideal_point[i]) for i in range(3)])
        return gte
    
    def mutation(self):
        active_sensor_index = []
        sleep_sensor_index = []
        for i in range(len(self.solution)):
            if(self.solution[i][0]==1):
                active_sensor_index.append(i)
            else:
                sleep_sensor_index.append(i)

        # Choose a sleep sensor and a active sensor, then exchange their state
        change_index = np.random.choice(active_sensor_index), np.random.choice(sleep_sensor_index)

        temp = self.solution[change_index[0]]
        self.solution[change_index[0]] = self.solution[change_index[1]]
        self.solution[change_index[1]] = temp
        
        return
                   
    def repair_solution(self):
        
        return
    
    def preprocess_for_LS(self):
        # print(self.solution)
        tmp_min_range = 1e9
        # search in a solution for a gene with 2 genes before and after it having range = 0
        for i in range(1,self.num_sensors-1):
            fw = self.solution[i+1]
            bw = self.solution[i-1]
            if(self.solution[i][0]==1 and bw[0]==0 and fw[0]==0):
                self.mem_FLS.append(i)
            if(self.solution[i][0]==0 and bw[0]==1 and fw[0]==1):
                if (bw[0] + fw[0]<tmp_min_range):
                    tmp_min_range = bw[0] + fw[0]
                    self.mem_BLS.insert(0, i)
                else: self.mem_BLS.append(i)
        # sort mem_FLS and mem_BLS in descending order of range
        self.mem_FLS.sort(key=lambda x: self.solution[x][1], reverse=True)
        # print(self.mem_FLS)
        # print(self.mem_BLS)
    
    def update_utility(self, new_solution):
        delta_i = self.compute_fitness(new_solution) - self.fitness
        if(delta_i>0.001):
            return 1
        else:
            return 0.99 + 0.01*delta_i/0.001

    def add_neighbor(self, individual):
        self.neighbor.append(individual)

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
            for i in range(len(self.lambdas)):
                for j in indices[i]:
                    self.pop[i].add_neighbor(self.pop[j])
        
        find_neighbor()

    def new_individual(self, individual:Individual)->Individual:
        new = Individual(individual.lambdas, individual.num_sensors, individual.num_sink_nodes, individual.sensors_positions, individual.sink_nodes_positions, self.ideal_point)

        return new

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
        # choose first element in mem_FLS
        ind = self.pop[k].mem_FLS[0]
        new_sol = self.pop[k].solution.copy()

        sensor_bw_2 = self.pop[k].sensors_positions[ind-2]
        sensor_bw_1 = self.pop[k].sensors_positions[ind-1]
        sensor_fw_1 = self.pop[k].sensors_positions[ind+1]
        sensor_fw_2 = self.pop[k].sensors_positions[ind+2]

        # calculate distance from center of 2 sensors
        dist = np.sqrt(np.abs((sensor_bw_1[0]-sensor_fw_1[0])**2 + (sensor_bw_1[1]-sensor_fw_1[1])**2))

        if (ind >= 2):
            range_bw = np.sqrt(np.abs((sensor_bw_2[0]-sensor_bw_1[0])**2 + (sensor_bw_2[1]-sensor_bw_1[1])**2))
            range_bw -= self.pop[k].solution[ind-2][1]
        else:
            range_bw = 0
        if (ind <= self.num_sensors-3):
            range_fw -= np.sqrt(np.abs((sensor_fw_1[0]-sensor_fw_2[0])**2 + (sensor_fw_1[1]-sensor_fw_2[1])**2))
            range_fw -= self.pop[k].solution[ind+2][1]
        else:
            range_fw = 0
        sum_range = range_bw + range_fw

        if (sum_range <= dist):
            new_sol[ind-1][1] = dist/2
            new_sol[ind+1][1] = dist/2
        else:
            new_sol[ind-1][1] = range_bw
            new_sol[ind+1][1] = range_fw

        # replace
        new_sol[ind-1][0] = 1
        new_sol[ind+1][0] = 1
        new_sol[ind][0] = 0

        # compute fitness of new solution
        new_fitness = self.pop[k].compute_fitness(new_sol, self.ideal_point)
        # TODO repair solution when?
        if (new_fitness < self.pop[k].fitness):
            self.pop[k].solution = new_sol
            self.pop[k].fitness = new_fitness
            self.pop[k].mu = self.pop[k].update_utility(new_sol)
            self.pop[k].preprocess_for_LS()
        
    def backward_local_search(self, k):
        # choose first element in mem_BLS
        ind = self.pop[k].mem_BLS[0]
        new_sol = self.pop[k].solution.copy()

        sensor = self.pop[k].sensors_positions[ind]
        sensor_bw_2 = self.pop[k].sensors_positions[ind-2]
        sensor_fw_2 = self.pop[k].sensors_positions[ind+2]

        # calculate distance to center of bw2 and fw2 sensors
        dist_bw = np.sqrt(np.abs((sensor_bw_2[0]-sensor[0])**2 + (sensor_bw_2[1]-sensor[1])**2))
        dist_bw -= self.pop[k].solution[ind-2][1]
        dist_fw = np.sqrt(np.abs((sensor_fw_2[0]-sensor[0])**2 + (sensor_fw_2[1]-sensor[1])**2))
        dist_fw -= self.pop[k].solution[ind+2][1]

        if (dist_bw < dist_fw):
            new_sol[ind][1] = dist_fw
        else:
            new_sol[ind][1] = dist_bw
            
        # replace
        new_sol[ind][0] = 1
        new_sol[ind-1][0] = 0
        new_sol[ind+1][0] = 0
        
        # compute fitness of new solution
        new_fitness = self.pop[k].compute_fitness(new_sol, self.ideal_point)
        # TODO repair solution when?
        if (new_fitness < self.pop[k].fitness):
            self.pop[k].solution = new_sol
            self.pop[k].fitness = new_fitness
            self.pop[k].mu = self.pop[k].update_utility(new_sol)
            self.pop[k].preprocess_for_LS()

    def local_search(self, k):
        # TODO fix logic to 2016 paper
        self.forward_local_search(k+1)
        self.backward_local_search(k-1)

    def selection(self, k=16)->list[Individual,int]:
        # k is number of individuals in selection pool
        indi_index = list(np.random.choice(range[0,self.pop_size],size=k))
        pool = [[self.pop[i],i] for i in indi_index]
        return sorted(pool, key=lambda x:pool[1])[-1]
    
    def crossover(self, individual:Individual, breed:Individual)->Individual:
        cross_point = len(individual.solution)/2
        new_individual = self.new_individual(individual)
        for i in range(cross_point,len(individual.solution)):
            new_individual.solution[i] = breed.solution[i]

        individual.compute_fitness(individual.solution, self.ideal_point)
        return new_individual

    def update_utility(self, individuals:list[Individual]):
        for indi in individuals:
            indi.update_utility()
        return
    
    def update_neighbor_solution(self, k):
        # get neighbor of k
        neighbors = self.neighbor[k]
        # evaluate solution k in neighbor sub-problems
        for i in neighbors:
            new_fitness = self.pop[i].compute_fitness(self.pop[k].solution, self.ideal_point)
            if(new_fitness<self.pop[i].fitness):
                self.pop[i].solution = self.pop[k].solution
                self.pop[i].fitness = new_fitness
                self.pop[i].mu = self.pop[i].update_utility(self.pop[k].solution)
                # if 1 solution updated, preprocess for LS again
                self.pop[i].preprocess_for_LS()
    
    def update_EP(self):
        # TODO update ideal point here 
        return
    
    def reproduct(self):
        # Select 1 sub-problem
        sub_problem,sub_problem_index = self.selection()

        # Offspring generation 
        choosen_neighbor = np.random.choice(sub_problem.neighbor)
        child = self.crossover(sub_problem, choosen_neighbor)
        if(child.fitness<sub_problem.fitness):
            sub_problem.update_utility(child.solution)
            sub_problem.solution = child.solution
            sub_problem.compute_fitness(sub_problem.solution,self.ideal_point)

        # Mutation
        sub_problem.mutation()

        # Repair solution
        sub_problem.repair_solution()

        # Update neighboring solution

        # Update EP
        
        return
