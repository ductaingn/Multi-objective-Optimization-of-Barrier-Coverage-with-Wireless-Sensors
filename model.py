import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

# 1 Individual contains 1 sub-problem and 1 solution
class Individual:
    def __init__(self,lambdas, num_sensors, num_sink_nodes, sensors_positions, sink_node_positions, ideal_point, nadir_point, distances, solution = []) -> None:
        self.num_sensors = num_sensors
        self.num_sink_nodes = num_sink_nodes
        self.lambdas = lambdas
        self.sensors_positions = sensors_positions
        self.sink_nodes_positions = sink_node_positions
        self.mu = 1
        self.distances = distances
        
        # Random solution
        activate = np.random.choice([0,1],num_sensors)
        srange = np.random.rand(num_sensors)
        self.solution = solution
        if solution == []:
            for i in range(num_sensors):
                if(activate[i]==1):
                    self.solution.append([activate[i],srange[i]])
                else:
                    self.solution.append([activate[i],0])
            self.repair_solution()

        self.f = [1e9,1e9,1e9]
        self.fitness = self.compute_fitness(self.solution, ideal_point, nadir_point)
        self.neighbor:list[Individual] = []

        self.distances = distances


    def compute_fitness(self, solution, ideal_point, nadir_point):
        f = [0,0,0] 
        for i in range(self.num_sensors):
            if(solution[i][0]==1):
                f[0] += solution[i][1]**2
                f[1] += 1

                nearest_sink_node_distance = 1e9
                for j in range(self.num_sink_nodes):
                    distance = np.sqrt((self.sensors_positions[i][0]-self.sink_nodes_positions[j][0])**2 + (self.sensors_positions[i][1]-self.sink_nodes_positions[j][1])**2)
                    nearest_sink_node_distance = min(nearest_sink_node_distance, distance)
            
                f[2] += nearest_sink_node_distance
              
        f[2]/=f[1]

        # Normalizing
        for i in range(3):
            f[i] = (f[i]-ideal_point[i])/(nadir_point[i]-ideal_point[i])

        self.f = f
        gte = max([self.lambdas[i]*abs(f[i]-ideal_point[i]) for i in range(3)])
        self.fitness = 1/gte
        return self.fitness
    
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
        barier_length = 1000

        # Get index of active sensors
        active_indx = []
        # Distance between active adjacent sensors 
        distance = self.distances
        for i in range(len(self.sensors_positions)):            
            if(self.solution[i][0]==1):
                active_indx.append(i)

        # Coverage requirement
        self.solution[active_indx[0]][1] = max(
            np.sqrt((self.sensors_positions[active_indx[0]][0]-0)**2 + (self.sensors_positions[active_indx[0]][1]-0)**2),
            distance[active_indx[0], active_indx[1]]/2
        )
        self.solution[active_indx[-1]][1] = max(
            np.sqrt((self.sensors_positions[active_indx[-1]][0]-barier_length)**2 + (self.sensors_positions[active_indx[-1]][1]-0)**2),
            distance[active_indx[-1], active_indx[-2]]/2
        )

        for i in range(1,len(active_indx)-1):
            self.solution[active_indx[i]][1] = max(
                distance[active_indx[i], active_indx[i-1]]/2,
                distance[active_indx[i+1], active_indx[i]]/2
            )

        # Shrink
        length = len(active_indx)
        i = 1
        while(i<length-1):
            if(distance[active_indx[i],active_indx[i-1]] + self.solution[active_indx[i]][1] <= self.solution[active_indx[i-1]][1]
               or
               distance[active_indx[i],active_indx[i+1]] + self.solution[active_indx[i]][1] <= self.solution[active_indx[i+1]][1]):
                self.solution[active_indx[i]] = [0,0]
                active_indx.pop(i)
                length-=1
                if(i>1):
                    i-=1
                continue
            i+=1
        
        active_indx = []
        for i in range(len(self.sensors_positions)):            
            if(self.solution[i][0]==1):
                active_indx.append(i)

        for i in range(1,len(active_indx)-1):
            # If sensor i's range intersect with two of its adjacents
            if(distance[active_indx[i],active_indx[i-1]] < self.solution[active_indx[i]][1]+self.solution[active_indx[i-1]][1]
               and
               distance[active_indx[i],active_indx[i+1]] < self.solution[active_indx[i]][1]+self.solution[active_indx[i+1]][1]):

                # The distance between sensor i and i-1's range: d1 = distance(sensor_i, sensor_i-1) - R(sensor_i-1)
                d1 = distance[active_indx[i],active_indx[i-1]] - self.solution[active_indx[i-1]][1]
                # The distance between sensor i and i+1's range: d2 = distance(sensor_i, sensor_i+1) - R(sensor_i+1)
                d2 = distance[active_indx[i],active_indx[i+1]] - self.solution[active_indx[i+1]][1]

                self.solution[active_indx[i]][1] = max(d1,d2)

        return

    def update_utility(self, new_solution, ideal_point, nadir_point):
        delta_i = self.compute_fitness(new_solution, ideal_point, nadir_point) - self.fitness
        if(delta_i>0.001):
            self.mu = 1
        else:
            self.mu = 0.99 + 0.01*delta_i/0.001
        # print("update utility", self.mu)
        return self.mu

    def add_neighbor(self, individual):
        self.neighbor.append(individual)

class Population:
    def __init__(self, pop_size, neighborhood_size, num_sensors, sensors_positions,num_sink_nodes, sink_nodes_positions) -> None:
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.num_sensors = num_sensors
        self.sensors_positions = sensors_positions
        self.num_sink_nodes = num_sink_nodes
        self.sink_nodes_positions = sink_nodes_positions
        self.lambdas = self.generate_lambdas()
        self.pop:list[Individual] = []
        self.ideal_point = [0,0,0]
        self.nadir_point = [self.num_sensors*(1000**1)/10,self.num_sensors,1000]
        self.EP = []
        self.distances = np.zeros(shape=(self.num_sensors, self.num_sensors))

        for i in range(num_sensors):
            for j in range(num_sensors):
                d =  np.sqrt(
                    (self.sensors_positions[i][0]-self.sensors_positions[j][0])**2 + 
                    (self.sensors_positions[i][1]-self.sensors_positions[j][1])**2)

                self.distances[i,j] = self.distances[j,i] = d

        for i in range(self.pop_size):
            self.pop.append(Individual(self.lambdas[i], num_sensors, self.num_sink_nodes, sensors_positions, sink_nodes_positions, self.ideal_point, self.nadir_point, self.distances))

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
        # Pass by value
        sol = [copy.deepcopy(row) for row in individual.solution]
        new = Individual(individual.lambdas, individual.num_sensors, individual.num_sink_nodes, individual.sensors_positions, individual.sink_nodes_positions, self.ideal_point, self.nadir_point, self.distances,sol)

        return new

    def __repr__(self) -> str:
        # print every individual in population
        res = ""
        for i in range(self.pop_size):
            res += f"Solution to Individual {i}: {self.pop[i].solution}\n"
        return res
    
    # Genrate uniformly spread weighted vectors lambda 
    def generate_lambdas(self):
        weights = []
        # 36 weight vectors
        for i in range(1,9):
            for j in range(i+1,10):
                weights.append([i,j-i,10-j])
        res = []
        for i in range(int(self.pop_size/2)):
            res.append(weights[i])
            res.append(weights[-i-1])
        res = np.array(res)
        indx = np.lexsort((res[:,2],res[:,0],res[:,1]))
        return np.flip(res[indx],0)/10
    
  
    def forward_local_search(self, individual:Individual):
        sorted_gene_index = [i for i, gene in sorted(enumerate(individual.solution), key=lambda gene: gene[1],reverse=True)]
        new_sol = self.new_individual(individual)
        for index in sorted_gene_index:
            if(index!=0 and index!=len(individual.solution)-1 
               and individual.solution[index][0]==1
               and individual.solution[index-1][0]==0 and individual.solution[index+1][0]==0):

                d1 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index-1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index-1][1])**2) 

                d2 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index+1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index+1][1])**2) 

                r1 = individual.solution[0][1] - d1
                r2 = individual.solution[0][1] - d2

                new_sol.solution[index] = [0,0]
                new_sol.solution[index-1][0] = 1
                new_sol.solution[index+1][0] = 1

                new_sol.solution[index-1][1] = r1
                new_sol.solution[index+1][1] = r2

                break
        
        new_sol.repair_solution()
        new_sol.compute_fitness(new_sol.solution, self.ideal_point, self.nadir_point)
        if(new_sol.fitness>individual.fitness):
            individual.solution = [copy.deepcopy(row) for row in new_sol.solution]
            individual.fitness = new_sol.fitness
            individual.f = copy.deepcopy(new_sol.f)
        
        return 

    def backward_local_search(self, individual:Individual):
        active_index = []
        for i in range(len(individual.solution)):
            if(individual.solution[i][0]==1):
                active_index.append(i)
        if(len(active_index)<3):
            return
        d_min = np.inf
        turn_off = [] # Containts index of 2 sensors that will be deactivate
        turn_on = 0 # Containst index of sensor that will be activate
        for i in range(len(active_index)-1):
            # Get 2 sensors have smallest sum range 
            if(individual.solution[active_index[i]][1] + individual.solution[active_index[i+1]][1]<d_min):
                # Check if that 2 sensors have another sleep sensors in between
                if(active_index[i+1] - active_index[i] > 1):
                    d_min = individual.solution[active_index[i+1]][1] + individual.solution[active_index[i]][1]
                    turn_off.clear()
                    turn_off.append(active_index[i])
                    turn_off.append(active_index[i+1])
                    turn_on = int((active_index[i]+active_index[i+1])/2)

        new_sol = self.new_individual(individual)
        d1 = np.sqrt((individual.sensors_positions[turn_on][0]-individual.sensors_positions[turn_off[0]][0])**2 + (individual.sensors_positions[turn_on][1]-individual.sensors_positions[turn_off[0]][1])**2) 

        d2 = np.sqrt((individual.sensors_positions[turn_on][0]-individual.sensors_positions[turn_off[1]][0])**2 + (individual.sensors_positions[turn_on][1]-individual.sensors_positions[turn_off[0]][1])**2) 
        
        r1 = d1 + individual.solution[turn_off[0]][1]
        r2 = d2 + individual.solution[turn_off[1]][1]
        r = max(r1,r2)

        new_sol.solution[turn_off[0]] = [0,0]
        new_sol.solution[turn_off[1]] = [0,0]
        new_sol.solution[turn_on] = [1,r]

        new_sol.repair_solution()
        new_sol.compute_fitness(new_sol.solution,self.ideal_point,self.nadir_point)
        if(new_sol.fitness>individual.fitness):
            individual.solution = [copy.deepcopy(row) for row in new_sol.solution]
            individual.fitness = new_sol.fitness
            individual.f = copy.deepcopy(new_sol.f)

        return 

    def local_search(self, k):
        if(k<self.pop_size-1):
            # idea: sub-problem k+1 assigns smaller weight to f2 
            self.forward_local_search(self.pop[k+1])
        if(k>0):
            self.backward_local_search(self.pop[k-1])


    def selection(self, k=16)->list[Individual,int]:
        # k is number of individuals in selection pool
        indi_index = list(np.random.choice(range(0,self.pop_size),size=k))
        pool = [[self.pop[i], self.pop[i].mu, i] for i in indi_index]
        # choose subproblem based on utility
        return sorted(pool, key=lambda x:pool[1])[-1]

    # copy first half of solution from individual, second half from breed to new_individual
    def crossover1(self, individual:Individual, breed:Individual)->Individual:
        cross_point = int(len(individual.solution)/2)
        new_individual = self.new_individual(individual)
        for i in range(cross_point,len(individual.solution)):
            new_individual.solution[i] = copy.deepcopy(breed.solution[i])

        new_individual.compute_fitness(new_individual.solution, self.ideal_point, self.nadir_point)
        return new_individual
    
    def crossover(self, individual:Individual, breed:Individual)->Individual:
        # random 2 point crossover
        cross_point1 = random.randint(0,len(individual.solution)-1)
        cross_point2 = random.randint(cross_point1,len(individual.solution)-1)

        new_individual = self.new_individual(individual)
        for i in range(cross_point1,cross_point2):
            new_individual.solution[i] = copy.deepcopy(breed.solution[i])

        new_individual.compute_fitness(new_individual.solution, self.ideal_point, self.nadir_point)
        return new_individual

    def update_utility(self, individuals:list[Individual]):
        for indi in individuals:
            indi.update_utility()
        return
    
    def update_neighbor_solution(self, individual:Individual):
        # get neighbor of k
        neighbors = individual.neighbor
        # evaluate solution k in neighbor sub-problems
        for neighbor in neighbors:
            new_sol = self.new_individual(neighbor)
            new_fitness = new_sol.compute_fitness(individual.solution, self.ideal_point, self.nadir_point)
            if(new_fitness>neighbor.fitness):
                neighbor.solution = [copy.deepcopy(row) for row in individual.solution]
                neighbor.f = copy.deepcopy(individual.f)
                neighbor.fitness = new_fitness
                # neighbor.mu = neighbor.update_utility(individual.solution, self.ideal_point, self.nadir_point)
    
    def update_EP(self, individual: Individual):
        new_EP = []
        add_to_EP = True

        if len(self.EP) == 0:
            self.EP.append(individual)
            return

        # loop through EP to find solutions dominated by individual, and find solutions that dominate individual
        # if individual dominates a solution in EP, replace that solution with individual
        # if individual is dominated by a solution in EP, do nothing
        for solution in self.EP:
            dominated_by_individual = False
            dominate_individual = False
            for j in range(3):
                if solution.f[j] > individual.f[j]:
                    dominated_by_individual = True
                    break
                elif solution.f[j] < individual.f[j]:
                    dominate_individual = True
            if not dominated_by_individual:
                new_EP.append(solution)
                if dominate_individual:
                    add_to_EP = False

        if add_to_EP:
            new_EP.append(individual)

        self.EP = new_EP
        
        # Update ideal_point and z_nadir
        for i in range(3):
            self.ideal_point[i] = min(self.ideal_point[i], individual.f[i])
            self.nadir_point[i] = max(self.nadir_point[i], individual.f[i])

    
    def reproduct(self):
        # Select 1 sub-problem
        sub_problem, _ , sub_problem_index = self.selection()

        # Offspring generation 
        choosen_neighbor = np.random.choice(sub_problem.neighbor)
        child = self.crossover(sub_problem, choosen_neighbor)
        # Mutation
        child.mutation()
        # Repair solution
        child.repair_solution()
        child.compute_fitness(child.solution, self.ideal_point, self.nadir_point)

        if(child.fitness > sub_problem.fitness):
            sub_problem.update_utility(child.solution, self.ideal_point, self.nadir_point)
            sub_problem.solution = [copy.deepcopy(row) for row in child.solution]
            sub_problem.f = copy.deepcopy(child.f)
            sub_problem.fitness = child.fitness
            # sub_problem.compute_fitness(sub_problem.solution, self.ideal_point, self.nadir_point)

        self.local_search(sub_problem_index)
        self.update_neighbor_solution(sub_problem)

        # Update EP
        # self.update_EP(sub_problem)
        
        return
