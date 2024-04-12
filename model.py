import numpy as np

class Individual:
    def __init__(self,lambdas, num_sensors, sensors_positions) -> None:
        self.num_sensors = num_sensors
        self.lambdas = lambdas
        self.mu = 0
        
        # Random solution
        activate = np.random.choice([0,1],num_sensors)
        range = np.random.rand(num_sensors)
        self.solution = [[activate[i],range[i]] for i in range(num_sensors)]
        self.repair_solution()

        self.fitness = self.compute_fitness()


    def compute_fitness(self):
        f1 = 
        f2 = 
        f3 = 
        return np.dot(self.lambdas,[f1,f2,f3])

    def repair_solution(self):
        
        return
    
    
class Population:
    def __init__(self, pop_size, neighborhood_size, num_sensors, sensors_positions) -> None:
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.lambdas = self.generate_lambdas()
        self.pop:list[Individual] = []
        for i in range(self.pop_size):
            self.pop.append(Individual(self.lambdas[i], num_sensors, sensors_positions))

        self.neighbor = {} # Use KNN/... to find neighbors of each sub-problem
        def find_neighbor(self):


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

    def local_search(self):        

    def mutation(self):

    def update(self):

    def update_utility(self):
    

pop = Population(5,3)