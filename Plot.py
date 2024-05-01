import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import model
import pickle

def Plot_solution(sensors_positions:list[list[float,float]], solution:list[list[int,float]]):
    '''
    # Draw a solution 
    Draw solution's all active sensors with its position and sensing range
    '''
    fix, ax = plt.subplots()
    active_x = []
    active_y = []
    for i in range(len(sensors_positions)):
        if(solution[i][0]==1):
            circle = patches.Circle((sensors_positions[i]),solution[i][1],edgecolor='blue',fill=False)
            ax.add_patch(circle)
            active_x.append(sensors_positions[i][0])
            active_y.append(sensors_positions[i][1])
    
    ax.scatter(active_x,active_y,marker='x')
    ax.set_xlim(-10,1010)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal',adjustable='box')
    plt.show()


def Scatter_objectives(objectives_by_generations, num_generations):
    '''
    # Scatter objectives by generations
    '''
    # Number of generation to plot
    num_gen_plot = 10
    step = num_generations/num_gen_plot
    # Choose a colormap (replace 'viridis' with your preferred choice)
    cmap = plt.cm.winter
    # Create an array of values from 0 to 100 (one for each point)
    values = np.linspace(0, 1, num_gen_plot)
    # Generate the list of colors using the colormap
    color = cmap(values)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    for i in range(len(objectives_by_generations)):
        if(i==0):
            f = []
            for individual in objectives_by_generations[i]:
                f.append([individual[0], individual[1], individual[2]])
            f = np.array(f)
            ax.scatter(f[:,0],f[:,1], f[:,2], c = color[int(i/step)],label=f'Gen #{i+1}')
        elif((i+1)%step==0):
            f = []
            for individual in objectives_by_generations[i]:
                f.append([individual[0], individual[1], individual[2]])
            f = np.array(f)
            ax.scatter(f[:,0],f[:,1], f[:,2], c = color[int((i+1)/step)-1],label=f'Gen #{i+1}')
    ax.view_init(azim=45,elev = 30)
    ax.set_xlabel('f_1')
    ax.set_ylabel('f_2')
    ax.set_zlabel('f_3')
    plt.legend()
    plt.title('Objective funtions value over generations')
    plt.show()


def Plot_fitness(num_sensors, num_results, name_pattern='best_indi_fitness', length=1000, width=50, distribution='uniform'):
    '''
    # Plot fitness of one dataset with mean and standard deviation
    Prerequisites: Results available
    
    Arguments:
        num_sensors: Number of sensors
        num_results: Number of fitness files
        name_pattern: Pattern of result file name, eg: 'best_indi_fitness' 
    '''
    fitneses = []
    for i in range(num_results):
        try:
            with open(f'Results/{distribution}/{width}x{length}unit/{num_sensors}sensors/{name_pattern}_{i}.pickle','rb') as file:
                fitneses.append(pickle.load(file))
        except FileNotFoundError:
            print('FileNotFoundError')
            return
    
    fitneses = np.array(fitneses)
    mean = np.mean(fitneses,axis=0)
    std = np.std(fitneses,axis=0)
    
    fix, ax = plt.subplots()
    x = np.arange(len(fitneses[0]))
    ax.plot(x,mean)
    ax.fill_between(x, mean-std, mean+std, alpha =0.2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_ylim(1,7)
    plt.title(f'{name_pattern} Mean and Standard deviation')
    plt.show()