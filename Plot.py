import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import model

def Plot_solution(sensors_positions:list[list[float,float]], solution:list[list[int,float]]):
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

def Plot_objectives(generations):
    # Choose a colormap (replace 'viridis' with your preferred choice)
    cmap = plt.cm.winter
    # Create an array of values from 0 to 100 (one for each point)
    values = np.linspace(0, 1, 10)
    # Generate the list of colors using the colormap
    color = cmap(values)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    for i in range(len(generations)):
        if((i+1)%1000==0):
            f = []
            for individual in generations[i]:
                f.append([individual[0], individual[1], individual[2]])
            f = np.array(f)
            ax.scatter(f[:,0],f[:,1], f[:,2], c = color[int((i+1)/1000)-1],label=f'Gen #{i+1}')
    ax.view_init(azim=45,elev = 30)
    ax.set_xlabel('f_1')
    ax.set_ylabel('f_2')
    ax.set_zlabel('f_3')
    plt.legend()
    plt.title('Objective funtions value over generations')
    plt.show()
