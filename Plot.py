import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

