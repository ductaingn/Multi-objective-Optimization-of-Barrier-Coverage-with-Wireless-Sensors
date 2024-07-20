import pygmo as pg
import numpy as np
import pickle 
import Plot
import copy
import matplotlib.pyplot as plt
with open('MOEAD_Results/uniform/50x1000unit/100sensors/dataset_0/last_solutions_0.pickle','rb') as file:
    moead_sol = pickle.load(file)
with open('MOEAD_Results/uniform/50x1000unit/300sensors/dataset_0/objectives_by_generations_0.pickle','rb') as file:
    moead_sol_obj = pickle.load(file)
with open('NSGA2_Results/uniform/50x1000unit/100sensors/dataset_0/last_solutions_0.pickle','rb') as file:
    nsga2_sol = pickle.load(file)
with open('NSGA2_Results/uniform/50x1000unit/300sensors/dataset_0/objectives_by_generations_0.pickle','rb') as file:
    nsga2_sol_obj = pickle.load(file)
Plot.Scatter_objectives_Plotly(nsga2_sol_obj)
# Plot.Compare_objectives_Plotly(moead_sol_obj,nsga2_sol_obj)