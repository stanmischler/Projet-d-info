from environment import environment
from path import path
import matplotlib.pyplot as plt
from swarm import swarm
from PSO import PSO

size = 100
obstacles = [(10,0,50,50), (60,70,30,30)]
env = environment(size, obstacles)

pso=PSO(env, 20, 0.1, 0.1, 0.5, 10000, 3)

pso.swarm_visualize()
pso.solve_and_plot(random_reset=True, reset_probability=0.1, temperature_annealing=False, temperature=100, beta=0.99, local_annealing=False, local_temperature=10, local_beta=0.99, dimension_learning=True, update_frequency=50)


