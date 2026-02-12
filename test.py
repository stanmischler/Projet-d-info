from environment import environment
from path import path
import matplotlib.pyplot as plt
from swarm import swarm
from PSO import PSO

size = 100
obstacles = [(40, 40, 10, 10), (80, 80, 10, 10), (40,10,10,10)]
env = environment(size, obstacles)

pso=PSO(env, 20, 0.1, 0.1, 0.3, 10000, 5)

pso.swarm_visualize()
pso.solve_pso()
pso.swarm_visualize()

