from environment import environment
from path import path
import matplotlib.pyplot as plt
from swarm import swarm

class PSO:
    def __init__(self, environment, S, w, c_1, c_2, n_iter,d):
        self.environment=environment
        self.S=S
        self.w=w
        self.c_1=c_1
        self.c_2=c_2
        self.n_iter=n_iter
        self.d=d
        self.current_swarm=swarm(self.S, self.d, self.environment)
        

    def solve_pso(self):
        for i in range(self.n_iter):
            self.current_swarm.swarm_update(self.w, self.c_1, self.c_2)
        return self.current_swarm

    def swarm_visualize(self):
        print(self.current_swarm.best_global_fitness)
        self.current_swarm.swarm_visualize()


    
