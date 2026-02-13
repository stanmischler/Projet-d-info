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
        
        

    def solve_pso(self, random_reset=False, reset_probability=0.01, temperature_annealing=False, temperature=1.0,beta=0.99, local_annealing=False, local_temperature=1.0, local_beta=0.99, dimension_learning=False, update_frequency=100):
        records=[]

        for i in range(self.n_iter):
            self.current_swarm.swarm_update(self.w, self.c_1, self.c_2, random_reset, reset_probability, temperature_annealing, temperature, local_annealing, local_temperature, dimension_learning, update_frequency)

            temperature=temperature*beta
            local_temperature=local_temperature*local_beta
            records.append(self.current_swarm.best_global_fitness)
        return self.current_swarm, records

    #plot the best path found by the PSO algorithm, along with the environment and its obstacles
    def solve_and_plot(self, random_reset=False, reset_probability=0.01, temperature_annealing=False, temperature=1.0,beta=0.99, local_annealing=False, local_temperature=1.0, local_beta=0.99, dimension_learning=False, update_frequency=100):
        swarm, records=self.solve_pso(random_reset, reset_probability, temperature_annealing, temperature,beta, local_annealing, local_temperature, local_beta, dimension_learning, update_frequency)
        plt.plot(records)
        plt.show()
        swarm.swarm_visualize()
        print(self.current_swarm.best_global_fitness)

    #
    def swarm_visualize(self):
        print(self.current_swarm.best_global_fitness)
        self.current_swarm.swarm_visualize()
        


    
