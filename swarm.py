from calendar import c
from path import path
from environment import environment
from matplotlib import pyplot as plt
import random
import numpy as np
from random import uniform


class swarm:
    def __init__(self, S, d, environment):
        self.S = S
        self.paths = [path(d, environment) for _ in range(S)]
        
        fitnesses=[p.fitness() for p in self.paths]
        best_index=fitnesses.index(max(fitnesses))
        self.best_global_points=[p.copy() for p in self.paths[best_index].points]
        self.best_global_fitness=fitnesses[best_index]

    def swarm_visualize(self):
        # Prepare figure and axes
        fig, ax = plt.subplots()
        size = self.paths[0].environment.size
        obstacles = self.paths[0].environment.obstacles

        # Environment size is a float (square)
        w = h = float(size)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")

        # Draw obstacles as rectangles
        from matplotlib.patches import Rectangle
        for (x, y, lx, ly) in obstacles:
            rect = Rectangle((x, y), lx, ly, facecolor="gray", edgecolor="black")
            ax.add_patch(rect)

        # Draw start and end markers
        ax.plot(0, 0, "go", markersize=8, label="start")
        ax.plot(w, h, "rs", markersize=8, label="end")

        # Plot all paths in light blue
        for p in self.paths:
            path_x = [0.0] + [pt[0] for pt in p.points] + [w]
            path_y = [0.0] + [pt[1] for pt in p.points] + [h]
            ax.plot(path_x, path_y, color="#7ec0ee", linestyle="--", linewidth=1, marker=".", markersize=4, alpha=0.7)

        # Highlight the best path in red
        best_path = self.best_global_points
        path_x = [0.0] + [pt[0] for pt in best_path] + [w]
        path_y = [0.0] + [pt[1] for pt in best_path] + [h]
        ax.plot(path_x, path_y, color="red", linewidth=2, marker="o", markersize=6, label="best path")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Swarm Paths in Environment")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax

    

    def generate_next(self, current_path, w, c_1, c_2, random_reset=False, reset_probability=0.01, temperature_annealing=False, temperature=1.0, local_annealing=False, local_temperature=1.0, dimension_learning=False,update_frequency=100):
        if dimension_learning and current_path.time_last_update>=update_frequency:
            best_global=[p.copy() for p in self.best_global_points]
            candidate=[p.copy() for p in current_path.best_state]
            fitness=self.best_global_fitness

            for i in range(current_path.d):
                new_candidate=[p.copy() for p in candidate]
                new_candidate[i]=best_global[i].copy()
                cand_path=path(current_path.d, current_path.environment, initialize=False)
                cand_path.points=new_candidate
                if cand_path.fitness()>fitness:
                    fitness=cand_path.fitness()
                    candidate=new_candidate
                    current_path.time_last_update=0
            
            current_path.best_state=candidate
            current_path.best_fitness=fitness


            

        if random_reset and random.uniform(0, 1) < reset_probability:
            new_points=[np.array((uniform(0, current_path.environment.size), uniform(0, current_path.environment.size))) for _ in range(current_path.d)]
            new_velocities=[np.array((uniform(-0.1*current_path.environment.size, 0.1*current_path.environment.size), uniform(-0.1*current_path.environment.size, 0.1*current_path.environment.size))) for _ in range(current_path.d)]

            new_path=path(current_path.d, current_path.environment, initialize=False)

            current_best_fitness=current_path.best_fitness
            current_best_state=[p.copy() for p in current_path.best_state]

            new_path.points=new_points
            new_path.velocities=new_velocities

            new_path.best_fitness=current_best_fitness
            new_path.best_state=current_best_state
            new_path.update_fitness()


            return new_path





        # Rename 'path' argument to 'current_path' to avoid shadowing class name
        velocities = [
            w * current_path.velocities[i] + 
            c_1 * random.uniform(0, 1) * (current_path.best_state[i] - current_path.points[i]) + 
            c_2 * random.uniform(0, 1) * (self.best_global_points[i] - current_path.points[i]) 
            for i in range(current_path.d)
        ]
        
        # Calculate new points based on velocity
        new_points = [current_path.points[i] + velocities[i] for i in range(current_path.d)]
        
        # Boundary checks with Velocity Reset (Fixing "Sticky Walls")
        size = current_path.environment.size
        for i in range(current_path.d):
            # Check X bounds
            if new_points[i][0] < 0:
                new_points[i][0] = 0
                velocities[i][0] = 0  # STOP moving left
            elif new_points[i][0] > size:
                new_points[i][0] = size
                velocities[i][0] = 0  # STOP moving right

            # Check Y bounds
            if new_points[i][1] < 0:
                new_points[i][1] = 0
                velocities[i][1] = 0
            elif new_points[i][1] > size:
                new_points[i][1] = size
                velocities[i][1] = 0

        current_best_fitness=current_path.best_fitness
        current_best_state=[p.copy() for p in current_path.best_state]

        new_path=path(current_path.d, current_path.environment, initialize=False)
        new_path.time_last_update=current_path.time_last_update
        new_path.points=new_points
        new_path.velocities=velocities
        new_path.best_fitness=current_best_fitness
        new_path.best_state=current_best_state
        new_path.update_fitness()

        if local_annealing and new_path.fitness()-current_best_fitness<0:
            p=min(1,np.exp((new_path.fitness()-current_best_fitness)/local_temperature))
            if random.uniform(0, 1) < p:
                new_path.best_fitness=new_path.fitness()
                new_path.best_state=[p.copy() for p in new_path.points]

        if temperature_annealing:
                p=np.exp((new_path.fitness()-self.best_global_fitness)/temperature)
                if random.uniform(0, 1) < p:
                    self.best_global_fitness=new_path.fitness()
                    self.best_global_points=[p.copy() for p in new_path.points]
        


        return new_path




    def swarm_update(self,w,c_1,c_2, random_reset=False, reset_probability=0.01,temperature_annealing=False, temperature=1.0, local_annealing=False, local_temperature=1.0, dimension_learning=False, update_frequency=100):
        new_paths=[self.generate_next(p,w,c_1,c_2, random_reset, reset_probability, temperature_annealing, temperature, local_annealing, local_temperature, dimension_learning, update_frequency) for p in self.paths]
        fitnesses=[p.fitness() for p in new_paths]
        best_index=fitnesses.index(max(fitnesses))

        if fitnesses[best_index]>self.best_global_fitness:
            self.best_global_points=   [p.copy() for p in new_paths[best_index].points]
            self.best_global_fitness=fitnesses[best_index]
        self.paths=new_paths