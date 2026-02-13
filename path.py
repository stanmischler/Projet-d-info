from environment import environment
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from random import uniform
from scipy.spatial.distance import euclidean
import numpy as np
import random


class path:

    def __init__(self, d, environment, initialize=True):
            self.d = d
            self.environment = environment
            self.time_last_update=0
            
            # 2. Only generate random points if initialize is True
            if initialize:
                self.points = [np.array((uniform(0, self.environment.size), uniform(0, self.environment.size))) for _ in range(self.d)]
                V_m = 0.1 * self.environment.size
                self.velocities = [np.array((uniform(-V_m, V_m), uniform(-V_m, V_m))) for _ in range(self.d)]

                self.best_state = [p.copy() for p in self.points]
                self.best_fitness = self.fitness()
            else:
                # Create empty attributes to be filled later
                self.points = []
                self.velocities = []
                self.best_state = []
                self.best_fitness = -float('inf')

    def path_visualize(self):
        """Plot the environment with obstacles and the path polyline (0,0) -> points -> (w,h)."""
        fig, ax = plt.subplots()
        size = self.environment.size
        if hasattr(size, "__len__"):
            w, h = size[0], size[1]
        else:
            w = h = size
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        # Draw obstacles
        for (x, y, lx, ly) in self.environment.obstacles:
            rect = Rectangle((x, y), lx, ly, facecolor="gray", edgecolor="black")
            ax.add_patch(rect)
        # Build path polyline: start -> checkpoints -> end
        path_x = [0.0] + [p[0] for p in self.points] + [float(w)]
        path_y = [0.0] + [p[1] for p in self.points] + [float(h)]
        ax.plot(path_x, path_y, "b.-", linewidth=2, markersize=6, label="path")
        ax.plot(0, 0, "go", markersize=8, label="start")
        ax.plot(w, h, "rs", markersize=8, label="end")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Path in environment")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax

    def is_valid(self):
        #segment_1=[m1,m2] and segment_2=[m3,m4]
        def intersection(m1, m2, m3, m4):
            x1, y1 = m1
            x2, y2 = m2
            x3, y3 = m3
            x4, y4 = m4

            #direction vectors of the lines
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3

            denom = dx1 * dy2 - dy1 * dx2
            #deal with round errors by adding a small tolerance
            if abs(denom) < 1e-10:
                return None  # parallel (or coincident)

            dx3, dy3 = x3 - x1, y3 - y1
            t = (dx3 * dy2 - dy3 * dx2) / denom

            x = x1 + t * dx1
            y = y1 + t * dy1
            return (x, y)

        def intersect(segment_1, segment_2):
            p = intersection(segment_1[0], segment_1[1], segment_2[0], segment_2[1])
            if p is None:
                return False  # parallel (or coincident) lines

            x, y = p
            x1, y1 = segment_1[0]
            x2, y2 = segment_1[1]
            x3, y3 = segment_2[0]
            x4, y4 = segment_2[1]

            on_seg1 = (min(x1, x2)-1e-2 <= x <= max(x1, x2)+1e-2 and
                       min(y1, y2)-1e-2 <= y <= max(y1, y2)+1e-2)
            on_seg2 = (min(x3, x4)-1e-2 <= x <= max(x3, x4)+1e-2 and
                       min(y3, y4)-1e-2 <= y <= max(y3, y4)+1e-2)

            return on_seg1 and on_seg2

        def intersect_obstacle(segment, obstacle):
            
            xo=obstacle[0]
            yo=obstacle[1]
            lx=obstacle[2]
            ly=obstacle[3]

            edges=[((xo,yo), (xo+lx,yo)), ((xo+lx,yo), (xo+lx,yo+ly)), ((xo+lx,yo+ly), (xo,yo+ly)), ((xo,yo+ly), (xo,yo))]

            for edge in edges:
                if intersect(segment, edge):
                    return True
            return False

        segments=[((0,0),(self.points[0]))]
        for i in range(len(self.points)-1):
            segments.append((self.points[i], self.points[i+1]))
        segments.append((self.points[-1], (self.environment.size,self.environment.size)))

        for segment in segments:
            for obstacle in self.environment.obstacles:
                if intersect_obstacle(segment, obstacle):
                    return False
        return True

    #calculate the size of the path (fitness) / return a very bad fitness if the path is invalid
    def fitness(self):
        if not self.is_valid():
            return -1e8
        else:
            s=sum(euclidean(self.points[i], self.points[i+1]) for i in range(self.d-1))
            s+=euclidean(self.points[-1], (self.environment.size,self.environment.size))
            s+=euclidean((0,0), self.points[0])
            return -s


    #update the best_fitness and best_state attributes, and return the current fitness
    def update_fitness(self):
        fitness = self.fitness()
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_state = [p.copy() for p in self.points] 
            self.time_last_update = 0
        else:
            self.time_last_update += 1
        return fitness

    #Return a deep copy of the path object
    def copy(self):
        new_path = path(self.d, self.environment, initialize=False)
        
        new_path.time_last_update = self.time_last_update
        new_path.points = [p.copy() for p in self.points]
        new_path.velocities = [v.copy() for v in self.velocities]
        new_path.best_state = [p.copy() for p in self.best_state]
        new_path.best_fitness = self.best_fitness
        
        return new_path