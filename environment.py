from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class environment:
    def __init__(self, size, obstacles):
        self.size = size # Scalar assumed
        self.obstacles = obstacles

    def plot_environment(self):
        fig, ax = plt.subplots()
        w = self.size
        ax.set_xlim(0, w)
        ax.set_ylim(0, w) # Use w for both dimensions
        ax.set_aspect("equal")
        
        for (x, y, lx, ly) in self.obstacles:
            rect = Rectangle((x, y), lx, ly, facecolor="gray", edgecolor="black")
            ax.add_patch(rect)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Environment")
        plt.tight_layout()
        plt.show()
        return fig, ax