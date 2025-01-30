import numpy as np
import matplotlib.pyplot as plt

class NutrientField:
    def __init__(self, length=10, resolution=100, diffusion_rate=0.1):
        """Represents the 1D nutrient field with diffusion."""
        self.length = length
        self.resolution = resolution
        self.diffusion_rate = diffusion_rate
        self.nutrient_grid = np.ones(resolution)  # Initial uniform distribution
        self.grid_x = np.linspace(-length / 2, length / 2, resolution)  # Centering around 0
    
    def diffuse(self):
        """Simulates diffusion using Fick's Law (finite difference method)."""
        new_grid = self.nutrient_grid.copy()
        for i in range(1, self.resolution - 1):
            flux_in = self.diffusion_rate * (self.nutrient_grid[i - 1] - self.nutrient_grid[i])
            flux_out = self.diffusion_rate * (self.nutrient_grid[i] - self.nutrient_grid[i + 1])
            new_grid[i] += flux_in - flux_out
        self.nutrient_grid = new_grid
    
    def consume_nutrients(self, cells, consumption_rate=0.1):
        """Cells consume nutrients sequentially from both positive and negative edges toward the center."""
        sorted_cells = sorted(cells, key=lambda c: abs(c.x), reverse=True)  # Start from outermost cells
        for cell in sorted_cells:
            left_index = np.searchsorted(self.grid_x, cell.x - cell.radius)
            right_index = np.searchsorted(self.grid_x, cell.x + cell.radius)
            if 0 <= left_index < self.resolution and 0 <= right_index < self.resolution:
                self.nutrient_grid[left_index:right_index] *= (1 - consumption_rate)  # Decrease within the cell region
                self.nutrient_grid[:left_index] *= (1 - consumption_rate)  # Decrease for all leftward regions
                self.nutrient_grid[right_index:] *= (1 - consumption_rate)  # Decrease for all rightward regions
    
    def draw(self):
        """Plots the nutrient concentration along the 1D space."""
        plt.plot(self.grid_x, self.nutrient_grid, label="Nutrient Concentration", color='green')
        plt.xlabel("Position")
        plt.ylabel("Nutrient Level")
        plt.title("Nutrient Distribution Over Time")
        plt.ylim(0,1.1)
        plt.legend()
        plt.show()

class Cell:
    def __init__(self, x, radius=1):
        self.x = x
        self.radius = radius
    
    def draw(self, ax):
        """Draws the cell as a circle on the given axis."""
        circle = plt.Circle((self.x, 0), self.radius, color='blue', alpha=0.6, edgecolor='black')
        ax.add_patch(circle)
    
class Spheroid:
    def __init__(self, num_cells, cell_radius, length=10, resolution=100):
        """Creates a 1D spheroid and initializes the nutrient environment."""
        self.num_cells = num_cells
        self.cell_radius = cell_radius
        self.cells = []
        self.nutrient_field = NutrientField(length=length, resolution=resolution)
        self.generate_cells()
    
    def generate_cells(self):
        spacing = 2 * self.cell_radius
        start_pos = -((self.num_cells - 1) * spacing) / 2  # Center cells around 0
        for i in range(self.num_cells):
            self.cells.append(Cell(start_pos + i * spacing, self.cell_radius))
    
    def simulate_nutrient_dynamics(self, iterations=10):
        """Runs the simulation for diffusion and sequential nutrient consumption."""
        for _ in range(iterations):
            self.nutrient_field.consume_nutrients(self.cells)
            self.nutrient_field.diffuse()
    
    def draw(self):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_xlim(-self.nutrient_field.length / 2, self.nutrient_field.length / 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title("1D Spheroid Model")
        
        for cell in self.cells:
            cell.draw(ax)
        
        plt.show()

# Example Usage:
spheroid = Spheroid(num_cells=5, cell_radius=1, length=10, resolution=100)
spheroid.draw()
spheroid.simulate_nutrient_dynamics(iterations=20)
spheroid.nutrient_field.draw()