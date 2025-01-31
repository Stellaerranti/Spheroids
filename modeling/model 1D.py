import numpy as np
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, x, radius=1, filtration_rate=0.01):
        self.x = x
        self.radius = radius
        self.filtration_rate = filtration_rate  # Defines how much substance is filtered
    
    def draw(self, ax):
        circle = plt.Circle((self.x, 0), self.radius, color='blue', alpha=0.6, edgecolor='black')
        ax.add_patch(circle)
    
class Spheroid:
    def __init__(self, num_cells, cell_radius, diffusion_coefficient, dx, dt):
        self.num_cells = num_cells
        self.cell_radius = cell_radius
        self.diffusion_coefficient = diffusion_coefficient
        self.dx = dx  # Spatial step
        self.dt = dt  # Time step
        self.length = num_cells * cell_radius * 2
        self.grid_size = int(self.length / dx) + 1
        self.concentration = np.ones(self.grid_size)  # Initially filled with substance
        self.generate_cells()
        
    def generate_cells(self):
        self.cells = []
        spacing = 2 * self.cell_radius
        start_pos = -((self.num_cells - 1) * spacing) / 2
        for i in range(self.num_cells):
            self.cells.append(Cell(start_pos + i * spacing, self.cell_radius))
    
    def diffuse_and_filter(self, iterations):
        alpha = self.diffusion_coefficient * self.dt / (self.dx ** 2)
        for _ in range(iterations):
            new_concentration = self.concentration.copy()
            for i in range(1, self.grid_size - 1):
                new_concentration[i] = (
                    self.concentration[i] + alpha * (self.concentration[i + 1] - 2 * self.concentration[i] + self.concentration[i - 1])
                )
            # Apply filtering effect across cell radius
            for cell in self.cells:
                center_idx = int((cell.x + self.length / 2) / self.dx)
                radius_range = int(cell.radius / self.dx)
                for j in range(-radius_range, radius_range + 1):
                    idx = center_idx + j
                    if 0 <= idx < self.grid_size:
                        distance_factor = 1 - (j / radius_range) ** 2  # Parabolic filtration function
                        new_concentration[idx] *= (1 - cell.filtration_rate * distance_factor)
            self.concentration = new_concentration
    
    def plot_concentration(self):
        x_positions = np.linspace(-self.length / 2, self.length / 2, self.grid_size)
        plt.figure(figsize=(8, 4))
        plt.plot(x_positions, self.concentration, label='Concentration Profile')
        plt.ylim(0,1.1)
        plt.xlabel('Position')
        plt.ylabel('Concentration')
        plt.title('Diffusion with Parabolic Filtration')
        plt.legend()
        plt.show()
    
# Example Usage:
spheroid = Spheroid(num_cells=5, cell_radius=1, diffusion_coefficient=0.1, dx=0.1, dt=0.01)
spheroid.diffuse_and_filter(iterations=300)
spheroid.plot_concentration()
