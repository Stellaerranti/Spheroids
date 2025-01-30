import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class NutrientField:
    def __init__(self, grid_resolution=100, size=20):
        """
        Represents the nutrient field in the environment.
        :param grid_resolution: Resolution of the nutrient field grid
        :param size: Physical size of the grid
        """
        self.grid_resolution = grid_resolution
        self.size = size
        self.nutrient_grid = np.ones((grid_resolution, grid_resolution))
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(-10, 10, grid_resolution),
            np.linspace(-10, 10, grid_resolution)
        )
    
    def consume_nutrients(self, cells):
        """Reduces nutrient concentration where cells exist and in the surrounding environment."""
        for cell in cells:
            distances = np.sqrt((self.grid_x - cell.x) ** 2 + (self.grid_y - cell.y) ** 2)
            mask = distances <= cell.radius
            self.nutrient_grid[mask] *= 0.98  # Consume 10% of nutrients inside the entire cell area
    
    def mix_nutrients(self, diffusion_strength=0.2):
        """Redistributes nutrients via diffusion-like process."""
        self.nutrient_grid = gaussian_filter(self.nutrient_grid, sigma=diffusion_strength)
    
    def draw_nutrient_distribution(self):
        """Plots the nutrient density distribution across the entire environment using the viridis colormap."""
        plt.imshow(self.nutrient_grid, extent=[-10, 10, -10, 10], 
                   origin='lower', cmap='viridis', alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(label='Nutrient Density')
        plt.title("Nutrient Distribution After Consumption and Mixing")
        plt.show()

class Cell:
    def __init__(self, x, y, radius=1.0):
        """
        Represents a single cell in the 2D spheroid.
        :param x: X-coordinate of the cell
        :param y: Y-coordinate of the cell
        :param radius: Size of the cell (default=1.0)
        """
        self.x = x
        self.y = y
        self.radius = radius
    
    def draw(self, ax):
        """Draws the cell as a circle on the given axis."""
        circle = plt.Circle((self.x, self.y), self.radius, color='blue', alpha=0.6, edgecolor='black')
        ax.add_patch(circle)

class Spheroid:
    def __init__(self, num_cells=50, center=(0, 0), spheroid_radius=10, cell_radius=1.0, grid_resolution=100):
        """
        Creates a 2D spheroid made of individual cells that cover most of the spheroid's area.
        Cells are placed in a compact manner without using optimization-based packing.
        :param num_cells: Number of cells in the spheroid
        :param center: Center of the spheroid (x, y)
        :param spheroid_radius: Radius of the entire spheroid structure
        :param cell_radius: Radius of each cell
        :param grid_resolution: Resolution of the nutrient field grid
        """
        self.num_cells = num_cells
        self.center_x, self.center_y = center
        self.spheroid_radius = spheroid_radius
        self.cell_radius = cell_radius
        self.cells = []
        self.nutrient_field = NutrientField(grid_resolution=grid_resolution, size=20)
        self.generate_cells()
    
    def generate_cells(self):
        """Generates cells in a compact hexagonal pattern inside the spheroid."""
        row_spacing = self.cell_radius * np.sqrt(3)  # Distance between rows
        col_spacing = 2 * self.cell_radius  # Distance between columns
        
        x_min, x_max = self.center_x - self.spheroid_radius, self.center_x + self.spheroid_radius
        y_min, y_max = self.center_y - self.spheroid_radius, self.center_y + self.spheroid_radius
        
        y = y_min
        row = 0
        while y <= y_max:
            x_offset = (row % 2) * self.cell_radius  # Offset every other row for hexagonal packing
            x = x_min + x_offset
            while x <= x_max:
                if np.hypot(x - self.center_x, y - self.center_y) + self.cell_radius <= self.spheroid_radius:
                    self.cells.append(Cell(x, y, self.cell_radius))
                x += col_spacing
            y += row_spacing
            row += 1
    
    def simulate_nutrient_consumption_and_mixing(self, iterations=10, diffusion_strength=0.2):
        """Simulates nutrient consumption and mixing over multiple iterations."""
        for _ in range(iterations):
            self.nutrient_field.consume_nutrients(self.cells)
            self.nutrient_field.mix_nutrients(diffusion_strength)
    
    def draw(self):
        """Draws the entire spheroid."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.set_title("2D Spheroid Model")
        
        for cell in self.cells:
            cell.draw(ax)
        
        plt.show()

# Example Usage:
spheroid = Spheroid(num_cells=100, spheroid_radius=10, cell_radius=0.8, grid_resolution=100)
spheroid.draw()
spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()

spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()

spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()

spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()

spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()

spheroid.simulate_nutrient_consumption_and_mixing(iterations=5, diffusion_strength=0.5)
spheroid.nutrient_field.draw_nutrient_distribution()