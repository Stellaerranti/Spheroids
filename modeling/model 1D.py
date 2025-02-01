import numpy as np
import matplotlib.pyplot as plt
import random

class Cell:
    def __init__(self, x, radius=1, filtration_rate=0.1, saturation_boundary=0.9, decay_rate=0.1, death_boundary=0.7, consumption=0.02,
                 division_chance=0.3, division_boundary=0.8):
        self.x = x
        self.radius = radius
        self.filtration_rate = filtration_rate
        self.saturation = 1
        self.saturation_coeff = filtration_rate
        self.saturation_boundary = saturation_boundary
        self.decay_rate = decay_rate
        self.alive = True
        self.consumption = consumption
        self.division_chance = division_chance
        self.death_boundary = death_boundary
        self.division_boundary = division_boundary
    
    def draw(self, ax):
        color = 'blue' if self.alive else 'red'
        circle = plt.Circle((self.x, 0), self.radius, color=color, alpha=0.6, edgecolor='black')
        ax.add_patch(circle)
        
    def divide(self, spheroid):
        if random.random() < self.division_chance:
            direction = random.choice([-1, 1])  # -1 for left, 1 for right
            new_x = self.x + direction * (spheroid.cell_radius * 2 )  # Ensure separation
            
            spheroid.push_cells(self.x, new_x, direction)
            
            new_cell = Cell(new_x, self.radius, self.filtration_rate, self.saturation_boundary,
                            self.decay_rate, self.death_boundary, self.consumption, self.division_chance, self.division_boundary)
            spheroid.cells.append(new_cell)
            spheroid.num_cells += 1
            spheroid.length = spheroid.num_cells * spheroid.cell_radius * 2            
            spheroid.compact()
    
    def live(self, spheroid):
        if not self.alive:
            return
        self.saturation -= self.consumption
        
        if self.saturation < self.saturation_boundary:
            self.radius *= (1 - self.decay_rate)
        
        if(self.saturation > self.saturation_boundary and self.radius < spheroid.cell_radius):
            self.radius*= (1 + self.decay_rate)
        
        if self.radius < self.death_boundary:
            self.alive = False
            return
        
        if self.saturation > self.division_boundary:
            self.divide(spheroid)
            

class Spheroid:
    def __init__(self, num_cells, cell_radius, diffusion_coefficient, dx, dt):
        self.num_cells = num_cells
        self.cell_radius = cell_radius
        self.diffusion_coefficient = diffusion_coefficient
        self.dx = dx
        self.dt = dt
        self.length = num_cells * cell_radius * 2
        self.grid_size = int(self.length / dx) + 1
        self.concentration = np.ones(self.grid_size)
        self.generate_cells()
    
    def generate_cells(self):
        self.cells = []
        spacing = 2 * self.cell_radius
        start_pos = -((self.num_cells - 1) * spacing) / 2
        for i in range(self.num_cells):
            self.cells.append(Cell(start_pos + i * spacing, self.cell_radius))
    
    def push_cells(self, parent_x, new_x, direction):

        if (parent_x < 0):            
            for cell in self.cells:
                if cell.x <= new_x:
                    cell.x -= 2*self.cell_radius            
                                    
        if (parent_x > 0):            
            for cell in self.cells:
                if cell.x >= new_x:
                    cell.x += 2*self.cell_radius 
        
        if (parent_x == 0):
            if (direction == 1):
                for cell in self.cells:
                    if cell.x > 0:
                        cell.x += 2*self.cell_radius 
            if (direction == -1):
               for cell in self.cells:
                   if cell.x < 0:
                       cell.x -= 2*self.cell_radius 
            
    
    def diffuse_and_filter(self):
        alpha = self.diffusion_coefficient * self.dt / (self.dx ** 2)
        new_concentration = self.concentration.copy()
        for i in range(1, self.grid_size - 1):
            new_concentration[i] = (
                self.concentration[i] + alpha * (self.concentration[i + 1] - 2 * self.concentration[i] + self.concentration[i - 1])
            )
        for cell in self.cells:
            if not cell.alive:
                break
            center_idx = int((cell.x + self.length / 2) / self.dx)
            radius_range = int(cell.radius / self.dx)
            consumed_amount = 0
            for j in range(-radius_range, radius_range + 1):
                idx = center_idx + j
                if 0 <= idx < self.grid_size:
                    distance_factor = 1 - (j / radius_range) ** 2  # Parabolic filtration function
                    filtered = new_concentration[idx] * cell.filtration_rate * distance_factor
                    consumed_amount += filtered
                    new_concentration[idx] *= (1 - cell.filtration_rate * distance_factor)
                 
            cell.saturation = min(1, cell.saturation + consumed_amount * cell.saturation_coeff)        
            cell.live(self)            
        self.concentration = new_concentration        
    
    def update_system(self, iterations):        
        for _ in range(iterations):      
            self.diffuse_and_filter()
            self.compact()        
    
    def compact(self):
        reference_cell = min(self.cells, key=lambda cell: abs(cell.x))
        reference_x = reference_cell.x  # Keep reference cell in place
        
        # Separate cells into positive and negative sides, sorting them by distance from reference
        negative_side = sorted([cell for cell in self.cells if cell.x < reference_x], key=lambda cell: -cell.x)
        positive_side = sorted([cell for cell in self.cells if cell.x > reference_x], key=lambda cell: cell.x)
        
        # Move positive side towards reference cell
        for i in range(len(positive_side)):
            prev_cell = positive_side[i - 1] if i > 0 else reference_cell
            expected_x = prev_cell.x + prev_cell.radius + positive_side[i].radius + self.dx
            if positive_side[i].x > expected_x:
                positive_side[i].x = expected_x
        
        # Move negative side towards reference cell
        for i in range(len(negative_side)):
            prev_cell = negative_side[i - 1] if i > 0 else reference_cell
            expected_x = prev_cell.x - prev_cell.radius - negative_side[i].radius - self.dx
            if negative_side[i].x < expected_x:
                negative_side[i].x = expected_x 
    
    def plot_concentration(self):
        x_positions = np.linspace(-self.length / 2, self.length / 2, self.grid_size)
        plt.figure(figsize=(8, 4))
        plt.plot(x_positions, self.concentration, label='Concentration Profile')
        plt.ylim(0, 1.1)
        plt.xlabel('Position')
        plt.ylabel('Concentration')
        plt.title('Diffusion with Parabolic Filtration')
        plt.legend()
        plt.show()
    
    def plot_cells(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        #ax.set_xlim(-self.length / 2, self.length / 2)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title("Spheroid Cell Distribution")
        for cell in self.cells:
            cell.draw(ax)
        plt.show()
    
# Example Usage:
plt.close('all')    
    
spheroid = Spheroid(num_cells=5, cell_radius=1, diffusion_coefficient=0.1, dx=0.1, dt=0.01)
spheroid.update_system(iterations=1)
spheroid.plot_cells()
