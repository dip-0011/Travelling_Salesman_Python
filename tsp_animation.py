import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools
import time

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def total_distance(path, points):
    return sum(distance(points[path[i]], points[path[i+1]]) for i in range(len(path)-1)) + distance(points[path[-1]], points[path[0]])

def brute_force_tsp(points):
    num_points = len(points)
    min_distance = float('inf')
    min_path = []

    for perm in itertools.permutations(range(num_points)):
        d = total_distance(perm, points)
        if d < min_distance:
            min_distance = d
            min_path = perm

    return min_path, min_distance

def animate_tsp(points, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    line, = ax.plot([], [], 'b-')
    points_x = points[:, 0]
    points_y = points[:, 1]
    
    ax.scatter(points[:, 0], points[:, 1], color='red', zorder=2)
    ax.grid(True, linestyle='--', zorder=1)
    
    annotations = [ax.text(points_x[i], points_y[i], str(i), color='black', fontsize=10) for i in range(len(points_x))]


    # for i in  range(num_points-1):
    #     ax.text(points[:, 0], points[:, 1], str(i), color='black', fontsize=10)

    def update(frame):
        if frame == len(path):
            frame = 0
        x = [points_x[path[i]] for i in range(frame+1)]
        y = [points_y[path[i]] for i in range(frame+1)]
        line.set_data(x, y)
        
        for i, annotation in enumerate(annotations):
            annotation.set_position((points_x[i], points_y[i]))

        # # Add text annotations for each point
        # for i, (x_i, y_i) in enumerate(zip(x, y)):
        #     label = path[i]
        #     ax.text(x_i, y_i, label, color='black', fontsize=10)
        #     print(label)
        
        # Connect the end points to form a closed loop
        if frame == len(path) - 1:
            x.append(points_x[path[0]])
            y.append(points_y[path[0]])
            line.set_data(x, y)
        
        return line,*annotations

    ani = FuncAnimation(fig, update, frames=len(path)+1, interval=500, blit=True, repeat=True)
    ani.save('tsp_animation.gif', writer='pillow')  # 'pillow' is the writer for GIF format
    plt.show()

# Generate random points
np.random.seed(0)
num_points = 8
points = np.random.randint(0, 100, size=(num_points, 2))

# Compute TSP solution
start_time = time.time()
best_path, min_distance = brute_force_tsp(points)
end_time = time.time()
print("Shortest path:", best_path)
print("Shortest distance:", min_distance)
print("Time taken:", end_time - start_time, "seconds")

# Animate the TSP solution
animate_tsp(points, best_path)
