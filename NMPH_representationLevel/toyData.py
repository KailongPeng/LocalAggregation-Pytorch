import numpy as np
import matplotlib.pyplot as plt


def move_neighbors(points, center_point_index, close_count=20, background_count=40, move_factor=0.3):
    # Calculate the centroid (center) of all points
    center_point = points[center_point_index]

    # Calculate distances from the center to all other points
    distances = np.linalg.norm(points - center_point, axis=1)

    # Sort indices based on distances
    sorted_indices = np.argsort(distances)

    # Define categories for points
    center = [center_point]
    close_neighbors = points[sorted_indices[1:close_count+1]]
    background_neighbors = points[sorted_indices[close_count+1:close_count+background_count+1]]
    irrelevant_neighbors = points[sorted_indices[close_count+background_count+1:]]

    # Move close neighbors closer to the center and background neighbors further away
    close_neighbors_moved = close_neighbors - move_factor * (close_neighbors - center_point)
    background_neighbors_moved = background_neighbors + move_factor * (background_neighbors - center_point)

    return center, close_neighbors_moved, background_neighbors_moved, irrelevant_neighbors

# Generate random 2D points
np.random.seed(42)
points = np.random.rand(100, 2)

# Calculate the centroid (center) of all points
centroid = np.mean(points, axis=0)

# Find the index of the point closest to the centroid
center_index = np.argmin(np.linalg.norm(points - centroid, axis=1))
# # Specify any point index as the center point
# center_index = 50  # Change this index as needed

print("Index of the center point:", center_index)


# Move neighbors using the function
center, close_neighbors_moved, background_neighbors_moved, irrelevant_neighbors = move_neighbors(points, center_index)

# Plot the points
plt.scatter(center[0][0], center[0][1], color='red', label='Center')
plt.scatter(close_neighbors_moved[:, 0], close_neighbors_moved[:, 1], color='blue', label='Close Neighbors')
plt.scatter(background_neighbors_moved[:, 0], background_neighbors_moved[:, 1], color='black', label='Background Neighbors')
plt.scatter(irrelevant_neighbors[:, 0], irrelevant_neighbors[:, 1], color='grey', label='Irrelevant Neighbors')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Neighbor Categories (Moved)')

# Display the plot
plt.show()
