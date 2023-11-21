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
    close_neighbors = points[sorted_indices[1:close_count + 1]]
    background_neighbors = points[sorted_indices[close_count + 1:close_count + background_count + 1]]
    irrelevant_neighbors = points[sorted_indices[close_count + background_count + 1:]]

    # Move close neighbors closer to the center and background neighbors further away
    close_neighbors_moved = close_neighbors - move_factor * (close_neighbors - center_point)
    background_neighbors_moved = background_neighbors + move_factor * (background_neighbors - center_point)

    return center, close_neighbors, background_neighbors, irrelevant_neighbors, close_neighbors_moved, background_neighbors_moved


# Function to calculate distances between points
def calculate_distances(points):
    return np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)


# Function to calculate change in distances
def calculate_distance_change(points_before, points_after):
    return np.abs(np.linalg.norm(points_after[:, np.newaxis] - points_before, axis=2))


# Generate random 2D points
np.random.seed(42)
points = np.random.rand(100, 2)

# Calculate the centroid (center) of all points
centroid = np.mean(points, axis=0)

# Find the index of the point closest to the centroid
center_index = np.argmin(np.linalg.norm(points - centroid, axis=1))  # Specify any point index as the center point

print("Index of the center point:", center_index)

# Move neighbors using the function
center, close_neighbors, background_neighbors, irrelevant_neighbors, close_neighbors_moved, background_neighbors_moved = move_neighbors(points, center_index)

# Calculate distances before and after learning
distances_before_learning = calculate_distances(points)
distances_after_learning = calculate_distances(np.vstack((center[0], close_neighbors_moved, background_neighbors_moved, irrelevant_neighbors)))

# Calculate the change in distances
distance_change = calculate_distance_change(distances_before_learning, distances_after_learning)

# Plot the un-moved points
plt.figure(figsize=(10, 10))
plt.scatter(center[0][0], center[0][1], color='red', label='Center')
plt.scatter(close_neighbors[:, 0], close_neighbors[:, 1], color='blue', label='Close Neighbors (Un-Moved)')
plt.scatter(background_neighbors[:, 0], background_neighbors[:, 1], color='black',
            label='Background Neighbors (Un-Moved)')
plt.scatter(irrelevant_neighbors[:, 0], irrelevant_neighbors[:, 1], color='grey',
            label='Irrelevant Neighbors (Un-Moved)')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Neighbor Categories (Un-Moved)')

# Display the plot
plt.show()

# Plot the moved points
plt.figure(figsize=(10, 10))
plt.scatter(center[0][0], center[0][1], color='red', label='Center')
plt.scatter(close_neighbors_moved[:, 0], close_neighbors_moved[:, 1], color='blue', label='Close Neighbors (Moved)')
plt.scatter(background_neighbors_moved[:, 0], background_neighbors_moved[:, 1], color='black',
            label='Background Neighbors (Moved)')
plt.scatter(irrelevant_neighbors[:, 0], irrelevant_neighbors[:, 1], color='grey', label='Irrelevant Neighbors (Moved)')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Neighbor Categories (Moved)')

# Display the plot
plt.show()


# Flatten arrays for plotting
plt.figure(figsize=(10, 10))
distance_before_flat = distances_before_learning.flatten()
distance_change_flat = distance_change.flatten()

# Plot the change in distance
plt.scatter(distance_before_flat, distance_change_flat, color='purple', label='Change in Distance')

# Add labels and legend
plt.xlabel('Distance Before Learning')
plt.ylabel('Change in Distance')
plt.legend()
plt.title('Change in Distance Before and After Learning')

# Display the plot
plt.show()
