import numpy as np
import matplotlib.pyplot as plt


def move_neighbors(points, center_point_index, close_count=None, background_count=None, move_factor=None):
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
def calculate_distances(points_):
    # Euclidean distance between two points
    distance_matrix = np.zeros((points_.shape[0], points_.shape[0]))
    for _i in range(len(points_)):
        for _j in range(len(points_)):
            distance_matrix[_i][_j] = np.linalg.norm(points_[_i] - points_[_j])
    return distance_matrix
    # return np.linalg.norm(_points[:, np.newaxis, :] - _points, axis=2)
    # calculate_distances(np.array([[1,1],
    #                              [1,1],
    #                              [8,8],
    #                              [0,0],
    #                              [3,4]]))


# Function to create a label matrix based on point categories
def create_label_matrix(_num_points, num_close_neighbors, num_background_neighbors):
    # 100 points, 1 center, 20 close, 40 background, 39 irrelevant

    labels = np.zeros((_num_points, _num_points), dtype=int)

    # Label close_neighbors to close_neighbors as black
    labels[1:num_close_neighbors + 1, 1:num_close_neighbors + 1] = 1  # black

    # Label background_neighbors to background_neighbors as blue
    labels[num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors,
    num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors] = 2  # blue

    # Label irrelevant_neighbors to irrelevant_neighbors as grey
    labels[num_close_neighbors + 1 + num_background_neighbors:,
    num_close_neighbors + 1 + num_background_neighbors:] = 3  # grey

    # Label close_neighbors to center as red
    labels[1:num_close_neighbors + 1, 0] = 4  # red
    labels[0, 1:num_close_neighbors + 1] = 4  # red

    # Label background_neighbors to center as orange
    labels[num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors, 0] = 5  # orange
    labels[0, num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors] = 5  # orange

    # Label irrelevant_neighbors to center as purple
    labels[num_close_neighbors + 1 + num_background_neighbors:, 0] = 6  # purple
    labels[0, num_close_neighbors + 1 + num_background_neighbors:] = 6  # purple

    # Label close_neighbors to background_neighbors as yellow
    labels[1:num_close_neighbors + 1,
    num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors] = 7  # yellow
    labels[num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors,
    1:num_close_neighbors + 1] = 7  # yellow

    # Label close_neighbors to irrelevant_neighbors as green
    labels[1:num_close_neighbors + 1,
    num_close_neighbors + 1 + num_background_neighbors:] = 8  # green
    labels[num_close_neighbors + 1 + num_background_neighbors:,
    1:num_close_neighbors + 1] = 8  # green

    # Label background_neighbors to irrelevant_neighbors as lighter grey
    labels[num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors,
    num_close_neighbors + 1 + num_background_neighbors:] = 9  # lighter grey
    labels[num_close_neighbors + 1 + num_background_neighbors:,
    num_close_neighbors + 1:num_close_neighbors + 1 + num_background_neighbors] = 9  # lighter grey

    # fill in diagonal with 0
    np.fill_diagonal(labels, 0)

    return labels


# Function to calculate change in distances
def calculate_distance_change(points_before, points_after):
    return np.abs(np.linalg.norm(points_after[:, np.newaxis] - points_before, axis=2))


# Generate random 2D points
num_points = 100
np.random.seed(42)
points = np.random.rand(num_points, 2)

# Calculate the centroid (center) of all points
centroid = np.mean(points, axis=0)

# Find the index of the point closest to the centroid
center_index = np.argmin(np.linalg.norm(points - centroid, axis=1))  # Specify any point index as the center point

print("Index of the center point:", center_index)

# Move neighbors using the function
center, close_neighbors, background_neighbors, irrelevant_neighbors, close_neighbors_moved, background_neighbors_moved = move_neighbors(
    points, center_index, close_count=20, background_count=40, move_factor=0.3)

# Create a label matrix
label_matrix = create_label_matrix(num_points, len(close_neighbors), len(background_neighbors))

# Calculate distances before and after learning
distances_before_learning = calculate_distances(
    np.vstack((center[0],
               close_neighbors,
               background_neighbors,
               irrelevant_neighbors
               )))  # input (100, 2); output (100, 100)
distances_after_learning = calculate_distances(
    np.vstack((center[0],
               close_neighbors_moved,
               background_neighbors_moved,
               irrelevant_neighbors
               )))  # input (100, 2); output (100, 100)

# Calculate the change in distances
distance_change = distances_after_learning - distances_before_learning


def plot_dots():
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


plot_dots()

# Flatten arrays for plotting
plt.figure(figsize=(15, 10))
distance_before_flat = distances_before_learning.reshape(-1)
distance_change_flat = distance_change.reshape(-1)
label_matrix_flat = label_matrix.reshape(-1)


def plot_distance_change_components(co_activation, similarity_change, label_matrix_flat, use_distance=False):
    # Define a custom colormap with specific colors and alpha channel for transparency
    colors_with_alpha_original = {
        0: (1, 1, 1, 0),  # none
        1: (0, 0, 0, 0.1),  # black  # 'close_neighbors to close_neighbors'
        2: (0, 0, 1, 0.1),  # blue  # 'background_neighbors to background_neighbors'
        3: (1, 0.5, 0, 0.1),  # orange  # 'irrelevant_neighbors to irrelevant_neighbors'

        4: (1, 0, 0, 0.1),  # red  # 'close_neighbors to center'
        5: (0, 1, 1, 0.1),  # cyan  # 'background_neighbors to center'
        6: (0.5, 0, 0.5, 0.1),  # purple  # 'irrelevant_neighbors to center'

        7: (1, 1, 0, 0.1),  # yellow  # 'close_neighbors to background_neighbors'
        8: (0, 1, 0, 0.1),  # green  # 'close_neighbors to irrelevant_neighbors'
        9: (0.7, 0.7, 0.7, 0.1)  # light grey  # 'background_neighbors to irrelevant_neighbors'
    }

    titles = {
        0: 'None',
        1: 'black: close_neighbors to close_neighbors ',
        2: 'blue: background_neighbors to background_neighbors',
        3: 'orange: irrelevant_neighbors to irrelevant_neighbors',

        4: 'red: close_neighbors to center',
        5: 'cyan: background_neighbors to center',
        6: 'purple: irrelevant_neighbors to center',

        7: 'yellow: close_neighbors to background_neighbors',
        8: 'green: close_neighbors to irrelevant_neighbors',
        9: 'light grey: background_neighbors to irrelevant_neighbors',
        10: 'All'
    }
    # Iterate for 9 times, each time set all colors but one to be transparent
    for i in range(11):
        # Set all colors to be transparent
        colors_with_alpha = colors_with_alpha_original.copy()
        for j in range(10):
            if j != i:
                colors_with_alpha[j] = (
                    colors_with_alpha_original[j][0], colors_with_alpha_original[j][1], colors_with_alpha_original[j][2], 0)
            else:
                colors_with_alpha[j] = (
                    colors_with_alpha_original[j][0], colors_with_alpha_original[j][1], colors_with_alpha_original[j][2], 1)
        if i == 10:
            colors_with_alpha = colors_with_alpha_original

        # Function to convert label_matrix_flat values to RGBA colors
        def map_labels_to_colors(labels, label_colors):
            return [label_colors[label] for label in labels]

        # Convert label_matrix_flat values to RGBA colors
        colors = map_labels_to_colors(label_matrix_flat, colors_with_alpha)

        _ = plt.figure()
        scatter = plt.scatter(co_activation, similarity_change, c=colors,
                              label='Change in Distance')

        # Display the plot
        plt.title(f'{titles[i]}')
        plt.xlabel('Distance Before Learning')
        plt.ylabel('Change in Distance')
        plt.show()

    # Display the plot
    plt.show()

    # Plot the histogram of change in distance
    _ = plt.figure(figsize=(10, 10))
    _ = plt.hist(similarity_change, bins=100)
    if use_distance:
        _ = plt.title("distance_change_flat hist")
    else:
        _ = plt.title("similarity change hist")


use_distance = False
if use_distance:
    plot_distance_change_components(distance_before_flat, distance_change_flat, label_matrix_flat, use_distance=use_distance)
else:
    plot_distance_change_components(- distance_before_flat, - distance_change_flat, label_matrix_flat, use_distance=use_distance)


def multiple_pull_push():
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

    # Generate random 2D points
    np.random.seed(42)
    points = np.random.rand(100, 2)
    # original_points = points.copy()

    # List to store initial and final points
    initial_points_list = []
    final_points_list = []

    # Iterate through all possible center indices
    for center_index in range(len(points)):
        current_center_index = np.random.randint(len(points))

        # Move neighbors using the function
        center, close_neighbors, background_neighbors, irrelevant_neighbors, close_neighbors_moved, background_neighbors_moved = move_neighbors(
            points, current_center_index)

        # Update the points with the moved neighbors
        points = np.vstack((center[0], close_neighbors_moved, background_neighbors_moved, irrelevant_neighbors))

        # Record initial and final points
        initial_points_list.append(np.vstack((center[0], close_neighbors, background_neighbors, irrelevant_neighbors)))
        final_points_list.append(
            np.vstack((center[0], close_neighbors_moved, background_neighbors_moved, irrelevant_neighbors)))

    # Plot the points before any movement
    plt.figure(figsize=(10, 10))
    plt.scatter(initial_points_list[0][:, 0], initial_points_list[0][:, 1], alpha=0.5, color='grey')

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Points Before Any Movement')

    # Display the plot
    plt.show()

    # Plot the points after all movements
    plt.figure(figsize=(10, 10))
    final_points = final_points_list[-1]
    plt.scatter(final_points[:, 0], final_points[:, 1], alpha=0.5, color='grey')

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Points After All Movements')

    # Display the plot
    plt.show()

    # Calculate distances before and after learning
    distances_before_learning = calculate_distances(initial_points_list[0])  # input (100, 2); output (100, 100)
    distances_after_learning = calculate_distances(final_points_list[-1])  # input (100, 2); output (100, 100)

    # Calculate the change in distances
    # distance_change = calculate_distance_change(distances_before_learning, distances_after_learning)
    distance_change = distances_after_learning - distances_before_learning

    # Flatten arrays for plotting
    plt.figure(figsize=(10, 10))
    distance_before_flat = distances_before_learning.flatten()
    distance_change_flat = distance_change.flatten()

    # Plot the change in distance
    plt.scatter(distance_before_flat, distance_change_flat, color='purple', label='Change in Distance', alpha=0.5)

    # Add labels and legend
    plt.xlabel('Distance Before Learning')
    plt.ylabel('Change in Distance')
    plt.legend()
    plt.title('Change in Distance Before and After Learning')

    # Display the plot
    plt.show()

    # Plot the histogram of change in distance
    plt.figure(figsize=(10, 10))
    _ = plt.hist(distance_change_flat, bins=100)
    _ = plt.title("distance_change_flat hist")
