"""

For two sets of 100 points with 2 dimensions: set1 and set2 as given in the code, note that each dot of these two sets has a corresponding dot in the other set.
Their transformation from set1 to set2 can be achieved with a simple feedforward network. Initiate and train this neural network so that it can truthfully accomplish this transformation.

Display set2 and the transformed set1 with the un-trained and trained network with random rainbow colormap


"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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


set1 = np.vstack((
    center[0],
    close_neighbors,
    background_neighbors,
    irrelevant_neighbors
    ))
set2 = np.vstack((
    center[0],
    close_neighbors_moved,
    background_neighbors_moved,
    irrelevant_neighbors
    ))

def plot_points_with_colors(points, title):
    colors = [plt.cm.rainbow(i / len(points)) for i in range(len(points))]

    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Display set1 with random rainbow colors
plot_points_with_colors(set1, 'Set 1')

# Display set2 with random rainbow colors
plot_points_with_colors(set2, 'Set 2')


# Define a simple feedforward neural network
class SimpleTransformNet(nn.Module):
    def __init__(self):
        super(SimpleTransformNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # Input layer: 2 input features, 5 hidden units
        self.relu = nn.ReLU() # ReLU activation function
        self.fc2 = nn.Linear(5, 5)  # Hidden layer: 5 hidden units, 1 output feature
        self.fc3 = nn.Linear(5, 2)  # Output layer: 5 hidden units, 2 output features

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Function to transform points using the neural network
def transform_points(net, points):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(points)
        output_tensor = net(input_tensor)
    return output_tensor.numpy()

# Display points with random rainbow colors
def plot_points_with_colors(points, title):
    colors = [plt.cm.rainbow(i / len(points)) for i in range(len(points))]

    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Generate a neural network
model = SimpleTransformNet()
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Adding weight decay

# Convert sets to PyTorch tensors
set1_tensor = torch.FloatTensor(set1)
set2_tensor = torch.FloatTensor(set2)

# Training loop for untrained network
num_epochs_untrained = 1000
for epoch in tqdm(range(num_epochs_untrained)):
    outputs = model(set1_tensor)
    loss = criterion(outputs, set2_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display untrained transformed set1 with random rainbow colors
untrained_transformed_set1 = transform_points(model, set1)
plot_points_with_colors(untrained_transformed_set1, 'Untrained Transformed Set 1')

# Retrain the neural network for better results
model = SimpleTransformNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs_untrained, num_epochs_untrained + 1000):
    outputs = model(set1_tensor)
    loss = criterion(outputs, set2_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display trained transformed set1 with random rainbow colors
trained_transformed_set1 = transform_points(model, set1)
plot_points_with_colors(trained_transformed_set1, 'Trained Transformed Set 1')

# change this code so that instead of feedforward network, it is a recurrent network
