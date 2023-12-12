import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def generate_3d_scatter_plot(display_plot=False):
    # Generate 1000 random 3D points
    np.random.seed(42)
    points = np.random.rand(1000, 3)

    # Define the boundaries for region division
    boundaries = [1 / 3, 2 / 3]

    # Assign labels to each point based on region
    labels_x = np.digitize(points[:, 0], boundaries)
    labels_y = np.digitize(points[:, 1], boundaries)
    labels_z = np.digitize(points[:, 2], boundaries)

    # Combine the indices of the regions in each dimension to get a unique label for each point
    labels = labels_x + 3 * labels_y + 9 * labels_z

    if display_plot:
        # Create a randomized rainbow colormap
        rainbow_colormap = ListedColormap(np.random.rand(256, 3))

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with colored points based on labels using the rainbow colormap
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=rainbow_colormap)

        # Add colorbar to show label-color mapping
        colorbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
        colorbar.set_label('Label')

        # Set plot labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Scatter Plot with Region Labels')

        # Display the plot
        plt.show()

    # Return data and figure
    return points, labels


# Call the function to get data and figure
points_data, labels_data = generate_3d_scatter_plot(display_plot=True)


# For a 3D input (x, y, z) with 1000 points, uniformly distributed in the range [0, 1] for each dimension.
# Each point is assigned to one of 27 labels ranging from 0 to 26.
# Now, let's create a fully connected feedforward neural network with 4 layers.
# The input is 3D, the second-to-last layer is 2D, and the last layer classifies into 27 categories.
import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network model
class SimpleFeedforwardNN(nn.Module):
    def __init__(self):
        super(SimpleFeedforwardNN, self).__init__()
        self.input_layer = nn.Linear(3, 64)  # 3D input layer
        self.hidden_layer1 = nn.Linear(64, 32)  # First hidden layer
        self.hidden_layer2 = nn.Linear(32, 2)   # Second-to-last layer is 2D
        self.output_layer = nn.Linear(2, 27)    # Output layer, classifying into 27 categories

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


# Instantiate the neural network model
model = SimpleFeedforwardNN()

# Define training data for the 1000 points
# input_data = points_data
input_data = torch.tensor(points_data, dtype=torch.float32)
labels = torch.tensor(labels_data, dtype=torch.long)  # torch.randint(0, 27, (1000,))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(input_data)

    # Compute loss
    loss = criterion(output, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    # Epoch 0, Loss: 3.3573741912841797
    # Epoch 10, Loss: 3.3567841053009033
    # Epoch 20, Loss: 3.3561997413635254
    # Epoch 30, Loss: 3.355625867843628
    # Epoch 40, Loss: 3.3550524711608887
    # Epoch 50, Loss: 3.354484796524048
    # Epoch 60, Loss: 3.353917121887207
    # Epoch 70, Loss: 3.3533620834350586
    # Epoch 80, Loss: 3.3528056144714355
    # Epoch 90, Loss: 3.35225510597229
