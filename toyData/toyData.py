import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function to generate 3D scatter plot and return data
def generate_3d_scatter_plot(display_plot=False):
    np.random.seed(42)
    points = np.random.rand(2000, 3)  # Increase the number of points to 2000

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

# Split the data into training and testing sets (1000 points each)
train_data, test_data = points_data[:1000], points_data[1000:]
train_labels, test_labels = labels_data[:1000], labels_data[1000:]


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
input_train = torch.tensor(train_data, dtype=torch.float32)
labels_train = torch.tensor(train_labels, dtype=torch.long)

# Define testing data for the 1000 points
input_test = torch.tensor(test_data, dtype=torch.float32)
labels_test = torch.tensor(test_labels, dtype=torch.long)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Lists to store training loss values
train_loss_history = []

# Training loop
for epoch in range(10000):
    # Forward pass for training data
    output_train = model(input_train)
    loss_train = criterion(output_train, labels_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # Append the training loss to the history list
    train_loss_history.append(loss_train.item())

    # Print the loss for every 10 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss_train.item()}')

# Evaluate the model on the testing dataset and calculate accuracy
with torch.no_grad():
    output_test = model(input_test)
    _, predicted_test = torch.max(output_test, 1)
    accuracy = (predicted_test == labels_test).float().mean()
    print(f'Testing Accuracy: {accuracy.item()}')

# Plot the learning loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(0, 10000), train_loss_history, label='Training Loss')
plt.title('Learning Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
