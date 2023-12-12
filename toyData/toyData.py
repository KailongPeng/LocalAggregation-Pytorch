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
        self.hidden_layer2 = nn.Linear(32, 2)  # Second-to-last layer is 2D
        self.output_layer = nn.Linear(2, 27)  # Output layer, classifying into 27 categories

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
initial_learning_rate = 0.05
optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
# Lists to store training loss values
train_loss_history = []
# Total number of epochs
total_epochs = 10000
# Training loop
for epoch in range(total_epochs):
    # Adjust learning rate if epoch passes 1/3 of the total epochs
    if epoch > total_epochs / 3:
        optimizer.param_groups[0]['lr'] = initial_learning_rate / 2.0

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


def localAgg_test():
    """
        The definition of the local aggregation loss is as follows: for a specified data point
        x_i, identify c points as close neighbors (C_i) and b points as background neighbors (B_i). Then, the loss is defined as:
        \mathcal{L}_i=L\left(\mathbf{C}_i, \mathbf{B}_i \mid \boldsymbol{\theta}, \mathbf{x}_i\right)+\lambda\|\boldsymbol{\theta}\|_2^2
        Here, θ represents the model parameters, and λ is a regularization parameter. The goal is to minimize this final loss during training.

        Here, L\left(\mathbf{C}_i, \mathbf{B}_i \mid \boldsymbol{\theta}, \mathbf{x}_i\right)=-\log \frac{P\left(\mathbf{C}_i\mid \mathbf{v}_i\right)}{P\left(\mathbf{B}_i \mid \mathbf{v}_i\right)}
        Where v _i  represents the feature vector associated with x_i

        P\left(\mathbf{A}_i \mid \mathbf{v}_i\right) is the similarity between the set A and an arbitrary feature v. This is calculated as the opposite of the mean Euclidean distance between every element in A and v.

        Based on this definition of the local aggregation loss, for a 3D input (x,y,z) with 1000 points uniformly distributed in the range [0, 1] for each dimension, a fully connected feedforward neural network is created. The structure consists of 3 layers, with the input being 3D and the last layer being 2D. The output of the last 2D layer is then used as the v_i space for local aggregation. Training is performed for local aggregation based on the given 1000 points' training data, and the loss is recorded.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Define the local aggregation loss
    class LocalAggregationLoss(nn.Module):
        def __init__(self, lambda_reg):
            super(LocalAggregationLoss, self).__init__()
            self.lambda_reg = lambda_reg

        @staticmethod
        def similarity(A, v):
            # Calculate the similarity between set A and feature v
            # (opposite of mean Euclidean distance)
            distance = torch.norm(A - v, dim=1)
            similarity = torch.exp(-distance).mean()

            return similarity

        def forward(self, Ci, Bi, theta, xi, vi):
            # Compute the similarity between Ci and vi
            P_Ci_given_vi = self.similarity(Ci, vi)

            # Compute the similarity between Bi and vi
            P_Bi_given_vi = self.similarity(Bi, vi)

            # Compute the negative log ratio of probabilities
            loss_local_aggregation = -torch.log(P_Ci_given_vi / P_Bi_given_vi)

            # Regularization term
            reg_term = self.lambda_reg * torch.norm(theta, p=2)

            # Total loss
            total_loss = loss_local_aggregation + reg_term

            return total_loss

    # Define the neural network model
    class SimpleFeedforwardNN(nn.Module):
        def __init__(self):
            super(SimpleFeedforwardNN, self).__init__()
            self.input_layer = nn.Linear(3, 64)  # 3D input layer
            self.hidden_layer1 = nn.Linear(64, 32)  # First hidden layer
            self.hidden_layer2 = nn.Linear(32, 2)  # Second-to-last layer is 2D

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.hidden_layer1(x))
            x = self.hidden_layer2(x)
            return x

    # Generate 1000 random 3D points
    input_data = torch.rand(1000, 3)

    # Instantiate the neural network model and the local aggregation loss
    model = SimpleFeedforwardNN()
    local_aggregation_loss = LocalAggregationLoss(lambda_reg=0.001)

    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1000):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        vi = model(input_data)
        loss = local_aggregation_loss(Ci, Bi, theta, xi, vi)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print the loss for every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
