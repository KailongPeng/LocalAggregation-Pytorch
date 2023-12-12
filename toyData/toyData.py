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


def generate_2d_scatter_plot(display_plot=False):
    # Example usage:
    # points_2d, labels_2d = generate_2d_scatter_plot(display_plot=True)
    np.random.seed(42)
    points = np.random.rand(2000, 2)  # Generate 2D points

    # Define the boundaries for region division
    boundaries = [1 / 3, 2 / 3]

    # Assign labels to each point based on region
    labels_x = np.digitize(points[:, 0], boundaries)
    labels_y = np.digitize(points[:, 1], boundaries)

    # Combine the indices of the regions in each dimension to get a unique label for each point
    labels = labels_x + 3 * labels_y

    if display_plot:
        # Create a randomized rainbow colormap
        rainbow_colormap = ListedColormap(np.random.rand(256, 3))

        # Create a 2D scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=rainbow_colormap)

        # Add colorbar to show label-color mapping
        colorbar = plt.colorbar(orientation='vertical')
        colorbar.set_label('Label')

        # Set plot labels
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Scatter Plot with Region Labels')

        # Display the plot
        plt.show()

    # Return data
    return points, labels


def trainWith_crossEntropyLoss():
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


def trainWith_localAggLoss():
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
            # Calculate the similarity between set A and feature v using Gaussian kernel
            distance = torch.norm(A - v, dim=1)
            # import pdb; pdb.set_trace()
            _similarity_ = - distance.mean() + 1e-6
            # _similarity_ = torch.exp(-distance ** 2).mean()

            return _similarity_

        def forward(self, Ci, Bi, theta, xi, vi):
            # Compute the similarity between Ci and vi
            P_Ci_given_vi = self.similarity(Ci, vi)

            # Compute the similarity between Bi and vi
            P_Bi_given_vi = self.similarity(Bi, vi)

            # Compute the negative log ratio of probabilities
            loss_local_aggregation = - torch.log(P_Ci_given_vi / P_Bi_given_vi)
            # loss_local_aggregation = torch.log(P_Ci_given_vi / P_Bi_given_vi)

            # Regularization term
            reg_term = self.lambda_reg * torch.norm(torch.cat([p.view(-1) for p in theta]), p=2)

            # Total loss
            total_loss = loss_local_aggregation + reg_term

            return total_loss

    # Define the neural network model
    class SimpleFeedforwardNN(nn.Module):
        def __init__(self, inputSize=2):
            # set random seed
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            super(SimpleFeedforwardNN, self).__init__()
            self.input_layer = nn.Linear(inputSize, 64)  # 3D input layer
            self.hidden_layer1 = nn.Linear(64, 32)  # First hidden layer
            self.hidden_layer2 = nn.Linear(32, 2)  # Second-to-last layer is 2D

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.hidden_layer1(x))
            x = self.hidden_layer2(x)
            # x = torch.sigmoid(self.hidden_layer2(x))  # Apply sigmoid to ensure values are between 0 and 1
            return x

    # Generate 1000 random 3D points
    # points_data, labels_data = generate_3d_scatter_plot(display_plot=True)
    points_data, labels_data = generate_2d_scatter_plot(display_plot=True)

    # Split the data into training and testing sets (1000 points each)
    train_data, test_data = points_data[:1000], points_data[1000:]
    train_labels, test_labels = labels_data[:1000], labels_data[1000:]
    input_data = torch.tensor(train_data, dtype=torch.float32)

    # Instantiate the neural network model and the local aggregation loss
    model = SimpleFeedforwardNN(inputSize=2)
    # local_aggregation_loss = LocalAggregationLoss(lambda_reg=0)
    local_aggregation_loss = LocalAggregationLoss(lambda_reg=0.001)

    # Set up optimizer
    initial_learning_rate = 0.5
    total_epochs = 200
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)

    # Initialize lists to store initial and final latent space points
    initial_v_points = []
    final_v_points = []

    testMode = True
    if testMode:
        plot_neighborhood = True
    else:
        plot_neighborhood = False

    # Training loop
    loss_values = []
    for epoch in range(total_epochs):
        # Record initial and final latent space points
        if epoch == 0:
            initial_v_points = model(input_data).detach().numpy()
        if epoch == total_epochs - 1:
            final_v_points = model(input_data).detach().numpy()

        # Adjust learning rate if epoch passes 1/3 of the total epochs
        if epoch > total_epochs / 3:
            optimizer.param_groups[0]['lr'] = initial_learning_rate / 2.0
        # Zero gradients
        optimizer.zero_grad()

        i = epoch % len(input_data)
        xi = input_data[i]
        vi = model(xi)  # in latent space

        # Calculate Euclidean distances
        distances = torch.norm(model(input_data) - vi, dim=1)

        # Find the indices of the closest c and b points
        c = 40
        b = 40
        _, closest_c_indices = torch.topk(-distances, c)
        _, closest_b_indices = torch.topk(-distances, b + c)
        closest_b_indices = closest_b_indices[c:]

        if plot_neighborhood:
            # plot in latent space, closest_c_indices as blue and closest_b_indices as black and the chosen vi as red
            latent_points = model(input_data).detach().numpy()
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)
            ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                       label='Other Points', alpha=0.2)
            ax.scatter(latent_points[closest_c_indices, 0], latent_points[closest_c_indices, 1],
                       c='blue', marker='o', label='Closest C Points')
            ax.scatter(latent_points[closest_b_indices, 0], latent_points[closest_b_indices, 1],
                       c='black', marker='o', label='Closest B Points')
            ax.scatter(vi.detach().numpy()[0], vi.detach().numpy()[1], c='red', marker='*', s=200, label='Chosen vi')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_title(f'Epoch {epoch} before training')
            ax.legend()
            plt.show()

        # Get the actual latent vectors for C_i and B_i
        Ci = model(input_data[closest_c_indices])  # current closest c points in latent space as close neighbors (C_i)
        Bi = model(
            input_data[closest_b_indices])  # current closest b points in latent space as background neighbors (B_i)
        theta = model.parameters()

        # Forward pass
        loss = local_aggregation_loss(Ci, Bi, theta, xi, vi)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print the loss for every 100 epochs
        if epoch % int(total_epochs / 10) == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss_values.append(loss.item())

        if plot_neighborhood:
            # plot in latent space, closest_c_indices as blue and closest_b_indices as black and the chosen vi as red
            latent_points = model(input_data).detach().numpy()
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111)
            ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                       label='Other Points', alpha=0.2)
            ax.scatter(latent_points[closest_c_indices, 0], latent_points[closest_c_indices, 1],
                       c='blue', marker='o', label='Closest C Points')
            ax.scatter(latent_points[closest_b_indices, 0], latent_points[closest_b_indices, 1],
                       c='black', marker='o', label='Closest B Points')
            vi = model(xi).detach().numpy()  # in latent space
            ax.scatter(vi[0], vi[1], c='red', marker='*', s=200, label='Chosen vi')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_title(f'Epoch {epoch} after training, loss={loss.item()}')
            ax.legend()
            plt.show()

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, total_epochs), loss_values, label='Training Loss')
    plt.title('Local Aggregation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot initial and final latent space points with rainbow colormap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Initial latent space points
    scatter_initial = axes[0].scatter(initial_v_points[:, 0], initial_v_points[:, 1], c=range(len(initial_v_points)),
                                      cmap='rainbow', marker='o')
    axes[0].set_title('Initial Latent Space Points')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    fig.colorbar(scatter_initial, ax=axes[0], label='Point Index')

    # Final latent space points
    scatter_final = axes[1].scatter(final_v_points[:, 0], final_v_points[:, 1], c=range(len(final_v_points)),
                                    cmap='rainbow', marker='o')
    axes[1].set_title('Final Latent Space Points')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    fig.colorbar(scatter_final, ax=axes[1], label='Point Index')

    plt.tight_layout()
    plt.show()


# add another loss so that the latent space (aka v=model(x)) is encouraged to span 0-1.
# layer norm versus batch norm


def test():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np

    # Define toy dataset
    class ToyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Define neural network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Linear(2, 2)

        def forward(self, x):
            x = self.embedding(x)
            return x

    # Define local aggregation loss function
    def local_aggregation_loss(embeddings, close_neighbors, background_neighbors):
        # Compute pairwise distances between embeddings and close neighbors  # embeddings(50,2) close_neighbors(50,10,2) -> (50,10)
        expanded_embeddings = embeddings.unsqueeze(1).expand_as(
            close_neighbors)  # Expand the embeddings to have the same dimensions as close_neighbors
        close_distances = torch.norm(expanded_embeddings - close_neighbors,
                                     dim=2)  # Calculate the Euclidean distance

        # Compute pairwise distances between embeddings and background neighbors
        expanded_embeddings = embeddings.unsqueeze(1).expand_as(
            background_neighbors)  # Expand the embeddings to have the same dimensions as background_neighbors
        background_distances = torch.norm(expanded_embeddings - background_neighbors,
                                          dim=2)  # Calculate the Euclidean distance

        # Compute loss based on distances
        loss = torch.mean(torch.log(1 + torch.exp(close_distances - background_distances)))
        return loss

    # Define close and background neighbors
    def get_neighbors(embeddings, c=None, b=None):
        # Compute pairwise distances between embeddings
        distances = torch.cdist(embeddings, embeddings)
        # Get indices of k closest neighbors for each example
        _, indices = torch.topk(distances, c + b + 1, largest=False)
        # Remove self from list of neighbors
        indices = indices[:, 1:]
        # Get embeddings of close neighbors
        close_neighbors = embeddings[indices[:, :c]]
        # Get embeddings of background neighbors
        background_neighbors = embeddings[indices[:, c:]]
        return close_neighbors, background_neighbors

    # Define toy dataset shaped [1000, 2]
    data = torch.tensor(np.random.rand(1000, 2), dtype=torch.float32)
    dataset = ToyDataset(data)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    # Define neural network and optimizer
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Train network using local aggregation loss
    loss_values = []  # List to store loss values for each epoch

    for epoch in range(10):
        epoch_loss = 0.0  # Variable to accumulate loss within each epoch

        for batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            embeddings = net(batch.float())
            # Get close and background neighbors
            c = 10
            b = 10
            close_neighbors, background_neighbors = get_neighbors(embeddings, c=c, b=b)
            # Compute loss
            loss = local_aggregation_loss(embeddings, close_neighbors, background_neighbors)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        average_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(average_epoch_loss)

        # Print and record the average loss for the epoch
        print(f'Epoch [{epoch + 1}/10], Loss: {average_epoch_loss}')

    # Plot the loss curve
    plt.plot(range(1, 11), loss_values, marker='o')
    plt.title('Local Aggregation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()
