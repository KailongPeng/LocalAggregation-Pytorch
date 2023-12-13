import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


# Function to generate 3D scatter plot and return data
def generate_3d_scatter_plot(separator=1 / 3, display_plot=False):
    np.random.seed(42)
    points = np.random.rand(2000, 3)  # Increase the number of points to 2000

    # Define the boundaries for region division based on the separator value
    boundaries = [separator, 2 * separator]

    # Assign labels to each point based on region
    labels_x = np.digitize(points[:, 0], boundaries)
    labels_y = np.digitize(points[:, 1], boundaries)
    labels_z = np.digitize(points[:, 2], boundaries)

    # Combine the indices of the regions in each dimension to get a unique label for each point
    if separator == 1 / 3:
        labels = labels_x + 3 * labels_y + 9 * labels_z
    elif separator == 1 / 2:
        labels = labels_x + 2 * labels_y + 4 * labels_z
    else:
        raise Exception("separator should be 1/3 or 1/2")
    if display_plot:
        # Create a randomized rainbow colormap
        rainbow_colormap = ListedColormap(np.random.rand(256, 3))

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with colored points based on labels using the rainbow colormap
        # scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=rainbow_colormap)
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='rainbow')

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
        # plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=rainbow_colormap)
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='rainbow')

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


def test_multiple_dotsNeighbotSIngleBatch(threeD_input=None):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    if threeD_input is None:
        threeD_input = True

    # Define toy dataset
    class ToyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Define neural network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Linear(2, 2)

        def forward(self, x):
            x = self.embedding(x)
            return x

    class SimpleFeedforwardNN(nn.Module):
        def __init__(self):
            super(SimpleFeedforwardNN, self).__init__()
            if threeD_input:
                self.input_layer = nn.Linear(3, 64)
            else:
                self.input_layer = nn.Linear(2, 64)  # 3D input layer
            self.hidden_layer1 = nn.Linear(64, 32)  # First hidden layer
            self.hidden_layer2 = nn.Linear(32, 2)  # Second-to-last layer is 2D
            # self.output_layer = nn.Linear(2, 27)  # Output layer, classifying into 27 categories

        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.hidden_layer1(x))
            x = self.hidden_layer2(x)
            # x = self.output_layer(x)
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
    if threeD_input:
        points_data, labels_data = generate_3d_scatter_plot(separator=1 / 2,
                                                            display_plot=True)  # separator should be 1/3 or 1/2
    else:
        points_data, labels_data = generate_2d_scatter_plot(display_plot=True)

    # Split the data into training and testing sets (1000 points each)
    train_data, test_data = points_data[:1000], points_data[1000:]
    train_labels, test_labels = labels_data[:1000], labels_data[1000:]

    dataset = ToyDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    # Define neural network and optimizer
    net = SimpleFeedforwardNN()
    learning_rate = 0.05
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Train network using local aggregation loss
    loss_values = []  # List to store loss values for each epoch

    # record the initial latent space
    initial_v_points = []
    initial_v_labels = []

    # record the final latent space
    final_v_points = []
    final_v_labels = []

    total_epochs = 1000

    for epoch in tqdm(range(total_epochs)):
        if epoch == int(total_epochs / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 2.0
            print(f"learning rate changed to {learning_rate / 2.0}")
        if epoch == int(total_epochs * 2 / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 4.0
            print(f"learning rate changed to {learning_rate / 4.0}")
        epoch_loss = 0.0  # Variable to accumulate loss within each epoch

        for curr_batch, (batch, batch_labels) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            embeddings = net(batch.float())
            # record initial and final latent space points
            if epoch == 0:
                initial_v_points.append(embeddings)
                initial_v_labels.append(batch_labels)

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

            # record initial and final latent space points
            if epoch == total_epochs - 1:
                final_v_points.append(embeddings)
                final_v_labels.append(batch_labels)

        # Calculate average loss for the epoch
        average_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(average_epoch_loss)

        if epoch % int(total_epochs / 10) == 0:
            # Print and record the average loss for the epoch
            print(f'Epoch [{epoch + 1}/{total_epochs}], Loss: {average_epoch_loss}')

    # Plot the loss curve
    plt.plot(range(1, total_epochs + 1), loss_values, marker='o')
    plt.title('Local Aggregation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

    # Plot initial and final latent space points with rainbow colormap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Initial latent space points
    initial_v_points = torch.cat(initial_v_points, dim=0).detach().numpy()
    initial_v_labels = torch.cat(initial_v_labels, dim=0).numpy().flatten()
    scatter_initial = axes[0].scatter(initial_v_points[:, 0], initial_v_points[:, 1],
                                      c=initial_v_labels,
                                      cmap='rainbow', marker='o')
    axes[0].set_title('Initial Latent Space Points')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    fig.colorbar(scatter_initial, ax=axes[0], label='Point Index')

    # Final latent space points
    final_v_points = torch.cat(final_v_points, dim=0).detach().numpy()
    final_v_labels = torch.cat(final_v_labels, dim=0).numpy().flatten()
    scatter_final = axes[1].scatter(final_v_points[:, 0], final_v_points[:, 1],
                                    c=final_v_labels,
                                    cmap='rainbow', marker='o')
    axes[1].set_title('Final Latent Space Points')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    fig.colorbar(scatter_final, ax=axes[1], label='Point Index')

    plt.tight_layout()
    plt.show()


test_multiple_dotsNeighbotSIngleBatch(threeD_input=False)

"""
    representational level
        for every pair of points, calculate the distance between them before and after training for each batch, 
        calculate the difference between the distances and called it the learning 
        The distance between the points in the latent space before training is recorded. The opposite of this distance is called co-activation.
        
        Then the co-activation and learning are plotted against each other as X and Y axes respectively. This should be the representational level NMPH curve.
        
    synaptic level
        for the connection between the penultimate layer and the final layer, record the weight before and after training and calculate the difference between them.
        record the activation of the penultimate layer and the final layer before and after training.
        plot the weight difference against the co-activation as X and Y axes respectively. This should be the synaptic level NMPH curve.  
"""

