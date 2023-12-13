import os

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


def test_multiple_dotsNeighbotSingleBatch(threeD_input=None):
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
                self.input_layer = nn.Linear(2, 64)  # first layer weight
            self.hidden_layer1 = nn.Linear(64, 32)  #  second layer weight
            self.hidden_layer2 = nn.Linear(32, 2)  # final layer is 2D

        def forward(self, x):
            # x = (50,2)  input
            x = torch.relu(self.input_layer(x))  # x = (50,64)  first layer activation
            x = torch.relu(self.hidden_layer1(x))  # x = (50,32)  second layer activation (penultimate layer)
            x = self.hidden_layer2(x)  # x = (50,2) final layer activation
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

    # Record the weights, penultimate layer activations, and final layer activations
    weight_difference_history = {'input_layer': [], 'hidden_layer1': [], 'hidden_layer2': []}
    activation_history = {'penultimate_layer_before': [], 'final_layer_before': [],
                          'penultimate_layer_after': [], 'final_layer_after': []}

    for epoch in tqdm(range(total_epochs)):
        if epoch == int(total_epochs / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 2.0
            print(f"learning rate changed to {learning_rate / 2.0}")
        if epoch == int(total_epochs * 2 / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 4.0
            print(f"learning rate changed to {learning_rate / 4.0}")
        epoch_loss = 0.0  # Variable to accumulate loss within each epoch

        for curr_batch, (batch, batch_labels) in enumerate(dataloader):
            # Record weights
            input_layer_before = net.input_layer.weight.data.clone().detach().numpy()
            hidden_layer1_before = net.hidden_layer1.weight.data.clone().detach().numpy()
            hidden_layer2_before = net.hidden_layer2.weight.data.clone().detach().numpy()

            # Record activations
            penultimate_activations = torch.relu(net.hidden_layer1(torch.relu(net.input_layer(batch.float()))))
            final_activations = net.hidden_layer2(penultimate_activations)

            activation_history['penultimate_layer_before'].append(penultimate_activations.detach().numpy())
            activation_history['final_layer_before'].append(final_activations.detach().numpy())

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

            # Record weights
            input_layer_after = net.input_layer.weight.data.clone().detach().numpy()
            hidden_layer1_after = net.hidden_layer1.weight.data.clone().detach().numpy()
            hidden_layer2_after = net.hidden_layer2.weight.data.clone().detach().numpy()

            weight_difference_history['input_layer'].append(input_layer_after - input_layer_before)
            weight_difference_history['hidden_layer1'].append(hidden_layer1_after - hidden_layer1_before)
            weight_difference_history['hidden_layer2'].append(hidden_layer2_after - hidden_layer2_before)

            # Record activations
            penultimate_activations = torch.relu(net.hidden_layer1(torch.relu(net.input_layer(batch.float()))))
            final_activations = net.hidden_layer2(penultimate_activations)

            activation_history['penultimate_layer_after'].append(penultimate_activations.detach().numpy())
            activation_history['final_layer_after'].append(final_activations.detach().numpy())

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

    return weight_difference_history, activation_history


weight_difference_history, activation_history = test_multiple_dotsNeighbotSingleBatch(threeD_input=False)

weight_difference_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/weight_difference_folder/"
os.makedirs(weight_difference_folder, exist_ok=True)
np.save(f'{weight_difference_folder}/weight_difference_history_input_layer.npy', np.asarray(weight_difference_history['input_layer']))  # (20000, 64, 2)  1000points/50imagesPerBatch=20batchPerEpoch, in total there are 20*1000epoch=20000 batches
np.save(f'{weight_difference_folder}/weight_difference_history_hidden_layer1.npy', np.asarray(weight_difference_history['hidden_layer1']))  # (20000, 32, 64)
np.save(f'{weight_difference_folder}/weight_difference_history_hidden_layer2.npy', np.asarray(weight_difference_history['hidden_layer2']))  # (20000, 2, 32)

np.save(f'{weight_difference_folder}/activation_history_penultimate_layer_before.npy', np.asarray(activation_history['penultimate_layer_before']))  # (20000, 50, 32)
np.save(f'{weight_difference_folder}/activation_history_final_layer_before.npy', np.asarray(activation_history['final_layer_before']))  # (20000, 50, 2)
np.save(f'{weight_difference_folder}/activation_history_penultimate_layer_after.npy', np.asarray(activation_history['penultimate_layer_after']))  # (20000, 50, 32)
np.save(f'{weight_difference_folder}/activation_history_final_layer_after.npy', np.asarray(activation_history['final_layer_after']))  # (20000, 50, 2)

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


def synaptic_level():
    import random
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
    testMode = False
    directory_path = f"/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/synaptic_level/"
    os.makedirs(directory_path, exist_ok=True)
    def dataPrepare():
        # seed
        random.seed(131)
        selected_channel_penultimate_layer = random.sample(range(0, 32), 32)
        selected_channel_last_layer = random.sample(range(0, 2), 2)
        selected_channel_penultimate_layer.sort()
        selected_channel_last_layer.sort()

        # epochBatchNum = {}
        # startFromEpoch = 0
        # totalEpochNum = 1
        weight_difference_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/weight_difference_folder/"
        weight_difference_history_input_layer = np.load(f'{weight_difference_folder}/weight_difference_history_input_layer.npy')

        totalBatchNum = weight_difference_history_input_layer.shape[0]
        print(f"totalBatchNum={totalBatchNum}")
        # layerA_activations = np.zeros(
        #     (totalBatchNum, 50, len(selected_channel_penultimate_layer)))  # [#batch, batch size, #selected units]
        # layerB_activations = np.zeros(
        #     (totalBatchNum, 50, len(selected_channel_last_layer)))
        # weight_changes = np.zeros(
        #     (totalBatchNum, len(selected_channel_last_layer), len(selected_channel_penultimate_layer)))

        layerA_activations = np.load(f'{weight_difference_folder}/activation_history_penultimate_layer_before.npy')
        layerB_activations = np.load(f'{weight_difference_folder}/activation_history_final_layer_before.npy')
        weight_changes = np.load(f'{weight_difference_folder}/weight_difference_history_hidden_layer2.npy')

        # currBatchNum = 0
        # # for epoch in range(startFromEpoch, totalEpochNum):
        # for batch_i in tqdm(range(0, epochBatchNum[epoch])):
        #     # load activations and weights
        #     activation_secondLastLayer = np.load(
        #         f'{directory_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.npy')
        #     activation_lastLayer = np.load(
        #         f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.npy')
        #     weight_change = np.load(
        #         f'{directory_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.npy')  # .detach().numpy()
        #
        #     layerA_activations[currBatchNum, :, :] = activation_secondLastLayer[:,
        #                                           selected_channel_penultimate_layer]  # (128 batch#, 512)
        #     layerB_activations[currBatchNum, :, :] = activation_lastLayer[:,
        #                                           selected_channel_last_layer]  # (128 batch#, 128)
        #     weight_changes[currBatchNum, :, :] = weight_change[selected_channel_last_layer, :][:,
        #                                          selected_channel_penultimate_layer]  # (128 channel#, 512 channel#)
        #     currBatchNum += 1
        # print(f"layerA_activations.shape={layerA_activations.shape}")
        # print(f"layerB_activations.shape={layerB_activations.shape}")
        # print(f"weight_changes.shape={weight_changes.shape}")

        # obtain all the co-activation and changes.
        co_activations_flatten = []
        weight_changes_flatten = []
        pairIDs = []
        for curr_fc1_feature in tqdm(range(len(selected_channel_penultimate_layer))):  # 512*128 = 65536 pairs
            for curr_fc2_feature in range(len(selected_channel_last_layer)):
                activation_lastLayer = layerA_activations[:, :, curr_fc1_feature]  # [#batch, batch size， channel#]
                activation_secondLastLayer = layerB_activations[:, :, curr_fc2_feature]
                weight_change = weight_changes[:, curr_fc2_feature, curr_fc1_feature]
                weight_changes_flatten.append(weight_change)  # each batch has a single weight change
                # Calculate the co-activation
                co_activation = np.multiply(activation_lastLayer, activation_secondLastLayer)
                # print(f"co_activation.shape={co_activation.shape}")
                # each batch has a single weight change but multiple co-activations, average across the batch to obtain a batch specific co-activation
                co_activation = np.mean(co_activation, axis=1)
                # print(f"np.mean(co_activation, axis=1).shape={co_activation.shape}")
                co_activations_flatten.append(co_activation)
                pairIDs.append(
                    [selected_channel_penultimate_layer[curr_fc1_feature],
                     selected_channel_last_layer[curr_fc2_feature]])
        return co_activations_flatten, weight_changes_flatten, pairIDs

    co_activations_flatten_, weight_changes_flatten_, pairIDs_ = dataPrepare()
    # mkdir(f'{directory_path}/temp')
    if not os.path.exists(f'{directory_path}/temp'):
        os.mkdir(f'{directory_path}/temp')

    if not testMode:
        np.save(f'{directory_path}/temp/co_activations_flatten_.npy',
                co_activations_flatten_)  # shape = [pair#, batch#]
        np.save(f'{directory_path}/temp/weight_changes_flatten_.npy',
                weight_changes_flatten_)  # shape = [pair#, batch#]
        np.save(f'{directory_path}/temp/pairIDs_.npy',
                pairIDs_)  # shape = [pair#, [ID1, ID2]]

    # co_activations_flatten_ = np.load(f'{directory_path}/temp/co_activations_flatten_.npy',
    #                                   allow_pickle=True)  # shape = [pair#, batch#]
    # weight_changes_flatten_ = np.load(f'{directory_path}/temp/weight_changes_flatten_.npy',
    #                                   allow_pickle=True)  # shape = [pair#, batch#]
    # pairIDs_ = np.load(f'{directory_path}/temp/pairIDs_.npy',
    #                    allow_pickle=True)  # shape = [pair#, [ID1, ID2]]

    def cubic_fit_correlation_with_params(x, y, n_splits=10, random_state=42, return_subset=True):
        def cubic_function(_x, a, b, c, d):
            return a * _x ** 3 + b * _x ** 2 + c * _x + d

        # Function to compute correlation coefficient
        def compute_correlation(observed, predicted):
            return pearsonr(observed, predicted)[0]

        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Shuffle indices for k-fold cross-validation
        indices = np.arange(len(x))
        np.random.shuffle(indices)

        # Initialize arrays to store correlation coefficients and parameters
        correlation_coefficients = []
        fitted_params = []

        for curr_split in range(n_splits):
            # Split data into training and testing sets
            split_size = len(x) // n_splits
            test_indices = indices[curr_split * split_size: (curr_split + 1) * split_size]
            train_indices = np.concatenate([indices[:curr_split * split_size], indices[(curr_split + 1) * split_size:]])

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Perform constrained cubic fit on the training data
            params, _ = curve_fit(cubic_function, x_train, y_train)

            # Predict y values on the test data
            y_pred = cubic_function(x_test, *params)

            # Compute correlation coefficient and store it
            correlation_coefficient = compute_correlation(y_test, y_pred)
            correlation_coefficients.append(correlation_coefficient)

            # Store fitted parameters
            fitted_params.append(params)

        # Average correlation coefficients and parameters across folds
        mean_correlation = np.mean(correlation_coefficients)
        mean_params = np.mean(fitted_params, axis=0)

        if return_subset:
            # Randomly choose 9% of the data for future visualization
            subset_size = 10  # int(0.09 * len(x))
            subset_indices = random.sample(range(len(x)), subset_size)
            return mean_correlation, mean_params, x[subset_indices], y[subset_indices]
        else:
            return mean_correlation, mean_params

    def run_NMPH(co_activations_flatten, weight_changes_flatten, pairIDs, rows=None, cols=None, plotFig=False):
        if plotFig:
            if rows is None:
                rows = int(np.ceil(np.sqrt(len(co_activations_flatten))))
            if cols is None:
                cols = int(np.sqrt(len(co_activations_flatten)))

            fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Create a subplot matrix
            from matplotlib.cm import get_cmap
            cmap = get_cmap('viridis')  # Choose a colormap (you can change 'viridis' to your preferred one)
        else:
            axs = None
            cmap = None

        mean_correlation_coefficients = []
        # recorded_data = []  # Store recorded data for visualization
        mean_parameters = []
        x_partials = []
        y_partials = []
        for i in tqdm(range(len(co_activations_flatten))):
            if testMode:
                testBatchNum = 50
                x__ = co_activations_flatten[i][:testBatchNum]
                y__ = weight_changes_flatten[i][:testBatchNum]
                pairID = pairIDs[i]
            else:
                x__ = co_activations_flatten[i]
                y__ = weight_changes_flatten[i]
                pairID = pairIDs[i]
            mean_correlation_coefficient, mean_parameter, x_partial, y_partial = cubic_fit_correlation_with_params(
                x__, y__,
                n_splits=10,
                random_state=42,
                return_subset=True
            )
            mean_correlation_coefficients.append(mean_correlation_coefficient)
            mean_parameters.append(mean_parameter)
            x_partials.append(x_partial)
            y_partials.append(y_partial)

            if plotFig:
                row = i // cols
                col = i % cols

                ax = axs[row, col]  # Select the appropriate subplot

                # Color the dots based on a sequence
                sequence = np.linspace(0, 1, len(x__))  # Create a sequence of values from 0 to 1
                colors = cmap(sequence)  # Map the sequence to colors using the chosen colormap

                ax.scatter(x__, y__, s=10, c=colors)  # 's' controls the size of the points, 'c' sets the colors

                # Add labels and a title to each subplot
                ax.set_title(f'pairID: {pairID}')

                # Hide x and y-axis ticks and tick labels
                ax.set_xticks([])
                ax.set_yticks([])

        if plotFig:
            plt.tight_layout()  # Adjust subplot layout for better visualization
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        mean_correlation_coefficients = np.array(mean_correlation_coefficients)
        p_value = np.nanmean(mean_correlation_coefficients < 0)
        print(f"p value = {p_value}")

        # Return mean_correlation_coefficients along with recorded_data
        return mean_correlation_coefficients, np.array(mean_parameters), np.array(x_partials), np.array(y_partials)

    if testMode:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten_[:4], weight_changes_flatten_[:4], pairIDs_[:4])
    else:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten_, weight_changes_flatten_, pairIDs_)

    if not testMode:
        np.save(f'{directory_path}/temp/mean_correlation_coefficients_.npy', mean_correlation_coefficients_)
        np.save(f'{directory_path}/temp/mean_parameters_.npy', mean_parameters_)
        np.save(f'{directory_path}/temp/x_partials_.npy', x_partials_)
        np.save(f'{directory_path}/temp/y_partials_.npy', y_partials_)

    x_partials_ = x_partials_.flatten()
    y_partials_ = y_partials_.flatten()
    mean_parameters_avg = np.mean(mean_parameters_, axis=0)

    def plot_scatter_and_cubic(x_partials, y_partials, mean_parameters):
        def cubic_function(_x, a, b, c, d):
            print(f"a={a}, b={b}, c={c}, d={d}")
            return a * _x ** 3 + b * _x ** 2 + c * _x + d

        # Scatter plot
        plt.scatter(x_partials, y_partials, label='Data Points', color='green', marker='o', s=30)

        # Fit cubic curve using curve_fit
        # popt, _ = curve_fit(cubic_function, x_partials_, y_partials_)

        # Generate points for the fitted cubic curve
        x_fit = np.linspace(min(x_partials), max(x_partials), 100)
        y_fit = cubic_function(x_fit, *mean_parameters)

        # Plot the fitted cubic curve
        plt.plot(x_fit, y_fit, label='Fitted Cubic Curve', color='red')

        # Add labels and a legend
        plt.xlabel('X Partials')
        plt.ylabel('Y Partials')
        plt.legend()

        # Show the plot
        plt.show()

    plot_scatter_and_cubic(x_partials_, y_partials_, mean_parameters_avg)


def representational_level():
    pass

