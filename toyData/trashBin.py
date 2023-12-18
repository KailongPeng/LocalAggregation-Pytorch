def test_single_dotsNeighbotSingleBatch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

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
    def local_aggregation_loss(random_point, close_neighbors, background_neighbors):
        # Compute pairwise distances between random_point and close neighbors
        close_distances = torch.norm(random_point - close_neighbors, dim=1)

        # Compute pairwise distances between random_point and background neighbors
        background_distances = torch.norm(random_point - background_neighbors, dim=1)

        # Compute loss based on distances
        loss = torch.mean(torch.log(1 + torch.exp(close_distances - background_distances)))
        return loss

    # Define close and background neighbors for a single random point
    def get_neighbors_single_point(embeddings, c=None, b=None):
        # Choose a random index within the batch
        random_index = torch.randint(0, embeddings.size(0), (1,))
        random_point = embeddings[random_index]

        # Compute pairwise distances between embeddings and the random point
        distances = torch.cdist(embeddings, random_point.unsqueeze(0))

        # Get indices of k closest neighbors for the random point
        _, indices = torch.topk(distances.view(-1), c + b + 1, largest=False)
        # Remove self from list of neighbors
        indices = indices[indices != random_index.item()]

        # Get embeddings of close neighbors for the random point
        close_neighbors = embeddings[indices[:c]]
        # Get embeddings of background neighbors for the random point
        background_neighbors = embeddings[indices[c:]]

        return close_neighbors, background_neighbors, random_point, random_index, indices[:c], indices[c:]

    # Define toy dataset shaped [1000, 2]
    # data = torch.tensor(np.random.rand(1000, 2), dtype=torch.float32)
    # dataset = ToyDataset(data)
    # dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    # Define toy dataset shaped [1000, 2]
    threeD_input = False
    batch_size = 50
    if threeD_input:
        points_data, labels_data = generate_3d_scatter_plot(separator=1 / 2,
                                                            display_plot=True)  # separator should be 1/3 or 1/2
    else:
        points_data, labels_data = generate_2d_scatter_plot(display_plot=True)

    # Split the data into training and testing sets (1000 points each)
    train_data, test_data = points_data[:1000], points_data[1000:]
    train_labels, test_labels = labels_data[:1000], labels_data[1000:]

    dataset = ToyDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define neural network and optimizer
    # net = Net()
    net = SimpleFeedforwardNN()
    learning_rate = 0.05
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    testMode = False
    if testMode:
        plot_neighborhood = True
    else:
        plot_neighborhood = False

    # Train network using local aggregation loss
    loss_values = []  # List to store loss values for each epoch
    total_epochs = 100

    initial_v_points = []
    final_v_points = []

    initial_v_labels = []
    final_v_labels = []

    for epoch in range(total_epochs):
        epoch_loss = 0.0  # Variable to accumulate loss within each epoch

        for batch, batch_labels in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            embeddings = net(batch.float())
            # record initial and final latent space points
            if epoch == 0:
                initial_v_points.append(embeddings)
                initial_v_labels.append(batch_labels)
            # Get close and background neighbors for a single random point
            c = 2
            b = 2
            [close_neighbors, background_neighbors, random_point,
             random_index, close_neighbors_indices, background_neighbors_indices] = get_neighbors_single_point(
                embeddings, c=c, b=b)
            if plot_neighborhood:
                # plot in latent space, closest_c_indices as blue and closest_b_indices as black and the chosen vi as red
                latent_points = embeddings.detach().numpy()
                fig = plt.figure(figsize=(20, 20))
                ax = fig.add_subplot(111)
                ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                           label='Other Points', alpha=0.2)
                ax.scatter(latent_points[close_neighbors_indices, 0], latent_points[close_neighbors_indices, 1],
                           c='blue', marker='o', label='Closest C Points')
                ax.scatter(latent_points[background_neighbors_indices, 0],
                           latent_points[background_neighbors_indices, 1],
                           c='black', marker='o', label='Closest B Points')
                ax.scatter(latent_points[random_index, 0], latent_points[random_index, 1], c='red', marker='*',
                           s=200, label='Chosen vi')
            else:
                ax = None

            # Compute loss
            loss = local_aggregation_loss(random_point, close_neighbors, background_neighbors)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

            embeddings = net(batch.float())

            # record initial and final latent space points
            if epoch == total_epochs - 1:
                final_v_points.append(embeddings)
                final_v_labels.append(batch_labels)

            if plot_neighborhood:
                # plot in latent space, closest_c_indices as blue and closest_b_indices as black and the chosen vi as red
                latent_points = embeddings.detach().numpy()

                random_point = latent_points[random_index]
                close_neighbors = latent_points[close_neighbors_indices]
                background_neighbors = latent_points[background_neighbors_indices]

                ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                           label='Other Points', alpha=0.1)
                ax.scatter(close_neighbors[:, 0], close_neighbors[:, 1],
                           c='green', marker='o', label='Closest C Points', alpha=0.5)
                ax.scatter(background_neighbors[:, 0], background_neighbors[:, 1],
                           c='purple', marker='o', label='Closest B Points', alpha=0.5)
                ax.scatter(random_point[0], random_point[1], c='black', marker='*',
                           s=200, label='Chosen vi', alpha=0.5)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_title(f'Epoch {epoch} after training loss={epoch_loss}')
                ax.legend()
                plt.show()

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
    initial_v_labels = torch.cat(initial_v_labels, dim=0).detach().numpy()
    scatter_initial = axes[0].scatter(initial_v_points[:, 0], initial_v_points[:, 1], c=initial_v_labels,
                                      cmap='rainbow', marker='o')
    axes[0].set_title('Initial Latent Space Points')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    fig.colorbar(scatter_initial, ax=axes[0], label='Point Index')

    # Final latent space points
    final_v_points = torch.cat(final_v_points, dim=0).detach().numpy()
    final_v_labels = torch.cat(final_v_labels, dim=0).detach().numpy()
    scatter_final = axes[1].scatter(final_v_points[:, 0], final_v_points[:, 1], c=final_v_labels,
                                    cmap='rainbow', marker='o')
    axes[1].set_title('Final Latent Space Points')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    fig.colorbar(scatter_final, ax=axes[1], label='Point Index')

    plt.tight_layout()
    plt.show()


# test_single_dotsNeighbotSingleBatch()  # this works as long as the number of close neighbors is small ; later analysis is not based on the result of this function but on the result of test_multiple_dotsNeighbotSingleBatch()
# this function is integrated into test_multiple_dotsNeighbotSingleBatch() when batch_size=1


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

"""
Instead of simple push-pull binary decision in the local aggregation loss definition, utilizing the complete U shaped curve would likely improve the result.

The link between synaptic and representational level of NMPH can be easily imagined when the representation is
extremely sparsely coded, here the two levels are mostly equivalent, but when the representation is dense, the
connection between the two levels is not that obvious.

somehow the growth of the area of the representation should be the net effect of learning as indicated by Haig's lecture. 
This growth of the area of the representation is the result of integration or differentiation, this can be tested out by 
changing the number/power(push-pull force) of close neighbors and background neighbors.
It turns out that increasing the relative pulling force for close neighbors only leads to representation collapse.

I found out that even the supervised learning is not resulting in a clustered representation, which means that the 
representation learned by the local aggregation loss is good for now. They key is to reveal the original structure of
the data, which is not necessarily clustered. However, in real cases, cats and dogs are clustered, so the representation
should be clustered then.

What is the correct analogy of BCM theory in this local aggregation loss? The push-pull force or the number of neighbors?
Read "The BCM theory of synapse modification at 30: interaction of theory with experiment" to figure out.

Next step is to implement the test_single_dotsNeighbotSingleBatch() function to test simpler effect.
    design the output of test_single_dotsNeighbotSingleBatch should be
        for each batch:
            layer activation (A layer = penultimate layer, B layer = last layer)
                A layer ; before training ; center points                
                A layer ; before training ; close neighbors
                A layer ; before training ; background neighbors
                B layer ; before training ; center points
                B layer ; before training ; close neighbors
                B layer ; before training ; background neighbors
                
                A layer ; after training ; center points
                A layer ; after training ; close neighbors
                A layer ; after training ; background neighbors
                B layer ; after training ; center points                
                B layer ; after training ; close neighbors
                B layer ; after training ; background neighbors
            weight change  
        
    design the input of NMPH_synaptic should be
        A layer ; before training ; center points
        A layer ; before training ; close neighbors
        A layer ; before training ; background neighbors
        B layer ; before training ; center points
        B layer ; before training ; close neighbors
        B layer ; before training ; background neighbors
        weight change
        
    design the input of NMPH_representational should be
        A layer ; before training ; center points
        A layer ; before training ; close neighbors
        A layer ; before training ; background neighbors
        B layer ; before training ; center points        
        B layer ; before training ; close neighbors
        B layer ; before training ; background neighbors
        
        A layer ; after training ; center points
        A layer ; after training ; close neighbors
        A layer ; after training ; background neighbors
        B layer ; after training ; center points
        B layer ; after training ; close neighbors
        B layer ; after training ; background neighbors
        

The relative distance is always scalling up since manifold expansion is always happening right now.
    Would weight regularization help? Force weight to be small.
        weight regularization help, but it is hard to control the degree.
    invent a new loss to control the range explicitly. This works perfectly.
        range_loss
    Batch Norm ? Layer norm
        Neithor of them helps.
    
    Would increasing power of the pull force help?
    Would increasing the number of close neighbors help?
    
    
Next step is to look into the behavior of the cross entropy loss synaptic and representational level NMPH result.

result: (c, b) # c: number of close neighbors, b: number of background neighbors 
    (0, 1) gets normal result,
    (1, 1) gets normal result,
    (2, 2) gets normal result,
    (5, 5) gets normal result,
    (10, 10) gets normal result,
    (15, 15) gets collapsed result, 
    (20, 20) gets collapsed result
    
    (1, 20) gets normal result,        
    (20, 1) gets collapsed result,
    
    conclusion:
    1. the number of close neighbors and background neighbors should be small, otherwise the result will be collapsed.
    2. whether the result collapses or not most likely depends on the number of close neighbors, not the number of 
        background neighbors.
    3. intuitively, this is determined by whether probabilistically speaking, the close neighbors of two neighboring 
        center points overlap or not. If overlap, then the result will be collapsed.
    4. it turns out that the close neighbors are not important, the background neighbors are important. This means that 
        integration is not important while differentiation is important for representation learning.
    5. one way to boost integration might be to increase the weight of close neighbor pulling force to enforce the 
        formation of clusters.


trash bin:
    add another loss so that the latent space (aka v=model(x)) is encouraged to span 0-1.
    
    layer norm versus batch norm
    
    maybe try different close and background neighbors number would lead to different results because
    this simulates the BCM shifting threshold effect and also the NMPH curve scaling effect.
    I found that different number of close or background neighbors makes the model collapse or not. Increasing b and c makes it easier to collapse. I don't know why.
"""


# def prepare_data_for_NMPH(
#         curr_batch_=None,
#         layer_activations_before=None,
#         layer_activations_after=None,
#         repChange_distanceType_=None,
#         coactivation_distanceType_=None):
#     layer_activations_before = layer_activations_before[curr_batch_, :, :]  # (50, 2)
#     layer_activations_after = layer_activations_after[curr_batch_, :, :]  # (50, 2)
#
#     batchSize = layer_activations_before.shape[0]
#
#     # for each pair of images, calculate the cosine similarity of the activations before weight change
#     pairImg_similarity_before_repChange = np.zeros((batchSize, batchSize))
#     pairImg_similarity_before_coactivation = np.zeros((batchSize, batchSize))
#     for i in range(batchSize):
#         for j in range(batchSize):
#             if repChange_distanceType_ == 'cosine':
#                 pairImg_similarity_before_repChange[i, j] = np.dot(layer_activations_before[i, :],
#                                                                    layer_activations_before[j, :]) / (
#                                                                     np.linalg.norm(layer_activations_before[i,
#                                                                                    :]) * np.linalg.norm(
#                                                                 layer_activations_before[j, :]))
#             elif repChange_distanceType_ == 'dot':
#                 pairImg_similarity_before_repChange[i, j] = np.dot(layer_activations_before[i, :],
#                                                                    layer_activations_before[j, :])
#             elif repChange_distanceType_ == 'correlation':
#                 pairImg_similarity_before_repChange[i, j] = \
#                     pearsonr(layer_activations_before[i, :], layer_activations_before[j, :])[0]
#             elif repChange_distanceType_ == 'L1':
#                 pairImg_similarity_before_repChange[i, j] = - np.linalg.norm(
#                     layer_activations_before[i, :] - layer_activations_before[j, :],
#                     ord=1)
#             elif repChange_distanceType_ == 'L2':  # euclidean
#                 pairImg_similarity_before_repChange[i, j] = - np.linalg.norm(
#                     layer_activations_before[i, :] - layer_activations_before[j, :],
#                     ord=2)
#             elif repChange_distanceType_ == 'jacard':
#                 pairImg_similarity_before_repChange[i, j] = calculate_jaccard_similarity(
#                     layer_activations_before[i, :], layer_activations_before[j, :])
#             else:
#                 raise Exception("distanceType not found")
#
#             if coactivation_distanceType_ == 'cosine':
#                 pairImg_similarity_before_coactivation[i, j] = np.dot(layer_activations_before[i, :],
#                                                                       layer_activations_before[j, :]) / (
#                                                                        np.linalg.norm(layer_activations_before[i,
#                                                                                       :]) * np.linalg.norm(
#                                                                    layer_activations_before[j, :]))
#             elif coactivation_distanceType_ == 'dot':
#                 pairImg_similarity_before_coactivation[i, j] = np.dot(layer_activations_before[i, :],
#                                                                       layer_activations_before[j, :])
#             elif coactivation_distanceType_ == 'L1':
#                 pairImg_similarity_before_coactivation[i, j] = - np.linalg.norm(
#                     layer_activations_before[i, :] - layer_activations_before[j, :],
#                     ord=1)
#             elif coactivation_distanceType_ == 'L2':  # euclidean
#                 pairImg_similarity_before_coactivation[i, j] = - np.linalg.norm(
#                     layer_activations_before[i, :] - layer_activations_before[j, :],
#                     ord=2)
#             elif coactivation_distanceType_ == 'jacard':
#                 pairImg_similarity_before_coactivation[i, j] = calculate_jaccard_similarity(
#                     layer_activations_before[i, :], layer_activations_before[j, :])
#             elif coactivation_distanceType_ == 'correlation':
#                 pairImg_similarity_before_coactivation[i, j] = \
#                     pearsonr(layer_activations_before[i, :], layer_activations_before[j, :])[0]
#             else:
#                 raise Exception("distanceType not found")
#
#     # for each pair of images, calculate the cosine similarity of the activations after weight change
#     pairImg_similarity_after_repChange = np.zeros((batchSize, batchSize))
#     pairImg_similarity_after_coactivation = np.zeros((batchSize, batchSize))
#     for i in range(batchSize):
#         for j in range(batchSize):
#             if repChange_distanceType_ == 'cosine':
#                 pairImg_similarity_after_repChange[i, j] = np.dot(layer_activations_after[i, :],
#                                                                   layer_activations_after[j, :]) / (
#                                                                    np.linalg.norm(layer_activations_after[i,
#                                                                                   :]) * np.linalg.norm(
#                                                                layer_activations_after[j, :]))
#             elif repChange_distanceType_ == 'dot':
#                 pairImg_similarity_after_repChange[i, j] = np.dot(layer_activations_after[i, :],
#                                                                   layer_activations_after[j, :])
#             elif repChange_distanceType_ == 'correlation':
#                 pairImg_similarity_after_repChange[i, j] = \
#                     pearsonr(layer_activations_after[i, :], layer_activations_after[j, :])[0]
#             elif repChange_distanceType_ == 'L1':
#                 pairImg_similarity_after_repChange[i, j] = - np.linalg.norm(
#                     layer_activations_after[i, :] - layer_activations_after[j, :],
#                     ord=1)
#             elif repChange_distanceType_ == 'L2':  # euclidean
#                 pairImg_similarity_after_repChange[i, j] = - np.linalg.norm(
#                     layer_activations_after[i, :] - layer_activations_after[j, :],
#                     ord=2)
#             elif repChange_distanceType_ == 'jacard':
#                 pairImg_similarity_after_repChange[i, j] = calculate_jaccard_similarity(
#                     layer_activations_after[i, :], layer_activations_after[j, :])
#             else:
#                 raise Exception("distanceType not found")
#
#             if coactivation_distanceType_ == 'cosine':
#                 pairImg_similarity_after_coactivation[i, j] = np.dot(layer_activations_after[i, :],
#                                                                      layer_activations_after[j, :]) / (
#                                                                       np.linalg.norm(layer_activations_after[i,
#                                                                                      :]) * np.linalg.norm(
#                                                                   layer_activations_after[j, :]))
#             elif coactivation_distanceType_ == 'dot':
#                 pairImg_similarity_after_coactivation[i, j] = np.dot(layer_activations_after[i, :],
#                                                                      layer_activations_after[j, :])
#             elif coactivation_distanceType_ == 'L1':
#                 pairImg_similarity_after_coactivation[i, j] = - np.linalg.norm(
#                     layer_activations_after[i, :] - layer_activations_after[j, :],
#                     ord=1)
#             elif coactivation_distanceType_ == 'L2':  # euclidean
#                 pairImg_similarity_after_coactivation[i, j] = - np.linalg.norm(
#                     layer_activations_after[i, :] - layer_activations_after[j, :],
#                     ord=2)
#             elif coactivation_distanceType_ == 'jacard':
#                 pairImg_similarity_after_coactivation[i, j] = calculate_jaccard_similarity(
#                     layer_activations_after[i, :], layer_activations_after[j, :])
#             elif coactivation_distanceType_ == 'correlation':
#                 pairImg_similarity_after_coactivation[i, j] = \
#                     pearsonr(layer_activations_after[i, :], layer_activations_after[j, :])[0]
#             else:
#                 raise Exception("distanceType not found")
#
#     # for each pair of images, calculate distance between the activations before and after weight change
#     representationalChange = pairImg_similarity_after_repChange - pairImg_similarity_before_repChange
#
#     # prepare the data for NMPH
#     if co_activationType == 'before':
#         co_activations_flatten = pairImg_similarity_before_coactivation.reshape(-1)
#     elif co_activationType == 'after':
#         co_activations_flatten = pairImg_similarity_after_coactivation.reshape(-1)
#     else:
#         raise Exception("co_activationType not found")
#     representationChange_flatten = representationalChange.reshape(-1)
#     return co_activations_flatten, representationChange_flatten
#

# co_activations_flatten_, representationChange_flatten_ = prepare_data_for_NMPH(
#     curr_batch_=curr_batch,
#     layer_activations_before=layer_b_activations_before,
#     layer_activations_after=layer_b_activations_after,
#     repChange_distanceType_=repChange_distanceType,
#     coactivation_distanceType_=coactivation_distanceType
# )
