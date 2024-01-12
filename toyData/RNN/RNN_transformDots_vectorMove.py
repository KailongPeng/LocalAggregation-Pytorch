import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


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

# this random seed is preventing the model from training stably, thus this seed is removed.
# set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# Generate random 2D points

close_count = 10  # 20
background_count = 10 # 40
num_points = 1 + close_count + background_count  # now I removed irrelevent dots from the loss so that it is easier to learn for the network
num_epochs = 60

hidden_dim = 20
num_timepoints = 10

points = np.random.rand(num_points, 2)

# Calculate the centroid (center) of all points
centroid = np.mean(points, axis=0)

# Find the index of the point closest to the centroid
center_index = np.argmin(np.linalg.norm(points - centroid, axis=1))  # Specify any point index as the center point

print("Index of the center point:", center_index)

# Move neighbors using the function
center, close_neighbors, background_neighbors, irrelevant_neighbors, close_neighbors_moved, background_neighbors_moved = move_neighbors(
    points, center_index, close_count=close_count, background_count=background_count, move_factor=0.3)


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

def plot_points_with_colors(points, title, seed=123):
    np.random.seed(seed)  # Set random seed for reproducibility
    # colors = [plt.cm.rainbow(i / len(points)) for i in range(len(points))]
    colors = np.random.rand(len(points), 3)  # Generate random RGB colors for each point

    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Display set1 with random rainbow colors
plot_points_with_colors(set1, 'Set 1')

# Display set2 with random rainbow colors
plot_points_with_colors(set2, 'Set 2')


def transform_points(net, points):
    with torch.no_grad():
        input_tensor = torch.tensor(points, dtype=torch.float32)
        output_tensor = net(input_tensor)
    return output_tensor.numpy()

# Function to plot the loss curve
def plot_loss_curve(loss_values, title='Training Loss Curve'):
    plt.plot(loss_values, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Define the Vanilla RNN model
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN forward pass
        out, _ = self.rnn(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out

# Instantiate the model
input_size = 2  # Size of input features
hidden_size = hidden_dim  # Size of hidden state
output_size = 2  # Size of output
model = VanillaRNN(input_size, hidden_size, output_size)

def prepare_data_rnn(dataset, sequence_length=None):
    """
    Prepares the data for RNN input.

    Parameters:
    - dataset: numpy array with shape (N, 2) where N is the number of points and 2 is the number of coordinates.
    - sequence_length: Desired sequence length for the RNN input.

    Returns:
    - rnn_input: numpy array with shape (N, sequence_length, 2).
    """
    num_points, num_coordinates = dataset.shape

    # Repeat each point sequence_length times
    rnn_input = np.tile(dataset[:, np.newaxis, :], (1, sequence_length, 1))

    return torch.tensor(rnn_input, dtype=torch.float32)

set1_rnn_input = prepare_data_rnn(set1, sequence_length=num_timepoints)

def check_repeat(set1_rnn_input):
    # Number of curves (i values)
    num_curves = set1_rnn_input.shape[0]

    # Plot each curve
    plt.figure()
    for i in range(num_curves):
        plt.plot(set1_rnn_input[i, :, 0], label=f'Curve {i + 1}')

    # Add labels and legend

    plt.xlabel('Sequence Index')
    plt.ylabel('Coordinate 1 Values')
    plt.title('Plot of set1_rnn_input[i, :, 0]')
    # plt.legend()
    plt.show()

check_repeat(set1_rnn_input)

# Define loss function and optimizer
criterion = nn.MSELoss()
initial_learning_rate = 0.05
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.001)  # Adding weight decay

# Convert sets to PyTorch tensors
set1_tensor = torch.tensor(set1, dtype=torch.float32)
set2_tensor = torch.tensor(set2, dtype=torch.float32)

# Display set1 to initial untrained
unTrainedTransform = transform_points(model, set1_rnn_input)
plot_points_with_colors(unTrainedTransform, 'set1 to initial untrained')

# Training loop for untrained network
loss_values_untrained = []
for epoch in tqdm(range(num_epochs)):
    if epoch == int(num_epochs / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/2.0
    elif epoch == int(num_epochs * 2 / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/4.0
    outputs = model(set1_rnn_input)
    loss = criterion(outputs, set1_tensor)
    # loss = criterion(outputs, set2_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values_untrained.append(loss.item())


# Display set1 to set1
transform_set1_to_set1 = transform_points(model, set1_rnn_input)
plot_points_with_colors(transform_set1_to_set1, 'set1 to set1')
plot_loss_curve(loss_values_untrained, title='set1 to set1: Loss Curve')

distance_between_center_close__set1 = calculate_distances(
    np.vstack((transform_set1_to_set1[0],
               transform_set1_to_set1[1:1 + close_count]
               )))[0]
distance_between_center_background__set1 = calculate_distances(
    np.vstack((transform_set1_to_set1[0],
               transform_set1_to_set1[1 + close_count:1 + close_count + background_count]
               )))[0]
distance_between_center_irr__set1 = calculate_distances(
    np.vstack((transform_set1_to_set1[0],
               transform_set1_to_set1[1 + close_count + background_count:]
               )))[0]


optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.001)  # Adding weight decay
loss_values_trained = []
for epoch in tqdm(range(num_epochs)):
    if epoch == int(num_epochs / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/2.0
    elif epoch == int(num_epochs * 2 / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/4.0
    outputs = model(set1_rnn_input)
    loss = criterion(outputs, set2_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values_trained.append(loss.item())

# # Display set1 to set2
trained_transformed_set1 = transform_points(model, set1_rnn_input)
plot_points_with_colors(trained_transformed_set1, 'set1 to set2')
plot_loss_curve(loss_values_trained, title='set1 to set2: Loss Curve')

distance_between_center_close__set2 = calculate_distances(
    np.vstack((trained_transformed_set1[0],
                trained_transformed_set1[1:1+close_count]
                )))[0]
distance_between_center_background__set2 = calculate_distances(
    np.vstack((trained_transformed_set1[0],
                trained_transformed_set1[1+close_count:1+close_count+background_count]
                )))[0]
distance_between_center_irr__set2 = calculate_distances(
    np.vstack((trained_transformed_set1[0],
                trained_transformed_set1[1+close_count+background_count:]
                )))[0]
# plot bar plot for distance_between_center_close__set2 - distance_between_center_close__set1
# plt.figure()
# plt.hist(distance_between_center_close__set2 - distance_between_center_close__set1, bins=20)
# plt.title('distance_between_center_close__set2 - distance_between_center_close__set1')
#
# plt.figure()
# plt.hist(distance_between_center_background__set2 - distance_between_center_background__set1, bins=20)
# plt.title('distance_between_center_background__set2 - distance_between_center_background__set1')
#
# plt.figure()
# plt.hist(distance_between_center_irr__set2 - distance_between_center_irr__set1, bins=20)
# plt.title('distance_between_center_irr__set2 - distance_between_center_irr__set1')

def bar(means=None, upper=None, lower=None, ROINames=None, title=None, xLabel="", yLabel="", fontsize=50,
        setBackgroundColor=False,
        savePath=None, showFigure=True):
    import matplotlib.pyplot as plt
    # plot barplot with percentage error bar
    if type(means) == list:
        means = np.asarray(means)
    if type(upper) == list:
        upper = np.asarray(upper)
    if type(means) == list:
        lower = np.asarray(lower)

    # plt.figure(figsize=(fontsize, fontsize/2), dpi=70)
    positions = list(np.arange(len(means)))

    fig, ax = plt.subplots(figsize=(fontsize/2, fontsize/2))
    ax.bar(positions, means, yerr=[means - lower, upper - means], align='center', alpha=0.5, ecolor='black',
           capsize=10)
    if setBackgroundColor:
        ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.set_xlabel(xLabel, fontsize=fontsize)
    ax.set_xticks(positions)
    ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=fontsize)

    if ROINames is not None:
        xtick = ROINames
        ax.set_xticklabels(xtick, fontsize=fontsize, rotation=45, ha='right')
    ax.set_title(title, fontsize=fontsize)
    ax.yaxis.grid(True)
    _ = plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath)
    if showFigure:
        _ = plt.show()
    else:
        _ = plt.close()


# Example usage
array1 = distance_between_center_close__set2 - distance_between_center_close__set1
array2 = distance_between_center_background__set2 - distance_between_center_background__set1

# change to integration score
array1 = - array1
array2 = - array2
def cal_resample(data=None, times=5000, return_all=False):
    # 这个函数的目的是为了针对输入的数据，进行有重复的抽取5000次，然后记录每一次的均值，最后输出这5000次重采样的均值分布    的   均值和5%和95%的数值。
    if data is None:
        raise Exception
    if type(data) == list:
        data = np.asarray(data)
    iter_mean = []
    for _ in range(times):
        iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
        iter_mean.append(np.nanmean(iter_distri))
    _mean = np.mean(iter_mean)
    _5 = np.percentile(iter_mean, 5)
    _95 = np.percentile(iter_mean, 95)
    if return_all:
        return _mean, _5, _95, iter_mean
    else:
        return _mean, _5, _95

mean1, _5_1, _95_1 = cal_resample(data=array1, times=500, return_all=False)
mean2, _5_2, _95_2 = cal_resample(data=array2, times=500, return_all=False)
means = np.array([mean1, mean2])
upper = np.array([_95_1, _95_2])
lower = np.array([_5_1, _5_2])
bar(
    means=means,
    upper=upper,
    lower=lower,
    title="integration score",
    ROINames=["close", "background"],
    )

