from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
# import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
testMode = True


def dataPrepare():
    fc1_activations = []
    fc2_activations = []
    fc2_partial_weights = []
    weight_changes = []

    for epoch in range(0, 1):
        for batch_i in range(0, 100):
            directory_path = '/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/weights_difference/numpy/'
            # torch.save(weights_difference,
            #            f'{weights_difference_folder}/weights_difference_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_lastLayer,
            #            f'{weights_difference_folder}/activation_lastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_secondLastLayer,
            #            f'{weights_difference_folder}/activation_secondLastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')

            # load activations and weights
            activation_lastLayer = np.load(
                f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')
            activation_secondLastLayer = np.load(
                f'{directory_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')
            weight_change = np.load(
                f'{directory_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')  # .detach().numpy()

            fc1_activations.append(activation_secondLastLayer[:, :10])  # (128 batch#, 512)
            fc2_activations.append(activation_lastLayer[:, :5])  # (128 batch#, 128)
            # fc2_partial_weights.append(fc2_partial_weight)
            weight_changes.append(weight_change[:5, :][:, :10]) # (128 channel#, 512 channel#)
    fc1_activations = np.asarray(fc1_activations)
    fc2_activations = np.asarray(fc2_activations)
    # fc2_partial_weights = np.asarray(fc2_partial_weights)
    weight_changes = np.asarray(weight_changes)
    print(f"fc1_activations.shape={fc1_activations.shape}")
    print(f"fc2_activations.shape={fc2_activations.shape}")
    # print(f"fc2_partial_weights.shape={fc2_partial_weights.shape}")
    print(f"weight_changes.shape={weight_changes.shape}")
    # fc1_activations.shape=(16, 64, 10)  # (#batch, batch size, #selected units)
    # fc2_activations.shape=(16, 64, 5)
    # fc2_partial_weights.shape=(16, 5, 10)
    # weight_changes.shape=(16, 5, 10)

    # Calculate the row-wise differences
    # fc2_partial_weights_differences = fc2_partial_weights[1:] - fc2_partial_weights[:-1]

    # obtain all the co-activation and changes.
    co_activations_flatten = []
    weight_changes_flatten = []

    for fc1_feature in range(10):
        for fc2_feature in range(5):
            activation_lastLayer = fc1_activations[:, :, fc1_feature]
            activation_secondLastLayer = fc2_activations[:, :, fc2_feature]
            weight_change = weight_changes[:, fc2_feature, fc1_feature]
            weight_changes_flatten.append(weight_change)
            # Calculate the co-activation
            co_activation = np.multiply(activation_lastLayer, activation_secondLastLayer)
            print(f"co_activation.shape={co_activation.shape}")
            co_activation = np.mean(co_activation, axis=1)
            print(f"np.mean(co_activation, axis=1).shape={co_activation.shape}")
            co_activations_flatten.append(co_activation)

            # Calculate the co-activation change
            # co_activation_change = np.multiply(fc1_activation, weight_change)
            # Calculate the co-activation change difference
            # co_activation_change_difference = np.multiply(fc1_activation, fc2_partial_weight_difference)

            # Save the co-activation and co-activation change
            # directory_path = '/content/features_weights'
    return co_activations_flatten, weight_changes_flatten


co_activations_flatten_, weight_changes_flatten_ = dataPrepare()


def run_NMPH(co_activations_flatten, weight_changes_flatten):
    # plot co_activations_flatten_ as x axis and weight_changes_flatten_ as y axis as scatter plot
    import matplotlib.pyplot as plt
    # Create a scatter plot
    plt.scatter(co_activations_flatten, weight_changes_flatten, s=10)  # 's' controls the size of the points

    # Add labels and a title
    plt.xlabel('Co-Activations')
    plt.ylabel('Weight Changes')
    plt.title('Scatter Plot of Co-Activations vs. Weight Changes')

    # Show the plot
    plt.show()


run_NMPH(co_activations_flatten_, weight_changes_flatten_)