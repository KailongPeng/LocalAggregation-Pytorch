from __future__ import print_function
import os
import random
from glob import glob
import numpy as np
from tqdm import tqdm
testMode = True


def dataPrepare():
    # selected_channel_penultimate_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 selected units
    # randomly select 10 units from the penultimate layer
    # seed
    random.seed(131)
    selected_channel_penultimate_layer = random.sample(range(0, 512), 5)
    selected_channel_last_layer = random.sample(range(0, 128), 5)

    totalBatchNum = 0
    epochBatchNum = {}
    startFromEpoch = 0
    totalEpochNum = 1

    directory_path = '/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/weights_difference/numpy/'
    for epoch in range(startFromEpoch, totalEpochNum):
        files = glob(f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i*.pth.tar.npy')
        if testMode:
            files = files[:50]
            epochBatchNum[epoch] = len(files)
            totalBatchNum += len(files)
        else:
            epochBatchNum[epoch] = len(files)-1
            totalBatchNum += len(files)-1  # minus 1 since the final batch usually has a different size, e.g. for batch size=128, the final batch only has 15 images

    fc1_activations = np.zeros((totalBatchNum, 128, len(selected_channel_penultimate_layer)))  # [#batch, batch size, #selected units]
    fc2_activations = np.zeros((totalBatchNum, 128, len(selected_channel_last_layer)))
    weight_changes = np.zeros((totalBatchNum, len(selected_channel_last_layer), len(selected_channel_penultimate_layer)))

    currBatchNum = 0
    for epoch in range(startFromEpoch, totalEpochNum):
        for batch_i in tqdm(range(0, epochBatchNum[epoch])):
            # torch.save(weights_difference,
            #            f'{weights_difference_folder}/weights_difference_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_lastLayer,
            #            f'{weights_difference_folder}/activation_lastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_secondLastLayer,
            #            f'{weights_difference_folder}/activation_secondLastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # load activations and weights
            activation_secondLastLayer = np.load(
                f'{directory_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')
            activation_lastLayer = np.load(
                f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')
            weight_change = np.load(
                f'{directory_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.pth.tar.npy')  # .detach().numpy()

            fc1_activations[currBatchNum, :, :] = activation_secondLastLayer[:, selected_channel_penultimate_layer]  # (128 batch#, 512)
            fc2_activations[currBatchNum, :, :] = activation_lastLayer[:, selected_channel_last_layer]  # (128 batch#, 128)
            weight_changes[currBatchNum, :, :] = weight_change[selected_channel_last_layer, :][:, selected_channel_penultimate_layer] # (128 channel#, 512 channel#)
            currBatchNum += 1
    print(f"fc1_activations.shape={fc1_activations.shape}")
    print(f"fc2_activations.shape={fc2_activations.shape}")
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
    pairIDs = []
    for curr_fc1_feature in range(len(selected_channel_penultimate_layer)):  # 512*128 = 65536 pairs
        for curr_fc2_feature in range(len(selected_channel_last_layer)):
            activation_lastLayer = fc1_activations[:, :, curr_fc1_feature]  # [#batch, batch size， channel#]
            activation_secondLastLayer = fc2_activations[:, :, curr_fc2_feature]
            weight_change = weight_changes[:, curr_fc2_feature, curr_fc1_feature]
            weight_changes_flatten.append(weight_change)  # each batch has a single weight change
            # Calculate the co-activation
            co_activation = np.multiply(activation_lastLayer, activation_secondLastLayer)
            # print(f"co_activation.shape={co_activation.shape}")
            # each batch has a single weight change but multiple co-activations, average across the batch to obtain a batch specific co-activation
            co_activation = np.mean(co_activation, axis=1)
            # print(f"np.mean(co_activation, axis=1).shape={co_activation.shape}")
            co_activations_flatten.append(co_activation)
            pairIDs.append([selected_channel_penultimate_layer[curr_fc1_feature], selected_channel_last_layer[curr_fc2_feature]])
    return co_activations_flatten, weight_changes_flatten, pairIDs


co_activations_flatten_, weight_changes_flatten_, pairIDs_ = dataPrepare()
# np.save('/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/temp/co_activations_flatten_.npy',
#         co_activations_flatten_)  # shape = [pair#, batch#]
# np.save('/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/temp/weight_changes_flatten_.npy',
#         weight_changes_flatten_)  # shape = [pair#, batch#]
# np.save('/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/temp/pairIDs_.npy',
#         pairIDs_)  # shape = [pair#, [ID1, ID2]]


def run_NMPH(co_activations_flatten, weight_changes_flatten, pairIDs, rows=None, cols=None):
    import matplotlib.pyplot as plt
    if rows is None:
        rows = int(np.ceil(np.sqrt(len(co_activations_flatten_)-1)))
    if cols is None:
        cols = int(np.sqrt(len(co_activations_flatten)))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Create a subplot matrix

    for i in range(len(co_activations_flatten)):
        x__ = co_activations_flatten[i]
        y__ = weight_changes_flatten[i]
        pairID = pairIDs[i]

        row = i // cols
        col = i % cols

        ax = axs[row, col]  # Select the appropriate subplot

        ax.scatter(x__, y__, s=10)  # 's' controls the size of the points

        # Add labels and a title to each subplot
        # ax.set_xlabel('Co-Activations')
        # ax.set_ylabel('Weight Changes')
        ax.set_title(f'pairID:{pairID}')

        # Hide x and y axis ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()  # Adjust subplot layout for better visualization
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


run_NMPH(co_activations_flatten_, weight_changes_flatten_, pairIDs_)

