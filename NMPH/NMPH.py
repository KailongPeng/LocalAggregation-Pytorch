def dataPrepare():
    fc1_activations = []
    fc2_activations = []
    fc2_partial_weights = []
    weight_changes = []

    for epoch in range(1, 2):
        for batch_idx in range(saveInterval - 1, trainBatchNum + 1, saveInterval):
            directory_path = './features_weights'  # Replace with your desired path
            os.makedirs(directory_path, exist_ok=True)

            # load activations and weights
            fc1_activation = torch.load(f'{directory_path}/fc1_activations_epoch{epoch}_batch_idx{batch_idx}.pth')
            fc2_activation = torch.load(f'{directory_path}/fc2_activations_epoch{epoch}_batch_idx{batch_idx}.pth')
            # fc2_partial_weight = torch.load(f'{directory_path}/fc2_partial_weights_batch_idx{batch_idx}.pth')
            weight_change = torch.load(
                f'{directory_path}/weight_change_epoch{epoch}_batch_idx{batch_idx}.pth').detach().numpy()

            fc1_activations.append(fc1_activation)
            fc2_activations.append(fc2_activation)
            # fc2_partial_weights.append(fc2_partial_weight)
            weight_changes.append(weight_change)
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
            fc1_activation = fc1_activations[:, :, fc1_feature]
            fc2_activation = fc2_activations[:, :, fc2_feature]
            weight_change = weight_changes[:, fc2_feature, fc1_feature]
            weight_changes_flatten.append(weight_change)
            # Calculate the co-activation
            co_activation = np.multiply(fc1_activation, fc2_activation)
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