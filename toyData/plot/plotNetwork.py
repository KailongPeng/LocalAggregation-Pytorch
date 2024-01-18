import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_network(input_size, hidden_sizes, output_size):
    fig, ax = plt.subplots(figsize=(20, 20))

    # Parameters
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    v_spacing = 0.2
    h_spacing = 0.7
    layer_padding = 0.5

    # Draw connections first
    for i in range(1, len(layer_sizes)):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i - 1]):
                color = 'red' if i == len(layer_sizes) - 1 else 'grey'
                alpha = 1 if i == len(layer_sizes) - 1 else 0.2
                ax.plot([i * h_spacing, (i - 1) * h_spacing],
                        [v_spacing * (layer_sizes[i] - 1) / 2.0 - j * v_spacing,
                         (v_spacing * (layer_sizes[i - 1] - 1) / 2.0) - k * v_spacing],
                        color=color, alpha=alpha)

    # Draw nodes on top of connections
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for j in range(layer_size):
            ax.add_patch(patches.Circle((i * h_spacing, layer_top - j * v_spacing), 0.1, fill=True, color='cyan'))

    # Set axis limits and remove axis labels
    ax.set_xlim(-1, len(layer_sizes) * h_spacing)
    ax.set_ylim(-max(layer_sizes) * v_spacing / 2, max(layer_sizes) * v_spacing / 2)
    ax.axis('off')

    plt.show()

# Specify neural network architecture
input_size = 2
hidden_sizes = [20, 20, 20]
output_size = 2

# Draw neural network
draw_neural_network(input_size, hidden_sizes, output_size)
