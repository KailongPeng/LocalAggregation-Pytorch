import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_3d_scatter_plot():
    # Generate 1000 random 3D points
    np.random.seed(42)
    points = np.random.rand(1000, 3)

    # Define the boundaries for region division
    boundaries = [1 / 3, 2 / 3]

    # Assign labels to each point based on region
    labels_x = np.digitize(points[:, 0], boundaries)
    labels_y = np.digitize(points[:, 1], boundaries)
    labels_z = np.digitize(points[:, 2], boundaries)

    # Combine the indices of the regions in each dimension to get a unique label for each point
    labels = labels_x + 3 * labels_y + 9 * labels_z

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colored points based on labels
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='viridis')

    # Add colorbar to show label-color mapping
    colorbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    colorbar.set_label('Label')

    # Set plot labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Scatter Plot with Region Labels')

    # Return data and figure
    return points, labels, fig


# Call the function to get data and figure
points_data, labels_data, plot_figure = generate_3d_scatter_plot()

# Display the plot
plt.show()
