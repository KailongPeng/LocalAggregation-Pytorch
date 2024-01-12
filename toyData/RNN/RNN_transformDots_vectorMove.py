import torch
import torch.nn as nn
import torch.optim as optim

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
hidden_size = 20  # Size of hidden state
output_size = 2  # Size of output
model = VanillaRNN(input_size, hidden_size, output_size)

# Create a sample input sequence and target sequence
sequence_length = 100  # Length of sequence
batch_size = 3
input_data = torch.randn(batch_size, sequence_length, input_size)
target_data = torch.randn(batch_size, output_size)  # You need to provide the actual target values

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = model(input_data)

    # Compute the loss
    loss = criterion(output, target_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Print the final output shape
print("Final Output Shape:", output.shape)
