import torch
import torch.nn as nn

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

# Create a sample input sequence
sequence_length = 100  # Length of sequence
batch_size = 3
input_data = torch.randn(batch_size, sequence_length, input_size)

# Forward pass
output = model(input_data)

# Print the output shape
print("Output Shape:", output.shape)
