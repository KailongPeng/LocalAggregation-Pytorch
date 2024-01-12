"""
Circuit mechanisms for the maintenance and manipulation of information in working memory
https://www.nature.com/articles/s41593-019-0414-3#code-availability
https://github.com/nmasse/Short-term-plasticity-RNN/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import stimulus
import analysis
import pickle
import time
from parameters import par
import os

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all weights, biases, and initial values

        self.var_dict = nn.ParameterDict()
        # All keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in par.items():
            if k[-1] == '0':
                name = k[:-1]
                self.var_dict[name] = nn.Parameter(torch.tensor(par[k], dtype=torch.float32))

        self.syn_x_init = torch.tensor(par['syn_x_init'])
        self.syn_u_init = torch.tensor(par['syn_u_init'])
        if par['EI']:
            # Ensure excitatory neurons only have positive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            self.w_rnn = torch.matmul(torch.tensor(par['EI_matrix']), torch.nn.functional.relu(self.var_dict['w_rnn']))
        else:
            self.w_rnn = self.var_dict['w_rnn']

    def forward(self, input_data, target_data, mask):
        # Load the input activity, the target data, and the training mask for this batch of trials
        input_data = input_data.unbind(0)

        self.h = []
        self.syn_x = []
        self.syn_u = []
        self.y = []

        h = self.var_dict['h']
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init

        # Loop through the neural inputs to the RNN, indexed in time
        for rnn_input in input_data:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.h.append(h)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            self.y.append(torch.matmul(h, torch.nn.functional.relu(self.var_dict['w_out'])) + self.var_dict['b_out'])

        self.h = torch.stack(self.h)
        self.syn_x = torch.stack(self.syn_x)
        self.syn_u = torch.stack(self.syn_u)
        self.y = torch.stack(self.y)

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):
        # Update neural activity and short-term synaptic plasticity values

        # Update the synaptic plasticity parameters
        if par['synapse_config'] is not None:
            # Implement both synaptic short-term facilitation and depression
            syn_x += (par['alpha_std'] * (1 - syn_x) - par['dt_sec'] * syn_u * syn_x * h) * par['dynamic_synapse']
            syn_u += (par['alpha_stf'] * (par['U'] - syn_u) + par['dt_sec'] * par['U'] * (1 - syn_u) * h) * par['dynamic_synapse']
            syn_x = torch.minimum(torch.tensor(1.0), torch.nn.functional.relu(syn_x))
            syn_u = torch.minimum(torch.tensor(1.0), torch.nn.functional.relu(syn_u))
            h_post = syn_u * syn_x * h
        else:
            # No synaptic plasticity
            h_post = h

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        h = torch.nn.functional.relu(h * (1 - par['alpha_neuron']) +
                                      par['alpha_neuron'] * (torch.matmul(rnn_input,
                                                                         torch.nn.functional.relu(self.var_dict['w_in'])) +
                                                            torch.matmul(h_post, self.w_rnn) +
                                                            self.var_dict['b_rnn']) +
                                      torch.randn_like(h) * par['noise_rnn'])

        return h, syn_x, syn_u


def main(gpu_id=None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Print key parameters
    print_important_params()

    # Reset PyTorch before running anything
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create the stimulus class to generate trial parameters and input activity
    stim = stimulus.Stimulus()

    # Define all placeholders
    model = Model()

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=par['learning_rate'])

    # Keep track of the model performance across training
    model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [],
                         'weight_loss': [], 'iteration': []}

    for i in range(par['num_iterations']):

        # Generate batch of batch_train_size
        trial_info = stim.generate_trial(set_rule=None)

        # Run the model
        model.forward(trial_info['neural_input'], trial_info['desired_output'], trial_info['train_mask'])

        loss, perf_loss, spike_loss, weight_loss = calculate_loss(model.y, trial_info['desired_output'],
                                                                  model.h, model.var_dict['w_rnn'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], model.y, trial_info['train_mask'])

        model_performance = append_model_performance(model_performance, accuracy, loss.item(),
                                                      perf_loss.item(), spike_loss.item(), weight_loss.item(), i)

        # Save the network model and output model performance to screen
        if i % par['iters_between_outputs'] == 0:
            print_results(i, perf_loss.item(), spike_loss.item(), weight_loss.item(), model.h, accuracy)

    # Save model and results
    weights = {key: value.detach().numpy() for key, value in model.var_dict.items()}
    save_results(model_performance, weights)


def calculate_loss(y, target_data, h, w_rnn):
    perf_loss = nn.CrossEntropyLoss()(y, target_data.argmax(dim=2))
    spike_loss = torch.mean(h**2) if par['spike_regularization'] == 'L2' else torch.mean(h)
    weight_loss = torch.mean(torch.nn.functional.relu(w_rnn)**2) if par['spike_regularization'] == 'L2' else \
        torch.mean(torch.nn.functional.relu(w_rnn))

    loss = perf_loss + par['spike_cost'] * spike_loss + par['weight_cost'] * weight_loss

    return loss, perf_loss, spike_loss, weight_loss


def save_results(model_performance, weights, save_fn=None):
    results = {'weights': weights, 'parameters': par}
    for k, v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ', fn)


def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration):
    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)

    return model_performance


def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy):
    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
          ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
          ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(h.mean().item()))


def print_important_params():
    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd', 'spike_cost',
                        'spike_regularization', 'weight_cost', 'test_cost_multiplier', 'trial_type', 'balance_EI', 'dt',
                        'delay_time', 'connection_prob', 'synapse_config', 'tau_slow', 'tau_fast']
    for k in important_params:
        print(k, ': ', par[k])


if __name__ == "__main__":
    main()
