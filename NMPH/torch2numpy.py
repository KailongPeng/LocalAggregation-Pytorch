import torch
import numpy as np
from tqdm import tqdm
for epoch in range(0, 1):
    for batch_i in tqdm(range(0, 100)):
        directory_torch_path = '/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/weights_difference/'
        directory_path = '/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/weights_difference/numpy/'
        # load activations and weights
        activation_lastLayer = torch.load(f'{directory_torch_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar')
        np.save(f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar',
                np.asarray(activation_lastLayer))
        activation_secondLastLayer = torch.load(f'{directory_torch_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar')
        np.save(f'{directory_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.pth.tar',
                np.asarray(activation_secondLastLayer))
        weight_change = torch.load(f'{directory_torch_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.pth.tar', map_location=torch.device('cpu'))  # .detach().numpy()
        np.save(f'{directory_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.pth.tar',
                np.asarray(weight_change))
