import os
import torch

import pandas as pd


def get_data(sampling_method='first_k',
             num_samples=5,
             data_root='./'):

    assert sampling_method in ['first_k']

    col_names = ['time', 'PPG', 'abp']
    sample_list = []

    if sampling_method == 'first_k':
        for patient_id in sorted(os.listdir(data_root)):
            patient_dir = os.path.join(data_root, patient_id)

            for signal_name in sorted(os.listdir(patient_dir))[:num_samples]:
                sample = pd.read_csv(f'{os.path.join(patient_dir, signal_name)}', names=col_names)
                sample_tensor = torch.tensor(sample['PPG'].values)
                sample_list.append(sample_tensor)

    len_seq = len(sample_list)
    training_seq = torch.stack(sample_list).unsqueeze(1).half()
    
    return training_seq, len_seq