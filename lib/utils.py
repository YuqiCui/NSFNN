import torch
import torch.utils.data as data
import argparse

def data_loader(*arrays, last_long=True, batch_size=1, shuffle=False):
    torch_array = [torch.as_tensor(array).float() for array in arrays]
    if last_long:
        torch_array[-1] = torch_array[-1].long()
    dataset = data.TensorDataset(*torch_array)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')