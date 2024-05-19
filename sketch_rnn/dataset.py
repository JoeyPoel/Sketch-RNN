import os
import six
import requests
import numpy as np
import torch

from .utils import get_max_len, to_tensor

__all__ = ['load_strokes', 'SketchRNNDataset', 'collate_drawings']

# start-of-sequence token
SOS = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)


def load_strokes(data_dir, hps):
    """Recursively loads .npz files from subclass directories and splits them into train and validation sets."""

    # Initialize lists for storing strokes
    all_strokes = []

    # Traverse each subclass directory
    for subclass in os.listdir(data_dir):
        subclass_path = os.path.join(data_dir, subclass)
        
        # Check if it's a directory
        if os.path.isdir(subclass_path):
            
            # Traverse files in the subclass directory
            for file_name in os.listdir(subclass_path):
                if file_name.endswith('.npz'):
                    file_path = os.path.join(subclass_path, file_name)
                    pen_strokes = np.load(file_path, allow_pickle=True)
                    
                    # Extract strokes from npz file
                    for item in pen_strokes.files:
                        pen_stroke = pen_strokes[item]
                        all_strokes.append(pen_stroke)

    # Shuffle the data to ensure randomness
    np.random.shuffle(all_strokes)

    # Split the data into train and validation sets
    split_index = int(len(all_strokes) * 0.8)  # 80% for training
    train_strokes = all_strokes[:split_index]
    valid_strokes = all_strokes[split_index:]

    return train_strokes, valid_strokes, None

class SketchRNNDataset:
    def __init__(self,
                 strokes,
                 max_len=250,
                 scale_factor=None,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        # Convert strokes to tensors
        strokes = [torch.tensor(stk, dtype=torch.float) for stk in strokes]  # Convert to float tensors
        self.max_len = max_len  # N_max in sketch-rnn paper
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.limit = limit # clamp x-y offsets to range (-limit, limit)
        self.preprocess(strokes) # list of drawings in stroke-3 format, sorted by size
        self.normalize(scale_factor)

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_len points.
        Clamp x-y values to (-limit, limit)
        """
        raw_data = []
        seq_len = []
        count_data = 0
        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_len):
                count_data += 1
                data = data.clamp(-self.limit, self.limit)
                raw_data.append(data)
                seq_len.append(len(data))
        self.sort_idx = np.argsort(seq_len)
        self.strokes = [raw_data[ix] for ix in self.sort_idx]
        print("total drawings <= max_seq_len is %d" % count_data)

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        strokes = [elt for elt in self.strokes if len(elt) <= self.max_len]
        data = torch.cat(strokes)
        return data[:,:2].std()

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:,:2] /= self.scale_factor

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        data = self.strokes[idx]
        if self.random_scale_factor > 0:
            data = random_scale(data, self.random_scale_factor)
        if self.augment_stroke_prob > 0:
            data = random_augment(data, self.augment_stroke_prob)
        return data


def random_scale(data, factor):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    data = data.clone()
    x_scale = (torch.rand(()) - 0.5) * 2 * factor + 1.0
    y_scale = (torch.rand(()) - 0.5) * 2 * factor + 1.0
    data[:,0] *= x_scale
    data[:,1] *= y_scale
    return data

def random_augment(data, prob):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    data = data.clone()
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(data)):
        candidate = [data[i][0], data[i][1], data[i][2]] if len(data[i]) > 2 else [data[i][0], data[i][1], 0]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        check = candidate[2] == 0 and prev_stroke[2] == 0 and count > 2
        if check and (torch.rand(()) < prob):
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    result = torch.tensor(result, dtype=torch.float)
    return result



# ---- methods for batch collation ----

def pad_batch(sequences, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    batch_size = len(sequences)
    output = torch.zeros(batch_size, max_len+1, 5)
    for i in range(batch_size):
        seq, out = sequences[i], output[i]
        l = len(seq)
        assert l <= max_len
        # fill sos value
        out[0] = SOS
        # fill remaining values
        out = out[1:]
        out[:l,:2] = seq[:,:2]
        if len(seq[0]) == 3:  # Check if the sequence has three dimensions
            out[:l,3] = seq[:,2]
            out[:l,2] = 1 - out[:l,3]
        else:  # If not, pad with zeros
            out[:l,2] = 0
            out[:l,3] = 0
        out[l:,4] = 1
    return output

def collate_drawings(sequences, max_len):
    lengths = torch.tensor([len(seq) for seq in sequences],
                           dtype=torch.long)
    batch = pad_batch(sequences, max_len)
    return batch, lengths