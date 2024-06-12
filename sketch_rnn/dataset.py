import os
import six
import requests
import numpy as np
import torch

from .utils import get_max_len, to_tensor

__all__ = ['load_strokes', 'SketchRNNDataset', 'collate_drawings']

# start-of-sequence token
SOS = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)


def load_sketches(data_dir, hps):
    """Recursively loads .npz files from subclass directories and splits them into train and validation sets."""
    # Initialize lists for storing sketches
    all_sketches = []

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

                    # Extract sketches from npz file
                    sketch = []
                    for item in pen_strokes.files:
                        pen_stroke = pen_strokes[item]
                        sketch.append(pen_stroke)
                    all_sketches.append(sketch)

    # Shuffle the data to ensure randomness
    np.random.shuffle(all_sketches)

    # Calculate splits for train, validation, and test sets
    train_split = int(len(all_sketches) * 0.8)  # 80% for training
    valid_split = int(len(all_sketches) * 0.2)  # 20% for validation

    # Split the data
    train_sketches = all_sketches[:train_split]
    valid_sketches = all_sketches[train_split:train_split + valid_split]
    test_sketches = all_sketches[train_split + valid_split:]

    return train_sketches, valid_sketches, test_sketches


class SketchRNNDataset:
    def __init__(self,
                 sketches,
                 max_len=250,
                 scale_factor=None,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):
        self.sketches = sketches
        self.max_len = max_len
        self.random_scale_factor = random_scale_factor
        self.augment_stroke_prob = augment_stroke_prob
        self.limit = limit
        self.preprocess()
        self.normalize(scale_factor)

    def preprocess(self):
        """Remove sketches with length greater than max_len."""
        self.sketches = [sketch for sketch in self.sketches if len(sketch) <= self.max_len]

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor."""
        all_strokes = []
        for sketch in self.sketches:
            for stroke in sketch:
                all_strokes.extend(stroke[:, :2])
        all_strokes = np.array(all_strokes)
        return np.std(all_strokes)

    def normalize(self, scale_factor=None):
        """Normalize the dataset."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.sketches)):
            for j in range(len(self.sketches[i])):
                self.sketches[i][j][:, :2] /= self.scale_factor

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch = self.sketches[idx]
        if self.random_scale_factor > 0:
            sketch = self.random_scale(sketch)
        if self.augment_stroke_prob > 0:
            sketch = self.random_augment(sketch)
        
        # Convert lists of arrays to a single numpy array
        sketch_np = np.concatenate(sketch, axis=0)  # Ensure correct axis for concatenation

        return sketch_np


    def random_scale(self, sketch):
        """Augment sketch by stretching x and y axis randomly."""
        scaled_sketch = []
        for stroke in sketch:
            scaled_stroke = stroke.copy()
            x_scale = (np.random.rand() - 0.5) * 2 * self.random_scale_factor + 1.0
            y_scale = (np.random.rand() - 0.5) * 2 * self.random_scale_factor + 1.0
            scaled_stroke[:, 0] *= x_scale
            scaled_stroke[:, 1] *= y_scale
            scaled_sketch.append(scaled_stroke)
        return scaled_sketch

    def random_augment(self, sketch):
        """Perform data augmentation by randomly dropping out strokes."""
        augmented_sketch = []
        prev_stroke = np.array([0, 0, 1])
        count = 0
        for stroke in sketch:
            if len(stroke) == 3:
                candidate = np.array([stroke[0], stroke[1], 0])
            else:
                candidate = stroke
            if len(candidate) == 3 and (candidate[2] == 1).any():
                count = 0
            else:
                count += 1
            check = (len(candidate) == 3 and (candidate[2] == 0).all()) and (len(prev_stroke) == 3 and (prev_stroke[2] == 0).all()) and count > 2
            if check and (np.random.rand() < self.augment_stroke_prob):
                prev_stroke[0] += candidate[0]
                prev_stroke[1] += candidate[1]
            else:
                prev_stroke = candidate
                augmented_sketch.append(prev_stroke)
        return augmented_sketch


# ---- methods for batch collation ----

def pad_batch(sequences, max_len):
    batch_size = len(sequences)
    padded_sequences = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded_seq = torch.tensor(seq[:length], dtype=torch.float32)
        padded_sequences[i, :length, :] = padded_seq
        lengths[i] = length


    return padded_sequences, lengths

def collate_drawings(sequences, max_len):
    print("Sequences before padding:")
    for i, seq in enumerate(sequences):
        print(f"Sequence {i} type:", type(seq))
        print(f"Sequence {i} length:", len(seq))

    padded_batch, lengths = pad_batch(sequences, max_len)

    return padded_batch, lengths

