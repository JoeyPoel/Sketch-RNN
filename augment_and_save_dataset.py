import os
import numpy as np

def augment_and_save_dataset(data_dir, random_scale_factors=[0.1, 0.2, 0.3], augment_stroke_probs=[0.05, 0.1, 0.15]):
    """Load sketches, apply multiple types of data augmentation, and save augmented sketches with new filenames."""

    def load_sketches_from_dir(directory):
        sketches = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith('.npz'):
                    file_path = os.path.join(root, file_name)
                    pen_strokes = np.load(file_path, allow_pickle=True)
                    sketch = []
                    for item in pen_strokes.files:
                        pen_stroke = pen_strokes[item]
                        sketch.append(pen_stroke)
                    sketches.append((sketch, file_path))
        return sketches

    def save_sketches_to_file(sketches, base_file_path, augmentation_type):
        augmented_file_path = base_file_path.replace('.npz', f'_{augmentation_type}.npz')
        np.savez_compressed(augmented_file_path, *sketches)

    def apply_random_scale(sketch, scale_factor):
        scaled_sketch = []
        for stroke in sketch:
            if stroke.ndim == 2 and stroke.shape[1] >= 2:  # Ensure the stroke is 2D with at least 2 columns
                scaled_stroke = stroke.copy()
                x_scale = (np.random.rand() - 0.5) * 2 * scale_factor + 1.0
                y_scale = (np.random.rand() - 0.5) * 2 * scale_factor + 1.0
                scaled_stroke[:, 0] *= x_scale
                scaled_stroke[:, 1] *= y_scale
                scaled_sketch.append(scaled_stroke)
            else:
                scaled_sketch.append(stroke)
        return scaled_sketch

    def apply_random_augment(sketch, augment_prob):
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
            if check and (np.random.rand() < augment_prob):
                prev_stroke[0] += candidate[0]
                prev_stroke[1] += candidate[1]
            else:
                prev_stroke = candidate
                augmented_sketch.append(prev_stroke)
        return augmented_sketch

    def flip_horizontally(sketch):
        """Flip the sketch horizontally by inverting the x-coordinates."""
        flipped_sketch = []
        for stroke in sketch:
            if stroke.ndim == 2 and stroke.shape[1] >= 2:  # Ensure the stroke is 2D with at least 2 columns
                flipped_stroke = stroke.copy()
                flipped_stroke[:, 0] = -flipped_stroke[:, 0]
                flipped_sketch.append(flipped_stroke)
            else:
                flipped_sketch.append(stroke)
        return flipped_sketch

    all_sketches = load_sketches_from_dir(data_dir)
    for sketch, file_path in all_sketches:
        # Apply multiple random scalings and save the augmented data
        for scale_factor in random_scale_factors:
            scaled_sketch = [apply_random_scale(s, scale_factor) for s in sketch]
            save_sketches_to_file(scaled_sketch, file_path, f'random_scale_{scale_factor}')
        
        # Apply multiple random augmentations and save the augmented data
        for augment_prob in augment_stroke_probs:
            augmented_sketch = [apply_random_augment(s, augment_prob) for s in sketch]
            save_sketches_to_file(augmented_sketch, file_path, f'random_augment_{augment_prob}')

        # Apply horizontal flipping and save the augmented data
        flipped_sketch = [flip_horizontally(s) for s in sketch]
        save_sketches_to_file(flipped_sketch, file_path, 'horizontal_flip')
        
        # Additional augmentations can be added here similarly

# Example usage
augment_and_save_dataset('sketches_npz')
