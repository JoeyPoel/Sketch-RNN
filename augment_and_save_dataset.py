import os
import numpy as np

def augment_and_save_dataset(data_dir):
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

    def flip_horizontally(sketch):
        """Flip the sketch horizontally by inverting the x-coordinates."""
        flipped_sketch = []
        for stroke in sketch:
            if stroke.ndim == 2 and stroke.shape[1] >= 2:
                flipped_stroke = stroke.copy()
                flipped_stroke[:, 0] = -flipped_stroke[:, 0]
                flipped_sketch.append(flipped_stroke)
            else:
                flipped_sketch.append(stroke)
        return flipped_sketch

    def save_sketches_to_file(sketches, base_file_path, augmentation_type):
        base_dir = os.path.dirname(base_file_path)
        base_name = os.path.basename(base_file_path)
        augmented_file_path = os.path.join(base_dir, base_name.replace('.npz', f'_{augmentation_type}.npz'))
        np.savez_compressed(augmented_file_path, *sketches)
        print(f"Saved {augmentation_type} to {augmented_file_path}")

    all_sketches = load_sketches_from_dir(data_dir)
    for sketch, file_path in all_sketches:
        original_sketch = [s.copy() for s in sketch]

        # Apply horizontal flipping and save the augmented data
        flipped_sketch = flip_horizontally(original_sketch)
        save_sketches_to_file(flipped_sketch, file_path, 'horizontal_flip')

# Example usage
augment_and_save_dataset('sketches_npz')
