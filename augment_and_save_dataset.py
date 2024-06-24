import os
import numpy as np
from scipy.ndimage import rotate

def augment_and_save_dataset(data_dir, random_scale_factors=[0.05, 0.025, 0.0125]):
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

    def apply_random_scale(sketch, scale_factor):
        scaled_sketch = []
        for stroke in sketch:
            if stroke.ndim == 2 and stroke.shape[1] >= 2:
                scaled_stroke = stroke.copy()
                x_scale = (np.random.rand() - 0.5) * 2 * scale_factor + 1.0
                y_scale = (np.random.rand() - 0.5) * 2 * scale_factor + 1.0
                print(f"Scaling stroke by x_scale: {x_scale}, y_scale: {y_scale}")  # Debug statement
                scaled_stroke[:, 0] *= x_scale
                scaled_stroke[:, 1] *= y_scale
                scaled_sketch.append(scaled_stroke)
            else:
                scaled_sketch.append(stroke)
        return scaled_sketch

    def apply_random_rotation(sketch, angle_range=(-5, 5)):
        rotated_sketch = []
        for stroke in sketch:
            if stroke.ndim == 2 and stroke.shape[1] >= 2:
                angle = np.random.uniform(angle_range[0], angle_range[1])
                print(f"Rotating stroke by angle: {angle} degrees")  # Debug statement
                rotated_stroke = rotate_single_stroke(stroke, angle)
                rotated_sketch.append(rotated_stroke)
            else:
                rotated_sketch.append(stroke)
        return rotated_sketch

    def rotate_single_stroke(stroke, angle):
        # Extract x and y coordinates from the stroke
        x_coords = stroke[:, 0]
        y_coords = stroke[:, 1]

        # Calculate the center of rotation (centroid)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        # Convert angle to radians (scipy.ndimage.rotate uses degrees)
        angle_rad = np.deg2rad(angle)

        # Rotate coordinates around the centroid
        rotated_x = center_x + np.cos(angle_rad) * (x_coords - center_x) - np.sin(angle_rad) * (y_coords - center_y)
        rotated_y = center_y + np.sin(angle_rad) * (x_coords - center_x) + np.cos(angle_rad) * (y_coords - center_y)

        # Combine rotated coordinates into a new stroke array
        rotated_stroke = np.stack((rotated_x, rotated_y), axis=-1)

        # Ensure stroke format (add pen state if it exists)
        if stroke.shape[1] > 2:
            rotated_stroke = np.column_stack((rotated_stroke, stroke[:, 2]))

        return rotated_stroke

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

        # Apply scaling and rotation to original sketches
        for scale_factor in random_scale_factors:
            scaled_sketch = apply_random_scale(original_sketch, scale_factor)
            save_sketches_to_file(scaled_sketch, file_path, f'original_scale_{scale_factor}')

            rotated_sketch = apply_random_rotation(original_sketch)
            save_sketches_to_file(rotated_sketch, file_path, 'original_rotation')

        # Flip sketches horizontally and apply scaling and rotation
        flipped_sketch = flip_horizontally(original_sketch)
        save_sketches_to_file(flipped_sketch, file_path, 'horizontally_flipped')

        for scale_factor in random_scale_factors:
            scaled_sketch = apply_random_scale(flipped_sketch, scale_factor)
            save_sketches_to_file(scaled_sketch, file_path, f'horizontally_flipped_scale_{scale_factor}')

            rotated_sketch = apply_random_rotation(flipped_sketch)
            save_sketches_to_file(rotated_sketch, file_path, 'horizontally_flipped_rotation')


if __name__ == '__main__':
    # Example usage
    augment_and_save_dataset('test_dataset')
