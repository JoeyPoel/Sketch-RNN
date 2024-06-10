import os
import numpy as np
from svgpathtools import svg2paths2
from tqdm import tqdm

def svg_to_npz(svg_path, npz_path):
    """
    Convert an SVG file to a numpy array and save it as a .npz file.
    """
    try:
        # Load the SVG file
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        
        all_paths = []
        
        for path in paths:
            for segment in path:
                all_paths.append([segment.start.real, segment.start.imag])
                all_paths.append([segment.end.real, segment.end.imag])
        
        # Convert to numpy array
        np_data = np.array(all_paths)

        # Save as .npz file
        np.savez(npz_path, drawing=np_data)
    except Exception as e:
        print(f"Error processing file {svg_path}: {e}")

def create_dataset(svg_dir, npz_dir):
    """
    Convert all SVG files in a directory to .npz files and save them to another directory.
    """
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)

    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]

    if not svg_files:
        print(f"No SVG files found in the directory: {svg_dir}")
    else:
        print(f"Found {len(svg_files)} SVG files in the directory: {svg_dir}")

    for svg_file in tqdm(svg_files, desc="Converting SVGs to NPZs"):
        svg_path = os.path.join(svg_dir, svg_file)
        npz_path = os.path.join(npz_dir, svg_file.replace('.svg', '.npz'))
        svg_to_npz(svg_path, npz_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SVG files to NPZ format for Sketch-RNN")
    parser.add_argument('--svg_dir', type=str, required=True, help="Directory containing SVG files")
    parser.add_argument('--npz_dir', type=str, required=True, help="Directory to save NPZ files")

    args = parser.parse_args()
    
    create_dataset(args.svg_dir, args.npz_dir)
