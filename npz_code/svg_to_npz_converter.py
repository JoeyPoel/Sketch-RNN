import os
import xml.etree.ElementTree as ET
import numpy as np
import re

class Data:
    def __init__(self, data_path='sketches_svg/svg', max_seq_length=100):
        # Initialize the Data object with the path to the data and the maximum sequence length
        self.data_path = data_path
        self.classes = os.listdir(data_path)  # List all directories in the data path
        self.num_classes = len(self.classes)  # Count the number of classes (directories)
        self.max_seq_length = max_seq_length
        self.train = self.load_data()  # Load the data

    def load_data(self):
        data = []
        # Iterate over each class (directory)
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            # Check if class_path is a directory
            if os.path.isdir(class_path):
                # Iterate over each file in the class directory
                for file_name in os.listdir(class_path):
                    # Check if the file is an SVG file
                    if file_name.endswith('.svg'):
                        file_path = os.path.join(class_path, file_name)
                        # Parse the SVG file into pen strokes
                        pen_strokes = self.parse_svg(file_path)
                        # Save each SVG's pen strokes to its own NPZ file
                        output_dir = os.path.join('sketches_npz', class_name)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.splitext(file_name)[0] + '.npz'
                        output_path = os.path.join(output_dir, output_file)
                        self.save_as_npz(pen_strokes, output_path)
                        # Append the class name and NPZ file path to the data list
                        data.append({'class': class_name, 'npz_file': output_path})
        return data

    def parse_svg(self, svg_file):
        pen_strokes = []
        tree = ET.parse(svg_file)
        root = tree.getroot()
        for path in root.iter('{http://www.w3.org/2000/svg}path'):
            d = path.attrib['d']
            commands = re.findall('([A-Za-z])([^A-Za-z]*)', d)
            pen_stroke = []
            for command, parameters in commands:
                parameters = re.split('[ ,]', parameters.strip())
                # Check if the command is a 'move to' command
                if command.upper() == 'M':
                    # If there are existing points, save them as a pen stroke
                    if pen_stroke:
                        pen_strokes.append(pen_stroke)
                        pen_stroke = []
                # Add the points to the current pen stroke
                for i in range(0, len(parameters), 2):
                    try:
                        x, y = map(float, parameters[i:i+2])
                        pen_stroke.append([x, y])
                    except ValueError:
                        print(f"Warning: Skipping malformed point (non-numeric) in {svg_file}: {parameters[i:i+2]}")
            # Save the last pen stroke
            if pen_stroke:
                pen_strokes.append(pen_stroke)
        return pen_strokes

    def save_as_npz(self, pen_strokes, file_path):
        # Save the pen strokes to an NPZ file
        np.savez(file_path, *pen_strokes)

# Example usage
data_loader = Data()
