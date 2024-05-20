import numpy as np
import matplotlib.pyplot as plt

# Replace 'path_to_your_file.npz' with the path to the .npz file you want to check
data = np.load('path_to_your_file.npz', allow_pickle=True)

# Print the number of arrays in the file
print(f"Number of arrays in the file: {len(data.files)}")

# Print the contents of each array
for i, item in enumerate(data.files):
    print(f"\nArray {i}:")
    print(data[item])

# Create a new plot
plt.figure()

# Plot each pen stroke
for item in data.files:
    pen_stroke = data[item]
    # Transpose the pen stroke array to separate the x and y coordinates
    x, y = np.transpose(pen_stroke)
    # Plot the pen stroke
    plt.plot(x, -y)  # Multiply y by -1 to flip the y-axis

# Display the plot
plt.show()
