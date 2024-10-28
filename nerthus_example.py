"""
When the dataset is downloaded, this script is an example of the application
of the first version of the model published on the dataset.

First, obtain the data. See ~/Usage_examples/Nerthus

Script tested in Ubuntu.

Author: [Your Name]
Institution: [Your Institution]
"""

# Python standard libraries
import os

# Third-party libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


#Controls
extract_data = False
plot_data = True



# Initialize paths
path_to_script = os.getcwd()
print("Path to script:", path_to_script)


dataset_path = os.path.join(
    path_to_script,
    "Usage_examples",
    "Nerthus",
    "nerthus-dataset-frames"
)
output_path = os.path.join(path_to_script, "Usage_examples", "Nerthus")
results_file = os.path.join(output_path, "results.txt")

# Loading model
model_path = os.path.join(path_to_script,  'Models', 'RESNET_BOWELPREP_SCALAR_26.h5')
model = load_model(model_path)

# Print model summary to find layer names
model.summary()






def preprocess_image(image):
    """
    Preprocess the input image.

    Parameters
    ----------
        image : np.array
            Input image to be preprocessed.

    Returns
    -------
        list of np.array
            List containing the original, grayscale, and HSV version of the preprocessed image.
    """
    image = cv2.resize(image,(256,256))
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    return gray_image,image/255


if extract_data:
    # Resetting results file
    with open(results_file, 'w') as f:
        f.write("Category;Scalar;\n")
    # Initialize lists to store predictions and categories
    scalar_preds = []
    categories = []

    # Process images
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if name.endswith(".jpg"):
                category = os.path.basename(root)
                print("\nCategory:", category, "Filename:", name)
                img_path = os.path.join(root, name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error reading image {img_path}")
                    continue

                print("Preprocessing image")
                im_grey, im_rgb = preprocess_image(img)

                print("Making prediction")
                predictions = model.predict(np.expand_dims(im_rgb, axis=0), batch_size=1)
                if isinstance(predictions, list) and len(predictions) == 2:
                    _, scalar_pred = predictions
                else:
                    scalar_pred = predictions

                scalar_value = scalar_pred[0][0]  # Extract scalar value

                print("Saving to results.txt")
                with open(results_file, 'a') as f:
                    f.write(f"{category};{scalar_value};\n")

                # Store results for plotting
                categories.append(int(category))
                scalar_preds.append(scalar_value)

if plot_data:
    print("\n-----------\n\nMaking plot\n\n-----------\n")

    # Read and interpret the results.txt file using NumPy
    data = np.genfromtxt(results_file, delimiter=';', skip_header=1, dtype=[('Category', 'i8'), ('Scalar', 'f8')])

    # Ensure data was read correctly
    if data.size == 0:
        print("No data to plot.")
        exit()

    # Extract unique categories
    categories = np.unique(data['Category'])

    # Prepare data for boxplot
    boxplot_data = []
    for cat in categories:
        scalars = data['Scalar'][data['Category'] == cat]
        boxplot_data.append(scalars)

    # Convert categories to strings for labeling
    category_labels = [str(cat) for cat in categories]

    # Make boxplot using matplotlib directly
    plt.figure(figsize=(8, 6))
    plt.boxplot(boxplot_data, labels=category_labels)
    plt.title('Scalar Predictions by BBPS')
    plt.xlabel('Nerthus BBPS')
    plt.ylabel('Scalar Prediction [OSABPS]]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'boxplot.png'))
    plt.show()