import os
import cv2  # for capturing videos
import math  # for mathematical operations
import matplotlib.pyplot as plt  # for plotting the images
import pandas as pd
import numpy as np  # for mathematical operations
import tensorflow as tf
from keras.src.utils import np_utils
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img # More specific imports
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize  # for resizing images
from keras.layers import Dense, InputLayer, Dropout

# -----------------------------------------------------------------
# Updated Step – 1: Updated by Gemini
# -----------------------------------------------------------------
# video_file = "tutorial.mp4"  # Path to your video file.
video_file = "shawn.mp4"  # Path to your video file.

cap = cv2.VideoCapture(video_file)  # Open the video file

if not cap.isOpened():  # Check if the video file opened successfully
    print(f"Error: Could not open video file: {video_file}")
    exit()  # Exit the script if the video couldn't be opened

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Correct way to get frame rate
if frame_rate <= 0:
    print("Error: Could not determine frame rate.")
    exit()


count = 0  # Initialize frame counter

while cap.isOpened():  # Loop through the video frames
    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Correct way to get frame number
    ret, frame = cap.read()  # Read a frame

    if not ret:  # Check if a frame was successfully read
        break  # Exit the loop if no frame was read

    if frame_id % math.floor(frame_rate) == 0:  # Save a frame approximately every second
        filename = f"frame{count}.jpg"  # Create the filename (using f-strings)
        cv2.imwrite(filename, frame)  # Save the frame
        count += 1  # Increment the counter

cap.release()  # Release the video capture object
print ("Done Step 1!")
# -----------------------------------------------------------------
# Step – 2: Label a few images for training the model
# Our next step is to read the images which we will do based on their names, aka, the Image_ID column.
# -----------------------------------------------------------------
# data = pd.read_csv('mapping.csv')  #reading the csv file
data = pd.read_csv('test_mapping.csv')
X = []
for img_name in data.Image_ID:
    try:  # Handle potential file errors
        img = plt.imread(img_name)
        X.append(img)
    except FileNotFoundError:
        print(f"Error: Image file '{img_name}' not found.")
        exit()  # Or handle the error differently if needed
X = np.array(X)    # converting list to array
# -----------------------------------------------------------------
# Since there are three classes, we will one hot encode them
# using the to_categorical() function of keras.utils.
# -----------------------------------------------------------------
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes
# -----------------------------------------------------------------
# We will be using a VGG16 pretrained model which takes an input image of shape (224 X 224 X 3).
# Since our images are in a different size, we need to reshape all of them. We will use the resize()
# function of skimage.transform to do this.
# -----------------------------------------------------------------
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
#
# # Correct way to preprocess for VGG16 (no 'mode' argument)
X = tf.keras.applications.vgg16.preprocess_input(X)
#
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set
# -----------------------------------------------------------------
# Step 3: Building the model
# The next step is to build our model. As mentioned, we shall be using the VGG16 pretrained model
# for this task. Let us first import the required libraries to build the model:
# -----------------------------------------------------------------
from keras.models import Sequential
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

X_train = X_train.reshape(X_train.shape[0], -1)  # Use shape[0] for dynamic reshaping
X_valid = X_valid.reshape(X_valid.shape[0], -1)  # Use shape[0] for dynamic reshaping

train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()

# i. Building the model
model = Sequential()
model.add(InputLayer((X_train.shape[1],)))  # Use X_train.shape[1] for dynamic input
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.summary()

# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# iii. Training the model
print("CHECKER")
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

print("Done Step 2!")

# -----------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------
video_file = "shawn2.mp4"  # Path to your video file.

cap = cv2.VideoCapture(video_file)  # Open the video file

if not cap.isOpened():  # Check if the video file opened successfully
    print(f"Error: Could not open video file: {video_file}")
    exit()  # Exit the script if the video couldn't be opened

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Correct way to get frame rate
if frame_rate <= 0:
    print("Error: Could not determine frame rate.")
    exit()


count = 0  # Initialize frame counter

while cap.isOpened():  # Loop through the video frames
    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Correct way to get frame number
    ret, frame = cap.read()  # Read a frame

    if not ret:  # Check if a frame was successfully read
        break  # Exit the loop if no frame was read

    if frame_id % math.floor(frame_rate) == 0:  # Save a frame approximately every second
        filename = f"test{count}.jpg"  # Create the filename (using f-strings)
        cv2.imwrite(filename, frame)  # Save the frame
        count += 1  # Increment the counter

cap.release()  # Release the video capture object
print("Extracted Test Frames")

# -----------------------------------------------------------------
# Gemini Reworked Above Code
# -----------------------------------------------------------------
# --- Configuration ---
TEST_CSV = "test.csv"  # Path to your test CSV
IMAGE_FOLDER = "."  # Path to the folder containing the images. **REPLACE THIS**
# ---------------------

test = pd.read_csv(TEST_CSV)

test_image = []
for img_name in test.Image_ID:
    image_path = os.path.join(IMAGE_FOLDER, img_name) # Construct full image path
    try:
        img = plt.imread(image_path)
        test_image.append(img)
    except FileNotFoundError:
        print(f"Error: Image '{img_name}' not found at '{image_path}'.")
        exit()  # Or handle the error differently

test_img = np.array(test_image)

# Efficient Resizing using NumPy
new_shape = (test_img.shape[0], 224, 224, 3) # assumes 3 channels (RGB)
test_image_resized = np.zeros(new_shape)

for i in range(test_img.shape[0]):
    test_image_resized[i] = resize(test_img[i], output_shape=(224, 224), preserve_range=True)

test_image_resized = np.array(test_image_resized)

# Consistent Preprocessing (Important!)
test_image_processed = tf.keras.applications.vgg16.preprocess_input(test_image_resized)

# ... (Now you can use test_image_processed for predictions) ...
# Preprocessing the images (Correct and consistent with training data)
test_image_processed = tf.keras.applications.vgg16.preprocess_input(test_image_resized)  # No 'mode' argument

# Extracting features using the pretrained model
test_image_features = base_model.predict(test_image_processed)

# Reshaping to 1-D (More robust)
test_image_features = test_image_features.reshape(test_image_features.shape[0], -1)  # Use shape[0] and -1

# Zero-centering (Important: Use the *training* data's max for consistency)
test_image_normalized = test_image_features / train.max() # Use train.max() not test_image.max()

# ... (Now use test_image_normalized for predictions) ...

# predictions = model.predict_classes(test_image) - This is outdated!
# Predictions (Corrected)
test_image_probabilities = model.predict(test_image_normalized)  # Get probabilities
test_image_predictions = np.argmax(test_image_probabilities, axis=-1)  # Get class indices

# print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
# print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")
# The rest of the code for calculating screen time should now work correctly with the new prediction array.
# print("The screen time of JERRY is", test_image_predictions[test_image_predictions == 1].shape[0], "seconds")
# print("The screen time of TOM is", test_image_predictions[test_image_predictions == 2].shape[0], "seconds")
print("The screen time of SHAWN is", test_image_predictions[test_image_predictions == 1].shape[0], "seconds")