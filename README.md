# Actor Screen Time Analysis

This project analyzes the screen time of actors in a video using computer vision and deep learning. It extracts frames from the video, classifies them based on the actor present, and calculates the total screen time for each actor.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Introduction

This project aims to automate the process of determining the screen time of actors in a video. It leverages the VGG16 pre-trained model for image classification and provides a breakdown of screen time for each actor.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/yourusername/yourrepositoryname.git](https://github.com/yourusername/yourrepositoryname.git)  # Replace with your repository URL
    cd actorScreenTime
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv  # Create a virtual environment
    source .venv/bin/activate  # Activate the environment (Linux/macOS)
    .venv\Scripts\activate  # Activate the environment (Windows)
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file listing all project dependencies.  See the [Dependencies](#dependencies) section for the contents)

## Usage

1.  **Prepare your data:**

    *   Place your video file (e.g., `shawn.mp4`, `shawn2.mp4`) in the project directory.
    *   Create a CSV file (e.g., `test_mapping.csv`, `test.csv`) that maps image filenames (e.g., `frame0.jpg`, `test0.jpg`) to actor labels (e.g., 0, 1, 2 for different actors).  Ensure that the filenames match the frames extracted from the video.

2.  **Run the script:**

    ```bash
    python test.py
    ```

    The script will:

    *   Extract frames from the video(s).
    *   Load and preprocess the image data from the CSV file.
    *   Train a model (if training data is provided).
    *   Make predictions on the test data.
    *   Print the screen time for each actor.

## Project Structure
actorScreenTime/
├── test.py             # Main Python script
├── extracted_frames/   # Directory for extracted frames
├── test_mapping.csv    # CSV file for training data (image filenames and labels)
├── test.csv            # CSV file for test data (image filenames)
├── shawn.mp4           # Example video file
├── shawn2.mp4          # Example video file
└── requirements.txt    # List of project dependencies

## Data Preparation

1.  **Video Frame Extraction:** The script extracts frames from the video and saves them as JPEG images in the `extracted_frames` directory.  The `frame_rate` variable in the script controls how frequently frames are extracted.  You can modify the `frame_interval` parameter in the `extract_frames` function to control this.

2.  **CSV Files:**

    *   `test_mapping.csv`: This file maps the names of the extracted frames (e.g., `frame0.jpg`) to the corresponding actor labels (e.g., 0, 1, 2).  This is used for training.  It should have columns named `Image_ID` and `Class`.
    *   `test.csv`: This file contains the names of the frames extracted from the test video, for which you want to predict the actor. It should have a column named `Image_ID`.

## Model Training

The script uses a pre-trained VGG16 model as a base and adds a few dense layers on top for classification.  The model is trained on the data provided in `test_mapping.csv`.

## Testing

The script evaluates the trained model on the test data (frames extracted from `shawn2.mp4` and listed in `test.csv`) and prints the predicted screen time for each actor.

## Dependencies

Create a file named `requirements.txt` in your project directory and include the following:

opencv-python
matplotlib
pandas
numpy
tensorflow
scikit-image
scikit-learn
keras

