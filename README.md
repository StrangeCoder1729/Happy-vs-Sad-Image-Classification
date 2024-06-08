# Happy vs Sad Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of people as either happy or sad. The project involves loading image data, preprocessing it, building and training the CNN model, and testing the model on new images.

## Table of Contents

1. [Setup](#setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Building](#model-building)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Making Predictions](#making-predictions)
7. [Saving and Loading the Model](#saving-and-loading-the-model)
8. [References](#references)

## Setup

Ensure you have the necessary libraries installed, including TensorFlow, OpenCV, and Matplotlib.

## Data Preprocessing

- **Data Loading**: Load the image data from the specified directory.
- **Image Filtering**: Check the format of each image and remove any that are not in the desired extensions (jpeg, jpg, bmp, png).
- **Normalization**: Normalize the pixel values to scale the images.

## Model Building

- **Define CNN Architecture**: Build a Sequential model using Keras, including convolutional layers, pooling layers, flattening, and dense layers.
- **Compilation**: Compile the model with the appropriate loss function and optimizer for binary classification.

## Training the Model

- **Dataset Splitting**: Split the dataset into training, validation, and test sets.
- **Model Training**: Train the model using the training dataset, with validation on the validation dataset.
- **Callbacks**: Use TensorBoard for logging and visualizing training progress.

## Evaluating the Model

- **Evaluation Metrics**: Calculate precision, recall, and accuracy using the test dataset.
- **Visualization**: Plot the loss and accuracy for both training and validation sets to evaluate model performance.

## Making Predictions

- **Image Preprocessing**: Load and preprocess new images.
- **Prediction**: Use the trained model to predict whether the person in the image is happy or sad.
- **Display Results**: Display the images and prediction results.

## Saving and Loading the Model

- **Model Saving**: Save the trained model to a file.
- **Model Loading**: Load the saved model and use it for making new predictions.

## References

This project references tutorials and resources from [Nicholas Renotte's YouTube Channel](https://www.youtube.com/@NicholasRenotte/featured).

## Libraries Used

- TensorFlow
- OpenCV
- Matplotlib
- NumPy

 
