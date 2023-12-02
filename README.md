# Classification of Tomato Plant Disease

This project focuses on building a deep learning model for classifying various diseases in tomato plants. The project utilizes TensorFlow and Keras for model development and image data augmentation. Below is a detailed guide on the project structure, data preparation, model building, training, evaluation, and prediction.

## Dataset

Datasets was collected from Kaggle: [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

### Motivation
The motivation behind this project is to create a deep learning model capable of classifying different diseases affecting tomato plants. By leveraging convolutional neural networks (CNNs), the aim is to assist farmers in identifying and addressing plant diseases early, thereby enhancing crop yield.

### Project Description
This project utilizes TensorFlow and Keras to build a CNN-based model for classifying various diseases in tomato plants. The dataset includes images of tomato plants with annotations for different disease categories. The model is trained, validated, and tested on these datasets to achieve accurate disease classification.

### Problem Statement
Plant diseases can significantly impact crop yield, and early detection is crucial for effective management. This project addresses the challenge of timely and accurate identification of diseases in tomato plants using machine learning.

### What I Learned
- Implementation of a Convolutional Neural Network using TensorFlow and Keras.
- Image data augmentation techniques to improve model generalization.
- Training a model for multi-class classification tasks.
- Evaluation metrics such as accuracy and loss for model performance assessment.
- Saving and loading models in the h5 format for convenience.

### What Makes the Project Stand Out
1. Utilization of data augmentation to enhance the model's ability to generalize to unseen data.
2. Clear visualization of training and validation accuracy/loss curves for model evaluation.
3. Implementation of a prediction function to infer disease categories with confidence percentages.
4. Saving the trained model in a single h5 file for easy deployment and sharing.

### Features
- Data augmentation using Keras ImageDataGenerator.
- Splitting the dataset into training, validation, and test sets.
- CNN architecture with multiple convolutional and pooling layers.
- Training the model with informative visualizations of accuracy and loss curves.
- Prediction function for real-time inference on sample images.
- Saving the trained model for future use.

### How to Install and Run the Project
1. Clone the repository to your local machine.
2. Install the required libraries: `tensorflow`, `numpy`, `matplotlib`, `split-folders`.
    ```bash
    pip install tensorflow numpy matplotlib split-folders
    ```
3. Run the following commands to split the dataset using `split-folders`.
    ```bash
    pip install split-folders
    splitfolders --ratio 0.8 0.1 0.1 -- ./training/PlantVillage/Tomato_disease_categories
    ```
4. Execute the provided code to train the model and save it.

### Results Obtained
The trained model achieved an accuracy of approximately 94% on the test dataset.

### How to Use the Project
1. Clone the repository to your local machine.
2. Install the required libraries as mentioned in the "How to Install and Run the Project" section.
3. Run the provided code to train the model.
4. Use the trained model for inference on new images using the prediction function.
5. Visualize the accuracy and loss curves to evaluate the model's performance.

Feel free to explore and modify the project based on your specific needs!
