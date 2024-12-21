# Handwritten-text-recogintion-using-Eminst-dataset-and-KNN
Here we are recognizing handwritten using Machine Learning .
This repository demonstrates the classification of handwritten digits from the MNIST dataset using several machine learning and deep learning models. The models include:

K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Logistic Regression
Decision Tree Classifier
Convolutional Neural Network (CNN)
Prerequisites
Make sure you have the necessary libraries installed:

pip install tensorflow scikit-learn matplotlib seaborn nump
Dataset
The dataset used for classification is MNIST, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). The images are 28x28 grayscale, and each image is labeled with the corresponding digit.

The dataset is loaded using the keras.datasets.mnist.load_data() function, and the input data is preprocessed to fit the input format of each model.

Overview of the Code
The code demonstrates the following steps:

Data Preprocessing
The MNIST images are reshaped and normalized for different models. The images are converted to 28x28 pixel arrays (flattened for non-CNN models) and scaled to the range [0, 1].

Model Training & Evaluation
Different machine learning models are trained on the MNIST dataset, and their performances are evaluated using accuracy and confusion matrix visualizations.

1. K-Nearest Neighbors (KNN)
The KNN classifier is trained on the MNIST dataset with two variations of the hyperparameter k:

k=5
k=3 (default)
The model's performance is evaluated by printing the test accuracy and displaying the confusion matrix to show where the model made correct and incorrect predictions.

knn = KNeighborsClassifier(n_neighbors=5)
A confusion matrix is plotted using Seaborn's heatmap for visualization.

2. Support Vector Machine (SVM)
Two variations of the SVM model are tested using a linear kernel and different values of the C hyperparameter:

C=1.0 (default)
C=5.0 (variation)
The SVM classifier is trained with a subset of the training data, and its performance is evaluated using accuracy and a confusion matrix.
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

3. Logistic Regression
The Logistic Regression model is trained on the flattened images of the MNIST dataset. The model is evaluated using accuracy and a confusion matrix.

logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train_flat, y_train)
y_pred_logreg = logreg.predict(x_test_flat)

4. Decision Tree Classifier
The Decision Tree classifier is used to classify the MNIST digits, and its accuracy is evaluated. A confusion matrix is also displayed to visualize the model's performance.

5. Convolutional Neural Network (CNN)
A simple CNN architecture is built using Keras for classifying MNIST digits. The model includes:

Conv2D layers for feature extraction
MaxPooling2D layers for downsampling
Dropout layer (rate=0.6) to prevent overfitting
Dense layer with 10 units for classification
The model is trained for 5 epochs with a batch size of 128, and the performance is evaluated using test accuracy and a confusion matrix.



model.c
Results Visualization
For each model, the test accuracy is printed, followed by a confusion matrix visualizing the classification performance. The confusion matrix is generated using sklearn.metrics.confusion_matrix and visualized with Seaborn.


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
The confusion matrix helps us identify where the model makes errors by comparing the predicted labels with the true labels.

Example of the Output:
Test accuracy: Displays the accuracy of the model on the test set.
Confusion Matrix: Visual representation of how well the model predicted the digits, with the actual digits on the y-axis and predicted digits on the x-axis.
Conclusion
This repository shows how different machine learning models (KNN, SVM, Logistic Regression, Decision Trees) and deep learning models (CNN) can be used for handwritten digit classification on the MNIST dataset. Each model's performance is evaluated, and confusion matrices are used to further analyze the classification results.

How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
Install the required dependencies:

pip install -r requirements.txt
Run the code in the provided Jupyter Notebook or as individual Python scripts.

References
MNIST dataset
Keras Documentation
Scikit-learn Documentation
