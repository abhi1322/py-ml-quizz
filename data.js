export const quizData = [
  {
    question: "Which of the following is a mutable data type in Python?",
    options: ["int", "float", "tuple", "list"],
    correctAnswer: "list",
  },
  {
    question:
      "What is the output of the following code?\n\n```python\nx = 10\ny = 5\nprint(x // y)\n```",
    options: ["2.0", "2", "1.5", "2.5"],
    correctAnswer: "2",
  },
  {
    question:
      "Which of the following loops will execute at least once even if the condition is false initially?",
    options: ["for loop", "while loop", "do-while loop", "None of the above"],
    correctAnswer: "while loop",
  },
  {
    question:
      "What will be the output of the following code snippet?\n\n```python\nfor i in range(5):\n    if i == 3:\n        break\n    print(i)\n```",
    options: ["0 1 2", "0 1 2 3", "0 1 2 3 4", "None of the above"],
    correctAnswer: "0 1 2",
  },
  {
    question: "What is the correct syntax for defining a function in Python?",
    options: [
      "def functionName:",
      "function functionName():",
      "def functionName():",
      "function functionName:",
    ],
    correctAnswer: "def functionName():",
  },
  {
    question: "Which of the following statements is true about Python modules?",
    options: [
      "Modules can only be imported once in a script",
      "A module can contain variables, functions, and classes",
      "Modules must be written in the same script",
      "All of the above",
    ],
    correctAnswer: "A module can contain variables, functions, and classes",
  },
  {
    question:
      "Which method is used to read the entire content of a file as a string in Python?",
    options: ["readline()", "read()", "readlines()", "readall()"],
    correctAnswer: "read()",
  },
  {
    question:
      "What will the following code do?\n\n```python\nwith open('file.txt', 'w') as f:\n    f.write('Hello, World!')\n```",
    options: [
      "Append 'Hello, World!' to 'file.txt'",
      "Read 'Hello, World!' from 'file.txt'",
      "Write 'Hello, World!' to 'file.txt', overwriting existing content",
      "Write 'Hello, World!' to 'file.txt' without overwriting existing content",
    ],
    correctAnswer:
      "Write 'Hello, World!' to 'file.txt', overwriting existing content",
  },
  {
    question: "Which of the following is not a type of machine learning?",
    options: [
      "Supervised Learning",
      "Unsupervised Learning",
      "Reinforcement Learning",
      "Predictive Learning",
    ],
    correctAnswer: "Predictive Learning",
  },
  {
    question:
      "What is the primary difference between supervised and unsupervised learning?",
    options: [
      "Supervised learning uses labeled data while unsupervised learning uses unlabeled data",
      "Supervised learning uses clustering while unsupervised learning uses regression",
      "Supervised learning uses dimensionality reduction while unsupervised learning uses feature engineering",
      "There is no difference",
    ],
    correctAnswer:
      "Supervised learning uses labeled data while unsupervised learning uses unlabeled data",
  },
  {
    question:
      "Which of the following techniques is used to handle missing data?",
    options: ["Normalization", "Standardization", "Imputation", "Encoding"],
    correctAnswer: "Imputation",
  },
  {
    question: "What is feature scaling?",
    options: [
      "Adding new features to the dataset",
      "Combining multiple features into one",
      "Transforming features to a similar scale",
      "Removing irrelevant features",
    ],
    correctAnswer: "Transforming features to a similar scale",
  },
  {
    question:
      "Which of the following algorithms is used for feature selection in machine learning?",
    options: ["K-means", "Decision Tree", "Random Forest", "PCA"],
    correctAnswer: "Random Forest",
  },
  {
    question:
      "What does the following PyTorch code do?\n\n```python\nimport torch\nx = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\ny = x + 2\ny.backward(torch.tensor([1.0, 1.0, 1.0]))\n```",
    options: [
      "Updates the value of `x`",
      "Computes the gradient of `y` with respect to `x`",
      "Converts `x` to a NumPy array",
      "None of the above",
    ],
    correctAnswer: "Computes the gradient of `y` with respect to `x`",
  },
  {
    question: "Which of the following is a characteristic of time-series data?",
    options: [
      "Data points are independent of time",
      "Data points are collected at different time intervals",
      "Data points are sequentially dependent",
      "Data points are categorical",
    ],
    correctAnswer: "Data points are sequentially dependent",
  },
  {
    question: "Which model is commonly used for time-series forecasting?",
    options: ["ARIMA", "K-means", "Random Forest", "SVM"],
    correctAnswer: "ARIMA",
  },
  {
    question:
      "Which activation function is commonly used in neural networks for binary classification?",
    options: ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    correctAnswer: "Sigmoid",
  },
  {
    question:
      "What is the purpose of regularization techniques in machine learning?",
    options: [
      "To reduce overfitting",
      "To increase the model complexity",
      "To speed up training",
      "To decrease the model performance",
    ],
    correctAnswer: "To reduce overfitting",
  },
  {
    question:
      "Which of the following evaluation metrics is used for regression tasks?",
    options: ["Precision", "Recall", "F1-score", "Mean Squared Error"],
    correctAnswer: "Mean Squared Error",
  },
  {
    question:
      "What is the purpose of dropout regularization in neural networks?",
    options: [
      "To randomly remove neurons during training",
      "To add noise to the input data",
      "To increase the learning rate",
      "To decrease the model complexity",
    ],
    correctAnswer: "To randomly remove neurons during training",
  },
  {
    question:
      "Which of the following is used to prevent gradient vanishing or exploding in deep neural networks?",
    options: ["Batch Normalization", "Dropout", "ReLU", "LSTM"],
    correctAnswer: "Batch Normalization",
  },
  {
    question: "Which algorithm is commonly used for text classification tasks?",
    options: ["K-means", "Naive Bayes", "PCA", "Random Forest"],
    correctAnswer: "Naive Bayes",
  },
  {
    question:
      "What is the purpose of hyperparameter tuning in machine learning?",
    options: [
      "To adjust the model parameters during training",
      "To optimize the learning rate",
      "To select the best model architecture",
      "To preprocess the input data",
    ],
    correctAnswer: "To select the best model architecture",
  },
  {
    question:
      "Which of the following is not a hyperparameter of a neural network?",
    options: [
      "Learning rate",
      "Number of hidden layers",
      "Number of epochs",
      "Feature dimension",
    ],
    correctAnswer: "Feature dimension",
  },
  {
    question: "Which algorithm is commonly used for anomaly detection?",
    options: [
      "Linear Regression",
      "K-means",
      "Random Forest",
      "Isolation Forest",
    ],
    correctAnswer: "Isolation Forest",
  },
  {
    question:
      "What is the primary goal of data preprocessing in machine learning?",
    options: [
      "To reduce the number of features",
      "To normalize the data",
      "To increase the model complexity",
      "To prepare the data for training",
    ],
    correctAnswer: "To prepare the data for training",
  },
  {
    question:
      "Which of the following techniques is used to handle imbalanced datasets?",
    options: [
      "Random Oversampling",
      "Random Undersampling",
      "SMOTE",
      "All of the above",
    ],
    correctAnswer: "All of the above",
  },
  {
    question:
      "Which loss function is commonly used for binary classification in neural networks?",
    options: [
      "Mean Squared Error",
      "Cross-Entropy Loss",
      "Kullback-Leibler Divergence",
      "Huber Loss",
    ],
    correctAnswer: "Cross-Entropy Loss",
  },
  {
    question:
      "What is the purpose of early stopping in training neural networks?",
    options: [
      "To stop training when the validation loss starts increasing",
      "To stop training when the training loss reaches zero",
      "To speed up the training process",
      "To prevent overfitting",
    ],
    correctAnswer:
      "To stop training when the validation loss starts increasing",
  },
  {
    question: "Which of the following techniques is used for feature scaling?",
    options: [
      "Normalization",
      "Standardization",
      "Min-Max Scaling",
      "All of the above",
    ],
    correctAnswer: "All of the above",
  },
  {
    question:
      "What is the primary purpose of cross-validation in machine learning?",
    options: [
      "To reduce overfitting",
      "To evaluate the model performance",
      "To preprocess the data",
      "To select the best hyperparameters",
    ],
    correctAnswer: "To evaluate the model performance",
  },
  {
    question: "Which of the following is a dimensionality reduction technique?",
    options: ["PCA", "K-means", "Random Forest", "Decision Tree"],
    correctAnswer: "PCA",
  },
  {
    question:
      "Which of the following algorithms is used for sequence generation tasks?",
    options: ["K-means", "Decision Tree", "LSTM", "Random Forest"],
    correctAnswer: "LSTM",
  },
  {
    question:
      "What is the primary advantage of using mini-batch gradient descent over batch gradient descent?",
    options: [
      "Faster convergence",
      "Less memory usage",
      "Avoidance of local minima",
      "Improved generalization",
    ],
    correctAnswer: "Less memory usage",
  },
  {
    question:
      "Which of the following activation functions is commonly used for hidden layers in neural networks?",
    options: ["Sigmoid", "Softmax", "ReLU", "Tanh"],
    correctAnswer: "ReLU",
  },
  {
    question:
      "What is the purpose of padding in convolutional neural networks (CNNs)?",
    options: [
      "To reduce the number of parameters",
      "To improve the accuracy of predictions",
      "To handle varying input sizes",
      "To speed up training",
    ],
    correctAnswer: "To handle varying input sizes",
  },
  {
    question:
      "Which technique is used to combat overfitting in neural networks?",
    options: [
      "Dropout",
      "Batch Normalization",
      "Weight initialization",
      "Data augmentation",
    ],
    correctAnswer: "Dropout",
  },
  {
    question:
      "What is the primary advantage of using ensemble methods in machine learning?",
    options: [
      "Reduced bias",
      "Reduced variance",
      "Faster training",
      "Improved interpretability",
    ],
    correctAnswer: "Reduced variance",
  },
  {
    question:
      "Which of the following evaluation metrics is used for multi-class classification?",
    options: ["Precision", "Recall", "F1-score", "Mean Absolute Error"],
    correctAnswer: "F1-score",
  },
  {
    question: "What is the purpose of the softmax function in neural networks?",
    options: [
      "To handle numerical instability",
      "To calculate the mean squared error",
      "To compute class probabilities",
      "To initialize the weights",
    ],
    correctAnswer: "To compute class probabilities",
  },
  {
    question:
      "Which of the following optimization algorithms is commonly used for training deep neural networks?",
    options: [
      "Gradient Descent",
      "Adam",
      "Stochastic Gradient Descent",
      "RMSprop",
    ],
    correctAnswer: "Adam",
  },
  {
    question: "What is the primary goal of data augmentation in deep learning?",
    options: [
      "To increase the training set size",
      "To reduce the training time",
      "To prevent overfitting",
      "To improve model generalization",
    ],
    correctAnswer: "To improve model generalization",
  },
  {
    question:
      "Which technique is used to preprocess text data before feeding it into a neural network?",
    options: [
      "Stemming",
      "Dimensionality reduction",
      "One-hot encoding",
      "Normalization",
    ],
    correctAnswer: "One-hot encoding",
  },
  {
    question:
      "What is the primary advantage of using batch normalization in neural networks?",
    options: [
      "Improved convergence",
      "Reduced memory usage",
      "Stabilization of training",
      "Increased model complexity",
    ],
    correctAnswer: "Stabilization of training",
  },
  {
    question:
      "Which of the following loss functions is used for regression tasks with outliers?",
    options: [
      "Mean Squared Error",
      "Mean Absolute Error",
      "Huber Loss",
      "Cross-Entropy Loss",
    ],
    correctAnswer: "Huber Loss",
  },
  {
    question:
      "What is the primary purpose of the learning rate scheduler in neural networks?",
    options: [
      "To adjust the learning rate during training",
      "To prevent overfitting",
      "To regularize the model",
      "To compute class probabilities",
    ],
    correctAnswer: "To adjust the learning rate during training",
  },
  {
    question:
      "Which technique is used to handle categorical variables in machine learning?",
    options: [
      "Label Encoding",
      "Min-Max Scaling",
      "Principal Component Analysis",
      "Feature Scaling",
    ],
    correctAnswer: "Label Encoding",
  },
  {
    question:
      "What is the purpose of the residual connections in deep neural networks?",
    options: [
      "To improve model interpretability",
      "To reduce overfitting",
      "To facilitate gradient flow",
      "To increase the model complexity",
    ],
    correctAnswer: "To facilitate gradient flow",
  },
  {
    question:
      "Which of the following techniques is used for dimensionality reduction in high-dimensional datasets?",
    options: ["PCA", "K-means", "Random Forest", "Support Vector Machines"],
    correctAnswer: "PCA",
  },
  {
    question:
      "What is the primary advantage of using transfer learning in deep learning?",
    options: [
      "Reduced training time",
      "Improved model accuracy",
      "Increased model complexity",
      "Enhanced generalization",
    ],
    correctAnswer: "Reduced training time",
  },
  {
    question:
      "Which of the following algorithms is commonly used for clustering?",
    options: [
      "Linear Regression",
      "K-means",
      "Decision Tree",
      "Logistic Regression",
    ],
    correctAnswer: "K-means",
  },
  {
    question: "What is the purpose of One-Hot Encoding in machine learning?",
    options: [
      "To convert categorical variables into numerical format",
      "To standardize the data",
      "To reduce the dimensionality of the data",
      "To remove outliers from the data",
    ],
    correctAnswer: "To convert categorical variables into numerical format",
  },
  {
    question: "Which of the following is a deep learning framework?",
    options: ["Scikit-learn", "TensorFlow", "NumPy", "Matplotlib"],
    correctAnswer: "TensorFlow",
  },
  {
    question:
      "What is the primary role of an activation function in a neural network?",
    options: [
      "To normalize the input data",
      "To initialize the weights of the network",
      "To introduce non-linearity into the network",
      "To compute the gradient during backpropagation",
    ],
    correctAnswer: "To introduce non-linearity into the network",
  },
  {
    question:
      "Which of the following is a hyperparameter of the K-nearest neighbors algorithm?",
    options: [
      "Number of neighbors",
      "Learning rate",
      "Number of epochs",
      "Activation function",
    ],
    correctAnswer: "Number of neighbors",
  },
  {
    question:
      "What is the purpose of regularization techniques in machine learning?",
    options: [
      "To prevent underfitting",
      "To increase the complexity of the model",
      "To speed up training",
      "To reduce overfitting",
    ],
    correctAnswer: "To reduce overfitting",
  },
  {
    question:
      "Which of the following is used to evaluate the performance of a classification model?",
    options: [
      "Accuracy",
      "Mean Squared Error",
      "R-squared",
      "Root Mean Squared Error",
    ],
    correctAnswer: "Accuracy",
  },
  {
    question:
      "What is the primary goal of data preprocessing in machine learning?",
    options: [
      "To reduce the dimensionality of the data",
      "To increase the complexity of the model",
      "To prepare the data for analysis",
      "To visualize the data",
    ],
    correctAnswer: "To prepare the data for analysis",
  },
  {
    question:
      "Which of the following techniques is used to handle missing data?",
    options: [
      "Imputation",
      "Normalization",
      "Standardization",
      "Feature scaling",
    ],
    correctAnswer: "Imputation",
  },
  {
    question:
      "What is the primary purpose of cross-validation in machine learning?",
    options: [
      "To reduce overfitting",
      "To evaluate the model performance",
      "To preprocess the data",
      "To select the best hyperparameters",
    ],
    correctAnswer: "To evaluate the model performance",
  },
  {
    question: "Which of the following is a characteristic of time-series data?",
    options: [
      "Data points are independent of time",
      "Data points are collected at different time intervals",
      "Data points are sequentially dependent",
      "Data points are categorical",
    ],
    correctAnswer: "Data points are sequentially dependent",
  },
  {
    question: "Which model is commonly used for time-series forecasting?",
    options: ["ARIMA", "K-means", "Random Forest", "SVM"],
    correctAnswer: "ARIMA",
  },
  {
    question:
      "Which activation function is commonly used in neural networks for binary classification?",
    options: ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    correctAnswer: "Sigmoid",
  },
  {
    question:
      "What is the purpose of hyperparameter tuning in machine learning?",
    options: [
      "To adjust the model parameters during training",
      "To optimize the learning rate",
      "To select the best model architecture",
      "To preprocess the input data",
    ],
    correctAnswer: "To select the best model architecture",
  },
  {
    question:
      "Which of the following is not a hyperparameter of a neural network?",
    options: [
      "Learning rate",
      "Number of hidden layers",
      "Number of epochs",
      "Feature dimension",
    ],
    correctAnswer: "Feature dimension",
  },
  {
    question: "Which algorithm is commonly used for anomaly detection?",
    options: [
      "Linear Regression",
      "K-means",
      "Random Forest",
      "Isolation Forest",
    ],
    correctAnswer: "Isolation Forest",
  },
  {
    question:
      "What is the primary goal of data preprocessing in machine learning?",
    options: [
      "To reduce the number of features",
      "To normalize the data",
      "To increase the model complexity",
      "To prepare the data for training",
    ],
    correctAnswer: "To prepare the data for training",
  },
];
