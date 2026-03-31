# ml-classifiers

Implementations of classic machine learning classifiers from scratch using NumPy, evaluated on Fashion-MNIST and a rice classification dataset.

## Models

- **Perceptron** - multi-class perceptron with exponential learning rate decay
- **Logistic Regression** - binary classifier with numerically stable sigmoid and gradient descent
- **Softmax Regression** - multi-class cross-entropy with L2 regularization
- **SVM** - multi-class Weston-Watkins hinge loss with L2 regularization

## Structure

```
├── classification_models.ipynb   # main notebook (training, evaluation, plots)
├── data_process.py               # data loading & preprocessing (Fashion-MNIST, rice CSV)
├── kaggle_submission.py          # Kaggle submission helper
├── logistic.py                   # logistic regression
├── perceptron.py                 # perceptron
├── softmax.py                    # softmax regression
├── svm.py                        # SVM
└── models/                       # same models as an importable package
```

## Usage

Open `classification_models.ipynb` and run cells top to bottom. Data is loaded automatically from the `fashion-mnist/` and `rice/` directories.

## Dependencies

```
numpy pandas scikit-learn matplotlib
```
