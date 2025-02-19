# Fatty Liver Classification using Deep Learning and Radiomics

A hybrid machine learning framework for classifying Non-alcoholic Fatty Liver Disease (NAFLD) severity from ultrasound images. This implementation combines deep learning features from multiple pre-trained CNNs with radiomic features to achieve high-accuracy classification of liver fat grades.

## Key Features
- Multi-model deep learning feature extraction using 7 pre-trained CNNs:
  - VGG19
  - ResNet-101
  - MobileNet
  - DenseNet-121
  - Inception-v3
  - Xception
  - EfficientNet-B7
- GLCM-based radiomic feature extraction
- Advanced feature selection using mRMR, ANOVA, and Mutual Information
- Multiple classifier implementations (XGBoost, LightGBM, LDA)
- Comprehensive evaluation metrics and cross-validation

## Performance
- Accuracy: 97.19% ± 1.24%
- Sensitivity: 96.39% ± 0.66%
- Precision: 97.44% ± 0.50%
- F1-score: 96.91% ± 0.59%

## Requirements
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- XGBoost
- LightGBM
- NumPy
- Pandas

## Dataset
The implementation is based on a dataset of 550 ultrasound images from 55 patients, categorizing liver conditions into four grades:
- Healthy liver (<5% fat)
- Low fat (5-30%)
- Medium fat (30-70%)
- High fat (>70%)

## Usage
```python
from fatty_liver_classifier import FattyLiverClassifier

# Initialize the classifier
classifier = FattyLiverClassifier()

# Train and evaluate
classifier.train(X_train, y_train)
metrics = classifier.evaluate(X_test, y_test)
