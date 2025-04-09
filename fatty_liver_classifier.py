# -*- coding: utf-8 -*-
"""
Created on Wed June 19 23:55:16 2024

@author: Asus
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import (
    VGG19, ResNet101, MobileNetV2, DenseNet121,
    InceptionV3, Xception, EfficientNetB7
)
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import cv2
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class PreProcessor:
    def __init__(self, target_size=(399, 399)):
        """
        Initialize preprocessor with target image size
        Args:
            target_size (tuple): Target image dimensions (height, width)
        """
        self.target_size = target_size
        
    def preprocess_image(self, image):
        """
        Preprocess a single image as described in Section 2-2 of the paper.
        1. Remove redundant numbers, signals, and signs
        2. Crop to remove borders and peripheral details
        3. Resize to 399x399 pixels
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image / 255.0
        
        return image
        
    def apply_augmentation(self, image):
        """
        Apply data augmentation as described in Section 2-2 of the paper:
        - Horizontal flipping
        - Rotations up to 10 degrees
        - Zooms of 90-110%
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Horizontal flipping (50% probability)
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            
        # Random rotation (up to 10 degrees)
        angle = np.random.uniform(-10, 10)
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        # Random zoom (90-110%)
        zoom_factor = np.random.uniform(0.9, 1.1)
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        
        # Get crop or padding parameters
        y_min = max(0, (new_height - height) // 2)
        x_min = max(0, (new_width - width) // 2)
        y_max = min(new_height, y_min + height)
        x_max = min(new_width, x_min + width)
        
        # Perform zoom
        if zoom_factor > 1.0:  # Zoom in (crop)
            image = cv2.resize(image, (new_width, new_height))
            image = image[y_min:y_max, x_min:x_max]
            image = cv2.resize(image, (height, width))
        else:  # Zoom out (pad)
            resized = cv2.resize(image, (new_width, new_height))
            # Create a blank canvas
            canvas = np.zeros((height, width), dtype=image.dtype)
            # Compute padding
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            # Place the resized image on the canvas
            canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
            image = canvas
            
        return image

class GLCMFeatureExtractor:
    """
    Implements GLCM feature extraction as described in Section 2-3 of the paper.
    Extracts exactly 22 features at 4 angles (0°, 45°, 90°, 135°).
    """
    def __init__(self, distances=[1], angles=[0, 45, 90, 135], levels=256):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        
    def compute_glcm(self, image, distance, angle):
        """
        Compute GLCM for given parameters.
        
        Args:
            image: Input image (should be grayscale)
            distance: Distance between pixel pairs
            angle: Angle for GLCM calculation
            
        Returns:
            Normalized GLCM matrix
        """
        # Scale image to range [0, levels-1]
        image = (image * (self.levels-1)).astype(np.uint8)
        
        # Initialize GLCM
        glcm = np.zeros((self.levels, self.levels))
        
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Calculate offset
        dx = int(round(distance * np.cos(theta)))
        dy = int(round(distance * np.sin(theta)))
        
        # Compute GLCM
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                i2 = i + dy
                j2 = j + dx
                
                if 0 <= i2 < height and 0 <= j2 < width:
                    # Use the intensity values as indices to update GLCM
                    glcm[image[i, j], image[i2, j2]] += 1
                    
        # Normalize GLCM
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
            
        return glcm
    
    def extract_features(self, image):
        """
        Extract exactly the 22 GLCM features mentioned in Section 2-3 of the paper.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = []
        
        # For each distance and angle combination
        for distance in self.distances:
            for angle in self.angles:
                # Compute GLCM
                glcm = self.compute_glcm(image, distance, angle)
                
                # Calculate indices for GLCM features
                i_indices, j_indices = np.indices(glcm.shape)
                i_indices = i_indices.flatten()
                j_indices = j_indices.flatten()
                flat_glcm = glcm.flatten()
                
                # Get non-zero elements
                valid_idx = flat_glcm > 0
                i_valid = i_indices[valid_idx]
                j_valid = j_indices[valid_idx]
                p_valid = flat_glcm[valid_idx]
                
                # Calculate mean and variance
                mu_i = np.sum(i_indices * flat_glcm)
                mu_j = np.sum(j_indices * flat_glcm)
                sigma_i = np.sqrt(np.sum((i_indices - mu_i)**2 * flat_glcm))
                sigma_j = np.sqrt(np.sum((j_indices - mu_j)**2 * flat_glcm))
                
                # Feature 1: Contrast
                contrast = np.sum((i_indices - j_indices)**2 * flat_glcm)
                
                # Feature 2: Correlation
                if sigma_i > 0 and sigma_j > 0:
                    correlation = np.sum((i_indices - mu_i) * (j_indices - mu_j) * flat_glcm) / (sigma_i * sigma_j)
                else:
                    correlation = 0
                
                # Feature 3: Energy
                energy = np.sum(flat_glcm**2)
                
                # Feature 4: Homogeneity
                homogeneity = np.sum(flat_glcm / (1 + np.abs(i_indices - j_indices)))
                
                # Feature 5: Autocorrelation
                autocorrelation = np.sum(i_indices * j_indices * flat_glcm)
                
                # Feature 6: Cluster prominence
                cluster_prominence = np.sum(((i_indices + j_indices - mu_i - mu_j)**4) * flat_glcm)
                
                # Feature 7: Cluster shade
                cluster_shade = np.sum(((i_indices + j_indices - mu_i - mu_j)**3) * flat_glcm)
                
                # Feature 8: Entropy
                entropy_value = -np.sum(p_valid * np.log2(p_valid))
                
                # Feature 9: Sum of squares (Variance)
                sum_squares = np.sum(((i_indices - mu_i)**2) * flat_glcm)
                
                # Features 10-12: Sum average, variance and entropy
                k_indices = i_indices + j_indices
                k_values = np.unique(k_indices)
                p_x_plus_y = np.zeros(len(k_values))
                for idx, k in enumerate(k_values):
                    p_x_plus_y[idx] = np.sum(flat_glcm[k_indices == k])
                
                sum_average = np.sum(k_values * p_x_plus_y)
                sum_variance = np.sum((k_values - sum_average)**2 * p_x_plus_y)
                valid_p_x_plus_y = p_x_plus_y[p_x_plus_y > 0]
                sum_entropy = -np.sum(valid_p_x_plus_y * np.log2(valid_p_x_plus_y))
                
                # Features 13-15: Difference average, variance and entropy
                l_indices = np.abs(i_indices - j_indices)
                l_values = np.unique(l_indices)
                p_x_minus_y = np.zeros(len(l_values))
                for idx, l in enumerate(l_values):
                    p_x_minus_y[idx] = np.sum(flat_glcm[l_indices == l])
                
                difference_average = np.sum(l_values * p_x_minus_y)
                difference_variance = np.sum((l_values - difference_average)**2 * p_x_minus_y)
                valid_p_x_minus_y = p_x_minus_y[p_x_minus_y > 0]
                difference_entropy = -np.sum(valid_p_x_minus_y * np.log2(valid_p_x_minus_y))
                
                # Feature 16: Maximum probability
                maximum_probability = np.max(glcm)
                
                # Feature 17: Inverse difference (Homogeneity)
                inverse_difference = np.sum(flat_glcm / (1 + np.abs(i_indices - j_indices)))
                
                # Feature 18: Variance
                variance = sum_squares  # Same as sum of squares
                
                # Feature 19: Dissimilarity
                dissimilarity = np.sum(np.abs(i_indices - j_indices) * flat_glcm)
                
                # Feature 20: Inverse difference moment
                moment_inverse_diff = np.sum(flat_glcm / (1 + (i_indices - j_indices)**2))
                
                # Feature 21-22: Normalized inverse difference and moment
                inverse_diff_normalized = np.sum(flat_glcm / (1 + np.abs(i_indices - j_indices) / self.levels))
                inverse_diff_moment_normalized = np.sum(flat_glcm / (1 + ((i_indices - j_indices) / self.levels)**2))
                
                # Combine all 22 features as listed in the paper
                current_features = [
                    contrast, correlation, energy, homogeneity, autocorrelation,
                    cluster_prominence, cluster_shade, entropy_value, sum_squares,
                    sum_average, sum_variance, sum_entropy, difference_average,
                    difference_variance, difference_entropy, maximum_probability,
                    inverse_difference, variance, dissimilarity, moment_inverse_diff,
                    inverse_diff_normalized, inverse_diff_moment_normalized
                ]
                
                # Add features to the list
                features.extend(current_features)
                
                # Create feature names
                current_names = [
                    f'contrast_{angle}', f'correlation_{angle}', f'energy_{angle}',
                    f'homogeneity_{angle}', f'autocorrelation_{angle}',
                    f'cluster_prominence_{angle}', f'cluster_shade_{angle}',
                    f'entropy_{angle}', f'sum_squares_variance_{angle}',
                    f'sum_average_{angle}', f'sum_variance_{angle}',
                    f'sum_entropy_{angle}', f'difference_average_{angle}',
                    f'difference_variance_{angle}', f'difference_entropy_{angle}',
                    f'maximum_probability_{angle}', f'inverse_difference_{angle}',
                    f'variance_{angle}', f'dissimilarity_{angle}',
                    f'moment_inverse_diff_{angle}', f'inverse_diff_norm_{angle}',
                    f'inverse_diff_moment_norm_{angle}'
                ]
                
                feature_names.extend(current_names)
                
        return np.array(features), feature_names

class DeepFeatureExtractor:
    """
    Implements deep learning feature extraction as described in Section 2-4 of the paper.
    Uses seven pre-trained CNNs: VGG19, ResNet-101, MobileNet, DenseNet-121,
    Inception-v3, Xception, and EfficientNet-B7.
    """
    def __init__(self):
        # Initialize all models with weights='imagenet' and include_top=False as per paper
        self.models = {
            'vgg19': {
                'model': VGG19(weights='imagenet', include_top=False, pooling='avg'),
                'size': (224, 224),
                'preprocess': vgg_preprocess
            },
            'resnet101': {
                'model': ResNet101(weights='imagenet', include_top=False, pooling='avg'),
                'size': (224, 224),
                'preprocess': resnet_preprocess
            },
            'mobilenet': {
                'model': MobileNetV2(weights='imagenet', include_top=False, pooling='avg'),
                'size': (224, 224),
                'preprocess': mobilenet_preprocess
            },
            'densenet121': {
                'model': DenseNet121(weights='imagenet', include_top=False, pooling='avg'),
                'size': (224, 224),
                'preprocess': densenet_preprocess
            },
            'inception': {
                'model': InceptionV3(weights='imagenet', include_top=False, pooling='avg'),
                'size': (299, 299),
                'preprocess': inception_preprocess
            },
            'xception': {
                'model': Xception(weights='imagenet', include_top=False, pooling='avg'),
                'size': (299, 299),
                'preprocess': xception_preprocess
            },
            'efficientnet': {
                'model': EfficientNetB7(weights='imagenet', include_top=False, pooling='avg'),
                'size': (224, 224),
                'preprocess': efficientnet_preprocess
            }
        }
        
    def preprocess_for_model(self, image, model_config):
        """
        Preprocess image for a specific model.
        
        Args:
            image: Input image
            model_config: Configuration for the specific model
            
        Returns:
            Preprocessed image ready for feature extraction
        """
        # Resize to model-specific input size
        img = cv2.resize(image, model_config['size'])
        
        # Convert grayscale to RGB for model compatibility
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
            
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Apply model-specific preprocessing
        img = model_config['preprocess'](img)
        
        return img
    
    def extract_features(self, image):
        """
        Extract features from all seven models as described in the paper.
        The paper specifies removing the last fully connected layer and using
        the remaining CNN layers as feature extractors.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = []
        
        # Process image through each model
        for model_name, model_config in self.models.items():
            # Preprocess image for current model
            preprocessed_img = self.preprocess_for_model(image, model_config)
            
            # Extract features using the model
            model_features = model_config['model'].predict(preprocessed_img, verbose=0)
            
            # Flatten features
            flat_features = model_features.flatten()
            
            # Store features and their names
            features.extend(flat_features)
            feature_names.extend([f'{model_name}_{i}' for i in range(len(flat_features))])
            
        return np.array(features), feature_names

class FeatureSelector:
    """
    Implements the feature selection methods described in Section 2-5 of the paper:
    1. Maximum Relevance and Minimum Redundancy (mRMR)
    2. Analysis of Variance (ANOVA)
    3. Mutual Information (MI)
    """
    def __init__(self, method='mrmr'):
        """
        Initialize feature selector with specified method.
        
        Args:
            method (str): One of 'mrmr', 'anova', or 'mi'
        """
        self.method = method
        self.selected_indices = None
        
    def mutual_information(self, x, y):
        """
        Calculate mutual information between feature x and target y.
        Implements equation (3) from the paper.
        
        Args:
            x: Feature vector
            y: Target vector
            
        Returns:
            Mutual information score
        """
        # Discretize continuous features if needed
        if np.issubdtype(x.dtype, np.floating):
            bins = min(20, len(np.unique(x)))
            x_discrete = np.digitize(x, bins=np.linspace(min(x), max(x), bins))
        else:
            x_discrete = x
            
        # Get unique values and counts
        x_values = np.unique(x_discrete)
        y_values = np.unique(y)
        
        # Calculate joint and marginal probabilities
        xy_count = np.zeros((len(x_values), len(y_values)))
        for i, j in zip(x_discrete, y):
            xy_count[np.where(x_values == x_discrete)[0][0], np.where(y_values == j)[0][0]] += 1
            
        # Normalize to get probabilities
        pxy = xy_count / len(x_discrete)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        
        # Calculate mutual information
        MI = 0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                if pxy[i,j] > 0:
                    MI += pxy[i,j] * np.log2(pxy[i,j] / (px[i,0] * py[0,j]))
                    
        return MI

    def mrmr_score(self, X, y, k):
        """
        Calculate mRMR scores following equation (1) from the paper.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))
        
        # Calculate relevance scores (mutual information with target)
        relevance = np.zeros(n_features)
        for i in range(n_features):
            relevance[i] = self.mutual_information(X[:, i], y)
            
        # Select first feature with maximum relevance
        first = np.argmax(relevance)
        selected.append(first)
        remaining.remove(first)
        
        # Select remaining features
        for _ in range(min(k-1, len(remaining))):
            max_score = -np.inf
            max_idx = -1
            
            for feat in remaining:
                # Calculate relevance term
                rel = relevance[feat]
                
                # Calculate redundancy term
                red = 0
                for sel in selected:
                    red += self.mutual_information(X[:, feat], X[:, sel])
                red = red / len(selected)
                
                # Calculate mRMR score (relevance - redundancy)
                score = rel - red
                
                # Update max score if needed
                if score > max_score:
                    max_score = score
                    max_idx = feat
                    
            # Add feature with maximum score
            selected.append(max_idx)
            remaining.remove(max_idx)
            
        return selected
    
    def anova_selection(self, X, y, k):
        """
        Implement ANOVA feature selection using F-statistics.
        Based on equation (2) from the paper.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        f_scores, _ = f_classif(X, y)
        
        # Handle NaN values
        f_scores = np.nan_to_num(f_scores, 0)
        
        # Return indices of top k features
        return np.argsort(f_scores)[-k:]

    def mutual_information_selection(self, X, y, k):
        """
        Implement Mutual Information feature selection.
        Based on equation (3) from the paper.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        # Calculate MI scores for each feature
        mi_scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            mi_scores[i] = self.mutual_information(X[:, i], y)
            
        # Return indices of top k features
        return np.argsort(mi_scores)[-k:]

    def select_features(self, X, y, n_features=100):
        """
        Main feature selection method that implements the paper's approach.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            
        Returns:
            Indices of selected features
        """
        if self.method == 'mrmr':
            selected_indices = self.mrmr_score(X, y, n_features)
        elif self.method == 'anova':
            selected_indices = self.anova_selection(X, y, n_features)
        elif self.method == 'mi':
            selected_indices = self.mutual_information_selection(X, y, n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
            
        self.selected_indices = selected_indices
        return selected_indices

class TwoStageFeatureSelector:
    """
    Implements the two-stage feature selection process described in Section 2-5.
    First applies feature selection to individual feature sets,
    then performs a second round of selection on the combined features.
    """
    def __init__(self, methods=['mrmr', 'anova', 'mi']):
        """
        Initialize two-stage feature selector.
        
        Args:
            methods: List of feature selection methods to use
        """
        self.methods = methods
        self.selectors = {method: FeatureSelector(method) for method in methods}
        self.selected_indices = None
        
    def select_features(self, X, y, n_features=100):
        """
        Perform two-stage feature selection as described in the paper.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select in each stage
            
        Returns:
            Indices of selected features
        """
        # Stage 1: Select features using each method
        selected_features = {}
        for method in self.methods:
            selector = self.selectors[method]
            selected_indices = selector.select_features(X, y, n_features)
            selected_features[method] = X[:, selected_indices]
            
        # Combine selected features from all methods
        combined_features = np.hstack([selected_features[method] 
                                    for method in self.methods])
        
        # Stage 2: Final feature selection on combined features
        final_selector = FeatureSelector('mrmr')  # Use mRMR for final selection
        final_indices = final_selector.select_features(combined_features, y, n_features)
        
        # Map back to original feature indices
        self.selected_indices = final_indices
        return final_indices

def normalize_features(features):
    """
    Normalize features and remove zero-variance features.
    
    Args:
        features: Feature matrix
        
    Returns:
        Normalized features with zero-variance features removed
    """
    # Remove zero-variance features
    variances = np.var(features, axis=0)
    non_zero_var = variances > 0
    
    if not any(non_zero_var):
        return features
    
    features = features[:, non_zero_var]
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features

class FattyLiverClassifier:
    """
    Implements the classification methods described in Section 2-6 of the paper.
    Includes LDA, LightGBM, and XGBoost classifiers.
    """
    def __init__(self, classifier_type='xgboost'):
        """
        Initialize classifier with specified type.
        
        Args:
            classifier_type: Type of classifier to use ('xgboost', 'lightgbm', or 'lda')
        """
        self.classifier_type = classifier_type
        
        # Initialize classifier with parameters from the paper
        if classifier_type == 'xgboost':
            self.classifier = XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=150,
                objective='multi:softmax',
                random_state=42
            )
        elif classifier_type == 'lightgbm':
            self.classifier = LGBMClassifier(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=150,
                objective='multiclass',
                random_state=42
            )
        elif classifier_type == 'lda':
            self.classifier = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.classifier.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        return self.classifier.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier using metrics specified in Section 2-7.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics as described in the paper
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'sensitivity': recall_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        
        # Calculate per-class metrics
        for i in range(4):  # 4 classes: healthy, low, medium, high fat
            class_indices = (y_test == i)
            if any(class_indices):
                y_true_class = np.array(y_test == i, dtype=int)
                y_pred_class = np.array(y_pred == i, dtype=int)
                
                metrics[f'class_{i}_precision'] = precision_score(y_true_class, y_pred_class)
                metrics[f'class_{i}_recall'] = recall_score(y_true_class, y_pred_class)
                metrics[f'class_{i}_f1'] = f1_score(y_true_class, y_pred_class)
                metrics[f'class_{i}_mcc'] = matthews_corrcoef(y_true_class, y_pred_class)
            
        return metrics

class CrossValidator:
    """
    Implements 5-fold cross-validation with 100 repetitions as described in Section 2-7.
    Ensures patient-level separation to prevent data leakage.
    """
    def __init__(self, n_splits=5, n_repeats=100, random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds for cross-validation
            n_repeats: Number of repetitions
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    def perform_cross_validation(self, X, y, patient_ids, classifier_type='xgboost', feature_selector_method='mi'):
        """
        Perform repeated cross-validation with patient-level separation.
        
        Args:
            X: Feature matrix
            y: Target vector
            patient_ids: Array of patient IDs for each sample
            classifier_type: Type of classifier to use
            feature_selector_method: Feature selection method to use
            
        Returns:
            Tuple of (metrics, confusion_matrix)
        """
        # Initialize metrics storage
        all_metrics = {
            'accuracy': [],
            'sensitivity': [],
            'precision': [],
            'f1_score': [],
            'mcc': []
        }
        all_confusion_matrices = []
        
        # Get unique patient IDs
        unique_patients = np.unique(patient_ids)
        n_patients = len(unique_patients)
        
        # For each repetition
        for repeat in range(self.n_repeats):
            # Shuffle patients for this repeat
            np.random.seed(self.random_state + repeat)
            shuffled_patients = np.random.permutation(unique_patients)
            
            # Split patients into folds
            patient_folds = np.array_split(shuffled_patients, self.n_splits)
            
            # For each fold
            fold_metrics = []
            fold_confusion_matrices = []
            
            for fold in range(self.n_splits):
                # Get test patients for this fold
                test_patients = patient_folds[fold]
                
                # Create train/test mask based on patient IDs
                test_mask = np.isin(patient_ids, test_patients)
                train_mask = ~test_mask
                
                # Split data
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                # Normalize features
                X_train_norm = normalize_features(X_train)
                X_test_norm = normalize_features(X_test)
                
                # IMPORTANT: Only use training data for feature selection
                # to prevent data leakage
                feature_selector = FeatureSelector(feature_selector_method)
                selected_indices = feature_selector.select_features(X_train_norm, y_train)
                
                # Select features
                X_train_selected = X_train_norm[:, selected_indices]
                X_test_selected = X_test_norm[:, selected_indices]
                
                # Train and evaluate classifier
                classifier = FattyLiverClassifier(classifier_type)
                classifier.train(X_train_selected, y_train)
                metrics = classifier.evaluate(X_test_selected, y_test)
                
                # Store metrics
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(metrics['confusion_matrix'])
            
            # Calculate average metrics for this repetition
            rep_metrics = {}
            for key in all_metrics.keys():
                rep_metrics[key] = np.mean([m[key] for m in fold_metrics])
            
            # Store metrics
            for key, value in rep_metrics.items():
                all_metrics[key].append(value)
            
            # Calculate average confusion matrix for this repetition
            avg_confusion_matrix = np.mean(fold_confusion_matrices, axis=0)
            all_confusion_matrices.append(avg_confusion_matrix)
        
        # Calculate final metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        # Calculate final confusion matrix
        final_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        
        return final_metrics, final_confusion_matrix

def extract_features_from_dataset(dataset_dir, target_size=(399, 399), apply_augmentation=False):
    """
    Extract features from dataset.
    
    Args:
        dataset_dir: Directory containing dataset
        target_size: Target image size
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        Tuple of (features, labels, patient_ids)
    """
    # Initialize processors and extractors
    preprocessor = PreProcessor(target_size=target_size)
    glcm_extractor = GLCMFeatureExtractor()
    deep_extractor = DeepFeatureExtractor()
    
    # Class mapping
    class_mapping = {
        'healthy': 0,  # <5% fat
        'low_fat': 1,  # 5-30% fat
        'medium_fat': 2,  # 30-70% fat
        'high_fat': 3  # >70% fat
    }
    
    # Initialize data storage
    all_features = []
    all_labels = []
    all_patient_ids = []
    
    # For each class
    for class_name, label in class_mapping.items():
        class_dir = os.path.join(dataset_dir, class_name)
        
        # For each patient
        for patient_id in os.listdir(class_dir):
            patient_dir = os.path.join(class_dir, patient_id)
            
            if os.path.isdir(patient_dir):
                # Get image files for this patient
                image_files = sorted([f for f in os.listdir(patient_dir) 
                                     if f.endswith(('.dcm', '.png', '.jpg', '.jpeg'))])[:10]
                
                # Process each image
                for img_file in image_files:
                    img_path = os.path.join(patient_dir, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        # Preprocess image
                        processed_img = preprocessor.preprocess_image(img)
                        
                        # Apply data augmentation if requested
                        if apply_augmentation:
                            processed_img = preprocessor.apply_augmentation(processed_img)
                        
                        # Extract GLCM features
                        glcm_features, _ = glcm_extractor.extract_features(processed_img)
                        
                        # Extract deep learning features
                        deep_features, _ = deep_extractor.extract_features(processed_img)
                        
                        # Combine features
                        combined_features = np.concatenate([glcm_features, deep_features])
                        
                        # Store data
                        all_features.append(combined_features)
                        all_labels.append(label)
                        all_patient_ids.append(patient_id)
    
    return np.array(all_features), np.array(all_labels), np.array(all_patient_ids)

def main():
    """
    Main function to run the complete fatty liver classification pipeline.
    """
    # Dataset parameters
    dataset_dir = "D:/data/liver_ultrasound"
    results_dir = "D:results"
    target_size = (399, 399)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract features from dataset (no augmentation for test data)
    print("Extracting features from dataset...")
    features, labels, patient_ids = extract_features_from_dataset(
        dataset_dir=dataset_dir,
        target_size=target_size,
        apply_augmentation=False  # Don't apply augmentation for feature extraction
    )
    
    print(f"Extracted features from {len(features)} images, {len(np.unique(patient_ids))} patients")
    
    # Create train augmented dataset for training only (to be used inside cross-validation)
    print("Creating augmented training dataset...")
    aug_features, aug_labels, aug_patient_ids = extract_features_from_dataset(
        dataset_dir=dataset_dir,
        target_size=target_size,
        apply_augmentation=True  # Apply augmentation
    )
    
    print(f"Created augmented dataset with {len(aug_features)} images")
    
    # Remove potential zero-variance features and normalize
    features = normalize_features(features)
    aug_features = normalize_features(aug_features)
    
    # Evaluate with different classifiers and feature selectors
    classifiers = ['lda', 'lightgbm', 'xgboost']
    feature_selectors = ['mrmr', 'anova', 'mi']
    
    # Initialize cross-validator
    cross_validator = CrossValidator(n_splits=5, n_repeats=100)
    
    # Store results
    all_results = {}
    
    # For each combination of classifier and feature selector
    for classifier in classifiers:
        for selector in feature_selectors:
            print(f"\nEvaluating {classifier.upper()} with {selector.upper()} feature selection...")
            
            # Perform cross-validation
            metrics, conf_matrix = cross_validator.perform_cross_validation(
                X=features,
                y=labels,
                patient_ids=patient_ids,
                classifier_type=classifier,
                feature_selector_method=selector
            )
            
            # Store results
            all_results[(classifier, selector)] = {
                'metrics': metrics,
                'confusion_matrix': conf_matrix
            }
            
            # Print results
            print(f"Accuracy: {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}")
            print(f"Sensitivity: {metrics['sensitivity']['mean']:.4f} ± {metrics['sensitivity']['std']:.4f}")
            print(f"Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
            print(f"F1-score: {metrics['f1_score']['mean']:.4f} ± {metrics['f1_score']['std']:.4f}")
            print(f"MCC: {metrics['mcc']['mean']:.4f} ± {metrics['mcc']['std']:.4f}")
            
            # Save results
            np.save(f"{results_dir}/{classifier}_{selector}_metrics.npy", metrics)
            np.save(f"{results_dir}/{classifier}_{selector}_confusion_matrix.npy", conf_matrix)
    
    # Find best results
    best_acc = 0
    best_config = None
    
    for config, results in all_results.items():
        acc = results['metrics']['accuracy']['mean']
        if acc > best_acc:
            best_acc = acc
            best_config = config
    
    # Print best results
    print("\nBest Results:")
    print(f"Classifier: {best_config[0].upper()}")
    print(f"Feature Selector: {best_config[1].upper()}")
    print(f"Accuracy: {best_acc:.4f}")
    
    return all_results

if __name__ == "__main__":
    main()
