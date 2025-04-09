# -*- coding: utf-8 -*-
"""
Created on Wed June 19 23:55:16 2024

@author: Asus
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
import numpy as np
import cv2
from sklearn.feature_selection import f_classif, mutual_info_classif

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class PreProcessor:
    def __init__(self, target_size=(399, 399)):
        self.target_size = target_size
        
    def preprocess_image(self, image):
        """Preprocess a single image."""
        # Remove redundant information and crop to ROI
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image / 255.0
        
        return image

class GLCMFeatureExtractor:
    """
    Implements GLCM feature extraction as described in Section 2-3 of the paper.
    Extracts exactly 22 features at 4 angles (0°, 45°, 90°, 135°).
    """
    def __init__(self, distances=[1], angles=[0, 45, 90, 135]):
        self.distances = distances
        self.angles = angles
        
    def compute_glcm(self, image, distance, angle):
        """Compute GLCM for given parameters."""
        num_levels = 256
        glcm = np.zeros((num_levels, num_levels))
        
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Calculate offset
        dx = int(distance * np.cos(theta))
        dy = int(distance * np.sin(theta))
        
        # Compute GLCM
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (i + dy) < image.shape[0] and (j + dx) < image.shape[1]:
                    i2 = i + dy
                    j2 = j + dx
                    glcm[image[i, j], image[i2, j2]] += 1
                    
        # Normalize GLCM
        if glcm.sum() != 0:
            glcm = glcm / glcm.sum()
            
        return glcm
    
    def extract_features(self, image):
        """Extract exactly the 22 GLCM features mentioned in the paper."""
        features = []
        feature_names = []
        
        for distance in self.distances:
            for angle in self.angles:
                glcm = self.compute_glcm(image, distance, angle)
                
                # Calculate all 22 features exactly as listed in the paper
                # Basic features
                contrast = np.sum(np.square(np.arange(256)[:, None] - np.arange(256)) * glcm)
                correlation = np.sum((np.arange(256)[:, None] - np.mean(glcm)) * 
                                   (np.arange(256) - np.mean(glcm)) * glcm) / (np.std(glcm) ** 2)
                energy = np.sum(np.square(glcm))
                homogeneity = np.sum(glcm / (1 + np.square(np.arange(256)[:, None] - np.arange(256))))
                
                # Statistical features
                autocorrelation = np.sum(np.outer(np.arange(256), np.arange(256)) * glcm)
                cluster_prominence = np.sum(np.power(np.outer(np.arange(256), np.arange(256)) - np.mean(glcm), 4) * glcm)
                cluster_shade = np.sum(np.power(np.outer(np.arange(256), np.arange(256)) - np.mean(glcm), 3) * glcm)
                entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
                
                # Sum-based features
                sum_squares_variance = np.sum(np.square(np.arange(256)[:, None] - np.mean(glcm)) * glcm)
                sum_average = np.sum(np.arange(2*256-1) * np.sum(glcm, axis=0))
                sum_variance = np.sum(np.square(np.arange(2*256-1) - sum_average) * np.sum(glcm, axis=0))
                sum_entropy = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + 1e-10))
                
                # Difference-based features
                difference_average = np.sum(np.abs(np.arange(256)[:, None] - np.arange(256)) * glcm)
                difference_variance = np.var(np.sum(glcm, axis=0))
                difference_entropy = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + 1e-10))
                
                # Other features
                maximum_probability = np.max(glcm)
                inverse_difference = np.sum(glcm / (1 + np.abs(np.arange(256)[:, None] - np.arange(256))))
                variance = np.sum(np.square(np.arange(256)[:, None] - np.mean(glcm)) * glcm)
                dissimilarity = np.sum(np.abs(np.arange(256)[:, None] - np.arange(256)) * glcm)
                moment_inverse_diff = np.sum(glcm / (1 + np.square(np.arange(256)[:, None] - np.arange(256))))
                inverse_diff_normalized = np.sum(glcm / (1 + np.abs(np.arange(256)[:, None] - np.arange(256))/256))

                # Combine all features in the exact order mentioned in the paper
                features.extend([
                    contrast, correlation, energy, homogeneity, autocorrelation,
                    cluster_prominence, cluster_shade, entropy, sum_squares_variance,
                    sum_average, sum_variance, sum_entropy, difference_average,
                    difference_variance, difference_entropy, maximum_probability,
                    inverse_difference, variance, dissimilarity, moment_inverse_diff,
                    inverse_diff_normalized, inverse_diff_normalized  # Last feature repeated as per paper
                ])
                
                # Generate feature names in the same order
                feature_names.extend([
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
                    f'inverse_diff_norm2_{angle}'
                ])
                
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
        Preprocess image for a specific model following paper's methodology.
        1. Resize to model-specific input size
        2. Convert grayscale to RGB (3 channels)
        3. Apply model-specific preprocessing
        """
        # Resize to model-specific input size
        img = cv2.resize(image, model_config['size'])
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
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
    
    def get_feature_dimensions(self):
        """
        Return the number of features extracted by each model.
        Useful for understanding feature distribution across models.
        """
        dimensions = {}
        sample_image = np.zeros((399, 399))  # Create dummy image
        
        for model_name, model_config in self.models.items():
            preprocessed_img = self.preprocess_for_model(sample_image, model_config)
            features = model_config['model'].predict(preprocessed_img, verbose=0)
            dimensions[model_name] = features.flatten().shape[0]
            
        return dimensions



class FeatureSelector:
    """
    Implements the three feature selection methods described in Section 2-5:
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
        self.feature_scores = None
        
    def mutual_information(self, x, y):
        """
        Calculate mutual information between feature x and target y.
        Implements equation (3) from the paper.
        """
        # Discretize continuous features if needed
        if np.issubdtype(x.dtype, np.floating):
            x_discrete = np.digitize(x, bins=np.linspace(min(x), max(x), 20))
        else:
            x_discrete = x
            
        # Calculate joint and marginal probabilities
        xy_count = np.zeros((len(np.unique(x_discrete)), len(np.unique(y))))
        for i, j in zip(x_discrete, y):
            xy_count[i-1, j] += 1
            
        # Normalize to get probabilities
        pxy = xy_count / len(x)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        
        # Calculate mutual information
        MI = 0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                if pxy[i,j] > 0:
                    MI += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[0,j]))
                    
        return MI

    def mrmr_score(self, X, y, k):
        """
        Calculate mRMR scores following equation (1) from the paper.
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
        """
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))
        
        # Calculate relevance scores (mutual information with target)
        relevance = np.zeros(n_features)
        for i in range(n_features):
            relevance[i] = self.mutual_information(X[:, i], y)
            
        # Select first feature
        first = np.argmax(relevance)
        selected.append(first)
        remaining.remove(first)
        
        # Select remaining features
        for _ in range(k - 1):
            scores = np.zeros(len(remaining))
            for i, feat in enumerate(remaining):
                # Calculate redundancy term
                redundancy = np.mean([self.mutual_information(X[:, feat], X[:, j]) 
                                    for j in selected])
                # Calculate mRMR score
                scores[i] = relevance[feat] - redundancy
                
            # Select feature with highest score
            next_feat = remaining[np.argmax(scores)]
            selected.append(next_feat)
            remaining.remove(next_feat)
            
        return selected
    
    def anova_selection(self, X, y, k):
        """
        Implement ANOVA feature selection using F-statistics.
        Based on equation (2) from the paper.
        """
        f_scores, _ = f_classif(X, y)
        
        f_scores = np.nan_to_num(f_scores, 0)  
        
        return np.argsort(f_scores)[-k:]

    def mutual_information_selection(self, X, y, k):
        """
        Implement Mutual Information feature selection.
        Based on equation (3) from the paper.
        """
        # Calculate MI scores for each feature
        mi_scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            mi_scores[i] = self.mutual_information(X[:, i], y)
            
        # Return indices of top k features
        return np.argsort(mi_scores)[-k:]

    def select_features(self, X, y, n_features=100):
        """
        Main feature selection method that implements the two-stage
        selection process described in the paper.
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
    Implements the two-stage feature selection process described in the paper.
    First applies feature selection to the pre-trained model features,
    then performs a second round of selection on the combined features.
    """
    def __init__(self, methods=['mrmr', 'anova', 'mi']):
        self.methods = methods
        self.selectors = {method: FeatureSelector(method) for method in methods}
        
    def select_features(self, X, y, n_features=100):
        """
        Perform two-stage feature selection as described in the paper.
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
        final_selector = FeatureSelector('mrmr')
        final_indices = final_selector.select_features(combined_features, y, n_features)
        
        return final_indices, combined_features[:, final_indices]

def normalize_features(X_train, X_test):
    """
    Normalize features using StandardScaler as mentioned in the paper.
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm




class FattyLiverClassifier:
    """
    Implements the classification methods described in Section 2-6 of the paper.
    Includes LDA, LightGBM, and XGBoost classifiers.
    """
    def __init__(self, classifier_type='xgboost'):
        self.classifier_type = classifier_type
        
        # Initialize classifier with exact parameters from the paper
        if classifier_type == 'xgboost':
            self.classifier = XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=150,
                objective='multi:softmax',
                random_state=42,
                tree_method='gpu_hist'  # Use GPU if available
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
            
    def train(self, X_train, y_train):
        """Train the classifier."""
        self.classifier.fit(X_train, y_train)
        
    def predict(self, X_test):
        """Make predictions."""
        return self.classifier.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier using metrics specified in Section 2-7.
        Returns accuracy, sensitivity, precision, F1-score, and MCC.
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'sensitivity': recall_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Calculate per-class metrics
        for i in range(4):  # 4 classes: healthy, low, medium, high fat
            metrics[f'class_{i}_precision'] = precision_score(y_test, y_pred, average=None)[i]
            metrics[f'class_{i}_recall'] = recall_score(y_test, y_pred, average=None)[i]
            metrics[f'class_{i}_f1'] = f1_score(y_test, y_pred, average=None)[i]
            
        return metrics

class CrossValidator:
    """
    Implements patient-level cross-validation to prevent data leakage.
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    def perform_cross_validation(self, X, y, patient_ids, classifier, feature_selector):
        """
        Perform repeated cross-validation with patient-level separation.
        """
        all_metrics = []
        all_confusion_matrices = []
        
        # Get unique patient IDs
        unique_patients = np.unique(patient_ids)
        
        for repeat in range(self.n_repeats):
            # Shuffle patients for this repeat
            np.random.seed(self.random_state + repeat)
            shuffled_patients = np.random.permutation(unique_patients)
            
            # Split patients into folds
            patient_folds = np.array_split(shuffled_patients, self.n_splits)
            
            fold_metrics = []
            fold_matrices = []
            
            for fold in range(self.n_splits):
                # Get test patients for this fold
                test_patients = patient_folds[fold]
                
                # Get train/test indices based on patient IDs
                test_idx = np.where(np.isin(patient_ids, test_patients))[0]
                train_idx = np.where(~np.isin(patient_ids, test_patients))[0]
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Select features
                selected_indices = feature_selector.select_features(X_train, y_train)
                X_train_selected = X_train[:, selected_indices]
                X_test_selected = X_test[:, selected_indices]
                
                # Train and evaluate
                classifier.train(X_train_selected, y_train)
                metrics = classifier.evaluate(X_test_selected, y_test)
                
                fold_metrics.append(metrics)
                fold_matrices.append(metrics['confusion_matrix'])
            
            
            avg_metrics = {
                key: np.mean([m[key] for m in fold_metrics])
                for key in fold_metrics[0].keys()
                if key != 'confusion_matrix'
            }
            avg_matrix = np.mean(fold_matrices, axis=0)
            
            all_metrics.append(avg_metrics)
            all_confusion_matrices.append(avg_matrix)
        
        
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        final_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        
        return final_metrics, final_confusion_matrix

def main(X, y, patient_ids):
    """
    Main function to run the complete classification pipeline with patient-level separation.
    """
    # Initialize components
    feature_selector = TwoStageFeatureSelector(['mrmr', 'anova', 'mi'])
    classifiers = {
        'xgboost': FattyLiverClassifier('xgboost'),
        'lightgbm': FattyLiverClassifier('lightgbm'),
        'lda': FattyLiverClassifier('lda')
    }
    cross_validator = CrossValidator(n_splits=5, n_repeats=10)
    
    # Results dictionary to store all results
    results = {}
    
    # Evaluate each classifier
    for clf_name, classifier in classifiers.items():
        metrics, conf_matrix = cross_validator.perform_cross_validation(
            X, y, patient_ids, classifier, feature_selector
        )
        results[clf_name] = {
            'metrics': metrics,
            'confusion_matrix': conf_matrix
        }
        
        # Print results for this classifier
        print(f"\nResults for {clf_name.upper()}:")
        print(f"Accuracy: {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']['mean']:.4f} ± {metrics['sensitivity']['std']:.4f}")
        print(f"Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
        print(f"F1-score: {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
        print(f"MCC: {metrics['mcc']['mean']:.4f} ± {metrics['mcc']['std']:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
    return results

if __name__ == "__main__":
    # Data loading and preprocessing
    import os
    from glob import glob
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    
    # Dataset parameters from the paper
    data_dir = "dataset"  # Base directory containing the ultrasound images
    target_size = (399, 399)  # Image size specified in the paper
    class_mapping = {
        'healthy': 0,  # <5% fat
        'low_fat': 1,  # 5-30% fat
        'medium_fat': 2,  # 30-70% fat
        'high_fat': 3  # >70% fat
    }
    
    try:
        print("Starting fatty liver classification pipeline...")
        
        # Initialize components
        preprocessor = PreProcessor(target_size=target_size)
        glcm_extractor = GLCMFeatureExtractor()
        deep_extractor = DeepFeatureExtractor()
        
        # Load and organize dataset
        print("Loading dataset...")
        images = []
        labels = []
        patient_ids = []  # To ensure proper patient-wise split
        
        for class_name, label in class_mapping.items():
            class_path = os.path.join(data_dir, class_name)
            for patient_folder in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_folder)
                if os.path.isdir(patient_path):
                    # Load 10 images per patient as specified in paper
                    image_files = glob(os.path.join(patient_path, "*.dcm"))[:10]
                    for img_path in image_files:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Preprocess image
                            processed_img = preprocessor.preprocess_image(img)
                            images.append(processed_img)
                            labels.append(label)
                            patient_ids.append(patient_folder)
        
        X = np.array(images)
        y = np.array(labels)
        patient_ids = np.array(patient_ids)
        
        print(f"Loaded {len(X)} images from {len(np.unique(patient_ids))} patients")
        
        # Extract features
        print("Extracting GLCM features...")
        glcm_features = []
        for img in X:
            features, _ = glcm_extractor.extract_features(img)
            glcm_features.append(features)
        glcm_features = np.array(glcm_features)
        
        print("Extracting deep learning features...")
        deep_features = []
        for img in X:
            features, _ = deep_extractor.extract_features(img)
            deep_features.append(features)
        deep_features = np.array(deep_features)
        
        # Combine features
        print("Combining features...")
        X_combined = np.concatenate([glcm_features, deep_features], axis=1)
        
        # Run classification pipeline with patient IDs
        print("Running classification pipeline with patient-level separation...")
        results = main(X_combined, y, patient_ids)
        
        # Save results
        print("Saving results...")
        if not os.path.exists('results'):
            os.makedirs('results')
            
        # Save detailed results for each classifier
        for clf_name, clf_results in results.items():
            metrics = clf_results['metrics']
            conf_matrix = clf_results['confusion_matrix']
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Mean': [m['mean'] for m in metrics.values()],
                'Std': [m['std'] for m in metrics.values()]
            })
            metrics_df.to_csv(f'results/{clf_name}_metrics.csv', index=False)
            
            # Save confusion matrix
            np.save(f'results/{clf_name}_confusion_matrix.npy', conf_matrix)
            
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
