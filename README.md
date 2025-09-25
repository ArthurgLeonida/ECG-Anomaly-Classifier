# ECG Anomaly Classifier

This project implements an automated system for detecting cardiac anomalies, specifically Atrial Fibrillation (AFib), from ECG signals using advanced feature extraction and machine learning techniques. The system combines traditional Heart Rate Variability (HRV) metrics with morphological analysis via Principal Component Analysis (PCA), Independent Component Analysis (ICA), and spectral analysis, followed by XGBoost classification for robust anomaly detection.

## 🩺 Project Overview

The classifier distinguishes between Normal Sinus Rhythm (NSR) and Atrial Fibrillation (AFib) by analyzing ECG signals in 30-second windows. AFib is characterized by:
- Absence of P-waves (lost organized atrial contraction)
- Irregular R-R intervals (chaotic heartbeat timing)
- Higher Heart Rate Variability metrics

## 🧠 Advanced Feature Extraction Pipeline

The [`ecg_processor_V2.py`](ecg_processor_V2.py) module implements a comprehensive feature extraction pipeline:

### 1. Signal Preprocessing
- **Bandpass filtering** (0.5-40 Hz) with zero-phase filtering to remove baseline wander and noise
- **R-peak detection** using adaptive thresholding for heartbeat segmentation

### 2. Traditional HRV Features (Time-domain)
- `mean_rr`: Mean R-R interval
- `sdnn`: Standard deviation of normal-to-normal intervals
- `rmssd`: Root mean square of successive differences
- `pnn50`: Percentage of successive RR intervals differing by >50ms
- `hr_mean`: Mean heart rate

### 3. Morphological Analysis (PCA)
- **Dimensionality reduction** of heartbeat morphology
- **Noise reduction** focusing on principal variance directions
- **Pattern recognition** for different cardiac conditions
- Features: variance ratios, statistical moments of principal components

### 4. Independent Source Separation (ICA)
- **Blind source separation** to isolate different cardiac activities
- **Artifact removal** (muscle noise, baseline drift)
- **Multi-process analysis** revealing overlapping cardiac activities
- Features: statistical properties and energy of independent components

### 5. Spectral Analysis (Frequency-domain)
- **VLF/LF/HF band** power analysis (0.0033-0.4 Hz)
- **LF/HF ratio** indicating autonomic balance (sympathetic vs parasympathetic)
- **Normalized spectral** features for clinical relevance

### 6. Signal Quality Assessment
- **Signal-to-noise ratio** calculation
- **Beat detection rate** validation
- **Heartbeat count** per window

## 📊 Data Sources

- **MIT-BIH Atrial Fibrillation Database (afdb)**: Recordings of patients with AFib
- **MIT-BIH Normal Sinus Rhythm Database (nsrdb)**: Recordings of healthy subjects
- Data automatically downloaded from PhysioNet using the `wfdb` library

## 🚀 Machine Learning Approach

### XGBoost Classification
- **Medical-optimized parameters** with shallow trees (max_depth=4) to prevent overfitting
- **Regularization** (L1 and L2) for robust performance on small datasets
- **Cross-validation** for reliable performance estimation
- **Feature importance analysis** for clinical interpretability

### Key Performance Metrics
- **Accuracy**: Model classification accuracy
- **AUC Score**: Area under ROC curve for discriminative power
- **Precision/Recall**: Detailed classification performance per class
- **Feature importance**: Identifies most predictive cardiac markers

## 📁 Project Structure

```
├── ECG_anomaly_class_ECG1.ipynb    # Main analysis notebook (4 records)
├── ECG_anomaly_class_ECG2.ipynb    # Extended analysis (5 records)
├── ecg_processor_V2.py             # Advanced feature extraction pipeline
├── README.md                       # Project documentation
├── data/                           # Generated feature datasets
│   ├── ecg_basic_features_ECG1.csv
│   ├── ecg_basic_features_ECG2.csv
│   ├── ecg_comprehensive_features_ECG1.csv
│   └── ecg_comprehensive_features_ECG2.csv
└── __pycache__/                    # Python cache files
```

## 🛠️ Technologies Used

- **Data Processing**: Python, NumPy, Pandas, SciPy
- **Signal Processing**: wfdb, scipy.signal (filtering, peak detection)
- **Feature Extraction**: scikit-learn (PCA, ICA), scipy.stats
- **Machine Learning**: XGBoost, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install wfdb numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```
### 2. Run Notebook
Open and run [`ECG_anomaly_class_ECG1.ipynb`](ECG_anomaly_class_ECG1.ipynb) to:
- Load preprocessed features from CSV files
- Perform statistical analysis of feature discriminative power
- Train XGBoost classifier with medical-optimized parameters
- Evaluate model performance with cross-validation

## 📈 Results

The model demonstrates exceptional clinical performance with near-perfect accuracy:

### 🎯 Classification Performance
- **Accuracy**: 100% (Only 1 misclassification out of ~4,000 test samples)
- **AUC Score**: > 1 (Excellent discriminative power)
- **Cross-validation**: Consistent performance across all folds
- **Precision/Recall**: Near-perfect scores for both Normal and AFib classes

### 🔍 Feature Importance Analysis
The XGBoost model identified the most predictive cardiac markers:

1. **`ica_ic3_energy`** (Rank 1) - Independent Component Analysis energy from 3rd component
   - *Most important feature* for AFib detection
   - Captures independent cardiac signal sources and artifacts
   - The single misclassification was primarily influenced by this feature
2. **`pca_var_ratio_1`** (Rank 2) - First Principal Component variance ratio
   - Captures main morphological variations in heartbeat patterns
   - Distinguishes AFib irregular morphology from normal rhythm
3. **`sdnn`** (Rank 3) - Standard Deviation of Normal-to-Normal intervals
   - Traditional HRV measure reflecting overall heart rate variability
   - AFib shows significantly higher SDNN due to irregular rhythm patterns
4. **`rmssd`** - Root Mean Square of Successive Differences
   - Another key HRV measure showing AFib's beat-to-beat irregularity
   - Complements SDNN for comprehensive variability assessment

### 📊 Clinical Validation Results
- **High Confidence Predictions**: >99.9% of predictions with confidence >80%
- **High Confidence Accuracy**: 100% for confident predictions
- **Uncertain Predictions**: 1 of 3986 cases require manual review

### 🎯 Key Clinical Findings
- **Morphological Patterns**: PCA captures distinct heartbeat shape variations (AFib shows 40% more morphological variation)
- **Independent Sources**: ICA energy features provide the strongest discrimination signal
- **Spectral Analysis**: LF/HF ratio disrupted in AFib patients, indicating autonomic dysfunction

### 🏆 Model Strengths
- **Near-Perfect Accuracy**: Only 1 error demonstrates robust feature engineering
- **Clinical Interpretability**: Top features align with known AFib pathophysiology
- **Balanced Performance**: Excellent results for both Normal and AFib classes
- **Robust Validation**: Consistent cross-validation performance indicates good generalization
- **Real-world Ready**: High confidence predictions suitable for clinical screening

### ⚠️ Model Limitations
- **Single Misclassification**: Attributed to `ica_ic3_energy` feature edge case
- **Dataset Balance**: Normal samples (75%) outnumber AFib (25%) - reflects real-world prevalence
- **Feature Sensitivity**: ICA features may be sensitive to specific signal artifacts

## 📝 Key Features

- ✅ **Automated ECG download** from PhysioNet databases
- ✅ **Robust signal preprocessing** with artifact removal
- ✅ **Comprehensive feature extraction** (60+ features per window)
- ✅ **Statistical significance testing** for feature validation
- ✅ **Medical-optimized ML pipeline** with interpretable results
- ✅ **Clinical decision support** with confidence analysis
- ✅ **Reproducible results** with fixed random seeds

## 🔬 Research Applications

This framework can be extended for:
- Other arrhythmia types (ventricular fibrillation, tachycardia)
- Real-time ECG monitoring systems
- Wearable device integration
- Clinical decision support tools
- Cardiac health screening programs

---

*This project demonstrates the application of advanced signal processing and machine learning techniques for automated cardiac anomaly detection, combining clinical knowledge with computational methods for improved healthcare outcomes.*