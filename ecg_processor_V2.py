import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch
from sklearn.decomposition import PCA, FastICA
from scipy.stats import skew, kurtosis
import warnings

# Suppress specific warnings that are expected during ICA convergence
warnings.filterwarnings('ignore', message='FastICA did not converge')
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation')

def preprocess_ecg_signal(signal, fs, lowcut=0.5, highcut=40.0):
    """
    Preprocesses ECG signal with bandpass filtering to remove baseline wander and noise.
    
    Parameters:
    - signal: ECG signal array
    - fs: Sampling frequency
    - lowcut: Low cutoff frequency (removes baseline wander)
    - highcut: High cutoff frequency (removes high-frequency noise)
    
    Returns:
    - Filtered ECG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist # Removes very slow drifts (baseline wander) - often caused by respiration
    high = highcut / nyquist # Removes high-frequency noise (muscle artifacts, power line interference, etc.)
    
    # Design Butterworth bandpass filter
    b, a = butter(4, [low, high], btype='band')
    
    # Apply zero-phase filtering to avoid phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def segment_heartbeats(signal, peaks, fs, before_r=200, after_r=400):
    """
    Segments individual heartbeats from ECG signal based on R-peak locations.
    
    Parameters:
    - signal: ECG signal
    - peaks: R-peak locations (sample indices)
    - fs: Sampling frequency
    - before_r: Samples before R-peak (in ms equivalent)
    - after_r: Samples after R-peak (in ms equivalent)
    
    Returns:
    - List of heartbeat segments
    """
    heartbeats = []
    
    # Convert ms to samples (/ 1000 to convert ms to seconds)
    before_samples = int(before_r * fs / 1000) # number of samples before R-peak
    after_samples = int(after_r * fs / 1000) # number of samples after R-peak

    for peak in peaks:
        # Range before and after R-peak which the professor told in class
        start_idx = peak - before_samples
        end_idx = peak + after_samples
        
        # Ensure indices are within signal bounds
        if start_idx >= 0 and end_idx < len(signal):
            heartbeat = signal[start_idx:end_idx]
            heartbeats.append(heartbeat)
    
    return heartbeats

def extract_pca_features(heartbeats, n_components=5):
    """
    Applies PCA to heartbeat segments to extract principal components.
    
    PCA is used here for:
    1. Dimensionality reduction of heartbeat morphology
    2. Noise reduction by focusing on main variance directions
    3. Feature extraction of dominant cardiac patterns
    
    Parameters:
    - heartbeats: List of heartbeat segments
    - n_components: Number of principal components to extract
    
    Returns:
    - Dictionary with PCA features and explained variance
    """
    if len(heartbeats) < 3:
        return None
    
    # Convert heartbeats to matrix (each row is a heartbeat)
    min_length = min(len(hb) for hb in heartbeats) # Truncate to shortest heartbeat
    heartbeat_matrix = np.array([hb[:min_length] for hb in heartbeats]) # Shape: (num_heartbeats, heartbeat_length)
    
    # Ensure we don't request more components than we can compute: PCA requires that the number of components are less or equal to the rank of the data matrix.
    max_components = min(heartbeat_matrix.shape[0], heartbeat_matrix.shape[1], n_components)
    if max_components < 2:
        return None
    
    # Apply PCA
    pca = PCA(n_components=max_components)
    principal_components = pca.fit_transform(heartbeat_matrix)
    
    # Extract features from principal components
    features = {}
    
    # Explained variance ratios
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        features[f'pca_var_ratio_{i+1}'] = var_ratio
    
    # Statistical features of each principal component
    for i in range(max_components):
        pc_values = principal_components[:, i]
        
        # Safe statistical calculations
        features[f'pca_pc{i+1}_mean'] = np.mean(pc_values)
        features[f'pca_pc{i+1}_std'] = np.std(pc_values)
        
        # Handle potential precision issues in skewness/kurtosis
        try:
            features[f'pca_pc{i+1}_skew'] = skew(pc_values) if np.std(pc_values) > 1e-10 else 0.0
        except:
            features[f'pca_pc{i+1}_skew'] = 0.0
            
        try:
            features[f'pca_pc{i+1}_kurtosis'] = kurtosis(pc_values) if np.std(pc_values) > 1e-10 else 0.0
        except:
            features[f'pca_pc{i+1}_kurtosis'] = 0.0
    
    # Fill remaining components with zeros if we computed fewer than requested
    for i in range(max_components, n_components):
        features[f'pca_var_ratio_{i+1}'] = 0.0
        features[f'pca_pc{i+1}_mean'] = 0.0
        features[f'pca_pc{i+1}_std'] = 0.0
        features[f'pca_pc{i+1}_skew'] = 0.0
        features[f'pca_pc{i+1}_kurtosis'] = 0.0
    
    # Total explained variance
    features['pca_total_variance'] = np.sum(pca.explained_variance_ratio_)
    
    return features

def extract_ica_features(heartbeats, n_components=3):
    """
    Applies ICA to heartbeat segments to extract independent sources.
    
    ICA is used here for:
    1. Blind source separation to isolate different cardiac activities
    2. Artifact removal (muscle noise, baseline drift)
    3. Feature extraction of independent physiological processes
    
    Parameters:
    - heartbeats: List of heartbeat segments
    - n_components: Number of independent components to extract
    
    Returns:
    - Dictionary with ICA features
    """
    if len(heartbeats) < 3:
        return None
    
    # Convert heartbeats to matrix
    min_length = min(len(hb) for hb in heartbeats)
    heartbeat_matrix = np.array([hb[:min_length] for hb in heartbeats])
    
    # Ensure we have enough data for ICA
    max_components = min(heartbeat_matrix.shape[0], heartbeat_matrix.shape[1], n_components)
    if max_components < 2:
        return None
    
    # Apply ICA with robust parameters
    ica = FastICA(n_components=max_components, random_state=42, max_iter=2000, tol=1e-3)
    
    '''
    Transpose heartbeat_matrix to have shape (heartbeat_length, num_heartbeats)
    This is because ICA expects features in rows and samples in columns.
    '''

    try:
        # Add small amount of noise to avoid convergence issues with identical data
        noisy_matrix = heartbeat_matrix.T + np.random.normal(0, 1e-8, heartbeat_matrix.T.shape)
        independent_sources = ica.fit_transform(noisy_matrix).T
    except Exception as e:
        # If ICA fails completely, return None
        return None
    
    features = {}
    
    # Statistical features of each independent component
    for i in range(max_components):
        ic_values = independent_sources[i, :]
        
        features[f'ica_ic{i+1}_mean'] = np.mean(ic_values)
        features[f'ica_ic{i+1}_std'] = np.std(ic_values)
        
        # Safe statistical calculations
        try:
            features[f'ica_ic{i+1}_skew'] = skew(ic_values) if np.std(ic_values) > 1e-10 else 0.0
        except:
            features[f'ica_ic{i+1}_skew'] = 0.0
            
        try:
            features[f'ica_ic{i+1}_kurtosis'] = kurtosis(ic_values) if np.std(ic_values) > 1e-10 else 0.0
        except:
            features[f'ica_ic{i+1}_kurtosis'] = 0.0

        # Energy of the component (L2 norm, the higher the energy, the more significant the component: e.g., main cardiac activity)
        features[f'ica_ic{i+1}_energy'] = np.sum(ic_values**2)
    
    # Fill remaining components with zeros if we computed fewer than requested
    for i in range(max_components, n_components):
        features[f'ica_ic{i+1}_mean'] = 0.0
        features[f'ica_ic{i+1}_std'] = 0.0
        features[f'ica_ic{i+1}_skew'] = 0.0
        features[f'ica_ic{i+1}_kurtosis'] = 0.0
        features[f'ica_ic{i+1}_energy'] = 0.0
    
    # Cross-correlation between components (measures independence)
    for i in range(max_components):
        for j in range(i+1, max_components):
            try:
                corr = np.corrcoef(independent_sources[i, :], independent_sources[j, :])[0, 1]
                features[f'ica_corr_ic{i+1}_ic{j+1}'] = abs(corr) if not np.isnan(corr) else 0.0
            except:
                features[f'ica_corr_ic{i+1}_ic{j+1}'] = 0.0
    
    # Fill remaining correlations with zeros
    for i in range(max_components):
        for j in range(i+1, n_components):
            if f'ica_corr_ic{i+1}_ic{j+1}' not in features:
                features[f'ica_corr_ic{i+1}_ic{j+1}'] = 0.0
    
    return features

def extract_spectral_features(signal, fs):
    """
    Extracts frequency-domain features from ECG signal.
    
    Returns:
    - Dictionary with spectral features
    """
    # Compute power spectral density
    freqs, psd = welch(signal, fs, nperseg=min(len(signal)//4, 256))
    
    # Define frequency bands
    vlf_band = (0.0033, 0.04)  # Very Low Frequency
    lf_band = (0.04, 0.15)     # Low Frequency
    hf_band = (0.15, 0.4)      # High Frequency
    
    # Calculate power in each band
    vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], 
                        freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], 
                       freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], 
                       freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    total_power = vlf_power + lf_power + hf_power
    
    features = {
        'spectral_vlf_power': vlf_power,
        'spectral_lf_power': lf_power,
        'spectral_hf_power': hf_power,
        'spectral_total_power': total_power,
        'spectral_lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
        'spectral_lf_norm': lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0,
        'spectral_hf_norm': hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    }
    
    return features

def extract_comprehensive_features(signal_segment, fs):
    """
    Comprehensive feature extraction combining HRV, PCA, ICA, and spectral analysis.
    
    This function implements a feature extraction pipeline that:
    1. Preprocesses the ECG signal to remove noise and artifacts
    2. Detects R-peaks for heartbeat segmentation
    3. Extracts traditional HRV features (time-domain)
    4. Applies PCA for morphological pattern analysis
    5. Uses ICA for independent source separation
    6. Computes spectral features for frequency-domain analysis
    
    Parameters:
    - signal_segment: ECG signal segment
    - fs: Sampling frequency
    
    Returns:
    - Dictionary with comprehensive feature set or None if insufficient data
    """
    # Preprocess the signal
    filtered_signal = preprocess_ecg_signal(signal_segment, fs)
    
    # R-peak detection with adaptive threshold
    peaks, _ = find_peaks(filtered_signal, 
                         height=np.std(filtered_signal) * 1.5, 
                         distance=fs*0.3)
    
    # Require minimum number of heartbeats for reliable analysis
    if len(peaks) < 4:
        return None
    
    # 1. Traditional HRV Features
    rr_intervals = np.diff(peaks) * (1000 / fs)
    
    # Time-domain HRV
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
    
    # Additional HRV metrics
    pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100 if len(successive_diffs) > 0 else 0
    hr_mean = 60000 / mean_rr if mean_rr > 0 else 0
    
    '''
    mean_rr: Mean R-R Interval (average time between consecutive R-peaks in milliseconds)
    sdnn: Standard Deviation of Normal-to-Normal Intervals (overall HRV measure)
    rmssd: Root Mean Square of Successive Differences (short-term HRV measure)
    pnn50: Percentage of successive RR intervals that differ by more than 50 ms
    hr_mean: Mean Heart Rate (beats per minute)
    '''

    features = {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'hr_mean': hr_mean
    }
    
    # 2. Morphological Features via PCA
    heartbeats = segment_heartbeats(filtered_signal, peaks, fs)
    if len(heartbeats) >= 3:
        pca_features = extract_pca_features(heartbeats)
        if pca_features:
            features.update(pca_features)
    
    # 3. Independent Component Analysis
    if len(heartbeats) >= 3:
        ica_features = extract_ica_features(heartbeats)
        if ica_features:
            features.update(ica_features)
    
    # 4. Spectral Features
    spectral_features = extract_spectral_features(filtered_signal, fs)
    features.update(spectral_features)
    
    # 5. Signal Quality Metrics
    features['signal_quality_snr'] = np.var(filtered_signal) / np.var(signal_segment - filtered_signal) if np.var(signal_segment - filtered_signal) > 0 else 0
    features['num_heartbeats'] = len(peaks)
    features['beat_detection_rate'] = len(peaks) / (len(signal_segment) / fs) * 60  # beats per minute
    
    return features

def extract_hrv_features(signal_segment, fs):
    """
    Calls the comprehensive feature extraction but returns only HRV features.
    """
    comprehensive_features = extract_comprehensive_features(signal_segment, fs)
    if comprehensive_features is None:
        return None
    
    # Return only traditional HRV features for backward compatibility
    hrv_features = {
        'mean_rr': comprehensive_features['mean_rr'],
        'sdnn': comprehensive_features['sdnn'],
        'rmssd': comprehensive_features['rmssd']
    }
    
    return hrv_features

def process_records_in_windows(record_list, db_dir, label, window_seconds=30, use_comprehensive=False):
    """
    Processes a list of records by slicing the signal into windows
    and extracting features from each window.
    
    Parameters:
    - record_list: List of record names to process
    - db_dir: Database directory path
    - label: Label for the data (e.g., 'Normal', 'AFib')
    - window_seconds: Window size in seconds
    - use_comprehensive: If True, extract comprehensive features including PCA/ICA
    
    Returns:
    - List of feature dictionaries
    """
    all_window_features = []
    total_records = len(record_list)
    
    for idx, rec_name in enumerate(record_list, 1):
        print(f"Processing record {idx}/{total_records}: {rec_name} from {db_dir}")
        try:
            # Download record data from PhysioNet
            record = wfdb.rdrecord(rec_name, pn_dir=db_dir)
            # Get the lead I channel for simplicity and consistency [:, 0]
            # Get the lead II channel for simplicity and consistency [:, 1]
            signal = record.p_signal[:, 0]
            fs = record.fs
            
            window_samples = window_seconds * fs
            
            # Slide a non-overlapping window across the entire signal
            num_windows = len(signal) // window_samples
            successful_windows = 0
            
            for i in range(num_windows):
                start = i * window_samples
                end = start + window_samples
                window_signal = signal[start:end]
                
                # Extract features from this specific window
                if use_comprehensive:
                    features = extract_comprehensive_features(window_signal, fs)
                else:
                    features = extract_hrv_features(window_signal, fs)
                
                # If features were successfully extracted, add them to our list
                if features:
                    features['record'] = rec_name
                    features['window_id'] = i
                    features['label'] = label
                    all_window_features.append(features)
                    successful_windows += 1
            
            print(f"Successfully processed {successful_windows}/{num_windows} windows")
            
        except Exception as e:
            print(f"Failed to process {rec_name}. Error: {e}")
    
    print(f"Total windows processed: {len(all_window_features)}")
    return all_window_features