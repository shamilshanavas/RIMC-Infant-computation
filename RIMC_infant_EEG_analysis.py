"""
RIMC / OFT Empirical Validation: Pediatric Developmental Burst
Dataset: 69 longitudinal infant EEG sessions (Zenodo: 10.5281/zenodo.13881206)
Analysis: Critical Slowing Down (Variance Divergence) and Holonomy Onset at 9 Months
"""

import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert

# Set dark theme to match the original RIMC Figure 1 & 2 aesthetics
plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def compute_relaxation_time(data, sfreq, max_lag_sec=1.0):
    """
    Computes the autocorrelation relaxation time (tau) of the amplitude envelope.
    """
    analytic_signal = hilbert(data, axis=1)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Demean envelope
    env_mean = amplitude_envelope - np.mean(amplitude_envelope, axis=1, keepdims=True)
    
    # Compute autocorrelation
    max_lag = int(max_lag_sec * sfreq)
    autocorr = np.zeros((data.shape[0], max_lag))
    
    for i in range(data.shape[0]):
        cov = np.correlate(env_mean[i, :], env_mean[i, :], mode='full')
        cov = cov[cov.size // 2:] # Keep positive lags
        autocorr[i, :] = cov[:max_lag] / cov[0] # Normalize
        
    mean_autocorr = np.mean(autocorr, axis=0)
    
    # Fit exponential decay: C(t) = exp(-t / tau)
    lags = np.arange(max_lag) / sfreq
    # Linearize the fit: ln(C(t)) = -t / tau
    # Only fit where autocorrelation is positive to avoid log(negative)
    valid = mean_autocorr > 0.05
    if np.sum(valid) > 2:
        p, cov = np.polyfit(lags[valid], np.log(mean_autocorr[valid]), 1)
        tau = -1.0 / p
        return tau, mean_autocorr, lags
    return np.nan, mean_autocorr, lags

def compute_phase_vorticity(data):
    """
    Computes the macroscopic phase-coupling vorticity (proxy for Holonomy H_onto)
    across a frontoparietal closed loop.
    """
    analytic_signal = hilbert(data, axis=1)
    phase = np.unwrap(np.angle(analytic_signal), axis=1)
    
    vorticity = np.zeros(phase.shape[1])
    n_channels = data.shape[0]
    for i in range(n_channels):
        next_i = (i + 1) % n_channels
        phase_diff = phase[next_i, :] - phase[i, :]
        # Wrap to [-pi, pi]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        vorticity += phase_diff
        
    return np.mean(vorticity**2) # Variance of the geometric phase

def process_zenodo_dataset(data_dir="zenodo_infant_eeg"):
    """
    Full pipeline to process the 69 infant EEG sessions.
    """
    files = glob.glob(os.path.join(data_dir, "*.set"))
    if not files:
        raise FileNotFoundError(f"No .set files found in {data_dir}. Please download the Zenodo dataset first.")

    results = []
    
    for f in files:
        try:
            # Parse age from filename (assuming format sub-XX_age-YYmo.set)
            basename = os.path.basename(f)
            age_months = float(basename.split('age-')[1].replace('mo.set', ''))
            
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            raw.filter(l_freq=8.0, h_freq=13.0, verbose=False) # Alpha band
            
            # Select frontoparietal loop
            channels = [ch for ch in ['F3', 'F4', 'C4', 'C3'] if ch in raw.ch_names]
            data = raw.copy().pick_channels(channels).get_data()
            sfreq = raw.info['sfreq']
            
            # 1. Variance
            variance = np.var(data)
            
            # 2. Relaxation Time
            tau, _, _ = compute_relaxation_time(data, sfreq)
            
            # 3. Holonomy
            holonomy = compute_phase_vorticity(data)
            
            results.append({
                'age': age_months,
                'variance': variance,
                'tau_ms': tau * 1000,
                'holonomy': holonomy
            })
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    df = pd.DataFrame(results)
    plot_developmental_trajectories(df)
    return df
    
def plot_developmental_trajectories(df):
    """
    Generates the developmental transition figures from the processed data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#FF9900', '#00E5FF', '#FF3366']
    titles = ['A. Empirical Pediatric Entropy Burst', 'B. Empirical Holonomy Seal Onset', 'C. Empirical Critical Slowing Down']
    ylabels = ['Alpha Variance ($\mu V^2$)', 'Holonomy Strength ($\mathcal{H}$)', 'Fitted Relaxation Tau (ms)']
    y_vars = ['variance', 'holonomy', 'tau_ms']
    
    for i in range(3):
        ax = axes[i]
        x = df['age']
        y = df[y_vars[i]]
        
        ax.scatter(x, y, color=colors[i], s=20, label='Infant Sessions')
        
        # Fit quadratic for A and B
        if i < 2:
            coeffs = np.polyfit(x, y, 2)
            p = np.poly1d(coeffs)
            x_line = np.linspace(x.min(), x.max(), 100)
            peak_x = -coeffs[1] / (2 * coeffs[0])
            ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2, label=f'Quadratic Fit (Peak: {peak_x:.2f}mo)')
        else:
            # Linear fit for C
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2, label='Linear Fit')
            
        ax.set_title(titles[i], fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel('Estimated Age (Months)', color='white')
        ax.set_ylabel(ylabels[i], color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='black', edgecolor='white', fontsize=9)
        ax.grid(True, alpha=0.2)
        
    plt.suptitle("Empirical RIMC Developmental Transition Study: The 9-Month Cognitive Revolution\nMeasured EEG Metrics vs. Estimated Age", fontsize=14, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    plt.savefig("Figure2_Developmental_Trajectories.png", bbox_inches='tight', facecolor='black')
    print("Saved Figure2_Developmental_Trajectories.png")

if __name__ == "__main__":
    process_zenodo_dataset()
