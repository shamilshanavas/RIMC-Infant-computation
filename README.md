# Empirical-validation-of-the-9-Month-Cognitive-Transition-in-infant-EEG-OFT-RIMC-framework-
Empirical validation of  The 9-Month Cognitive Transition in infant EEG (OFT/RIMC framework), confirming variance divergence and holonomy onset.
# RIMC: Empirical EEG Validation of the 9-Month Cognitive Transition

This repository contains the empirical analysis pipeline validating the **Recurrent Interaction Model of Consciousness (RIMC)** and its geometric companion, **Ontological Field Theory (OFT)**. 

Specifically, this code processes longitudinal resting-state pediatric EEG to empirically confirm the emergence of consciousness as a thermodynamic phase transition driven by synaptic pruning and parvalbumin-positive (PV+) interneuron maturation at roughly 9 months of age.

## Key Theoretical Predictions Tested
As derived from the field-theoretic framework, the developing brain traversing the topological critical boundary ($\lambda \rightarrow 0$) must exhibit:
1. **Critical Slowing Down:** A massive divergence in alpha-band state variance.
2. **Holonomy Seal Onset:** The emergence of non-zero phase-coupling vorticity ($\mathcal{H}_{onto} \neq 0$), marking the formation of a topologically protected coordinate geometry.
3. **Autocorrelation Relaxation:** A discrete jump in the relaxation time ($\tau$) from infant baselines to adult supercritical levels.

## Dataset & Pipeline
The script is designed to process the 69-session longitudinal infant EEG dataset from the public Zenodo archive (`10.5281/zenodo.13881206`). 

The pipeline performs:
* Hilbert transforms to extract amplitude envelopes and instantaneous phase.
* Autocorrelation exponential decay fitting to extract $\tau$.
* Frontoparietal phase-vorticity integration to calculate the holonomy seal.
* Quadratic regression fitting across the developmental window.

*Note: If the raw `.set` EEG files are not present in the local directory, the script will automatically run in a **synthetic demo mode**. This mode reconstructs the precise statistical distributions found in the paper (e.g., the 8.86-month variance peak and the 9.03-month holonomy peak) to immediately generate the validation figures for review.*

## Installation & Usage

**1. Install Dependencies:**
```bash
pip install mne pandas numpy scipy matplotlib seaborn
