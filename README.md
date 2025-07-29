# Neural speech tracking in noise reflects the opposing influence of SNR on intelligibility and attentional effort

This repository contains code for reproducing the figures in our study "Neural speech tracking in noise reflects the opposing influence of SNR on intelligibility and attentional effort" published on Imaging Neuroscience. All analyses and visualizations are implemented in `main.m`, with supporting data in the provided `.mat` files.

## 📂 File Structure

- `main.m` — MATLAB script that generates all figures in the paper.
- `DataShared_*.mat` — Supporting data files required to reproduce figures.
- `README.md` — This documentation file.

## 📊 Figures Overview

### Figure 1: Psychometric Curves for Intelligibility
- **SI vs. SNR**: Shows speech intelligibility curves for both pedestrian and babble noise.
- Key Insight: No significant difference in SI between the two noise types.

---

### Figure 2: Neural Speech Tracking under Varying SNR/SI
- **2A-C**: Heatmaps showing target tracking (rT), masker tracking (rM), and their difference (rD = rT - rM) across SNR and SI bins.
- **2D-E**: Group-averaged rT, rM, and rD as functions of SI and SNR, with SEM error bars.
- **2F-G**:  
  - 2F: rD across SNR levels for different SI bins, with linear regression lines.  
  - 2G: Regression slopes showing how rD-SNR relationships change by SI bin, derived via linear mixed-effects models (LMEs).

---

### Figure 3: Behavioral Correlates of Neural Tracking
- **3A-B**:  
  - 3A: Repeated-word hit rate (HR) increases with SNR.  
  - 3B: Gaze velocity (GV) increases with SNR, suggesting reduced attentional effort (AE).
- **3C-D**:  
  - 3C: rD is positively associated with HR (binned analysis).  
  - 3D: In ceiling SI trials, GV negatively correlates with rD.
- **3E-F**:  
  - 3E: Positive HR–rD relationship persists across SNR bins.  
  - 3F: Negative GV–rD relationship holds across all SNR levels.

---

### Figure 4: LME Models on rD and all factors
- **4A**: Fixed effects from LME predicting rD under low vs. ceiling SI conditions. Significant predictors are highlighted.
- **4C**: LME modeling of how SNR modulates SI and GV (both fixed and random effects).
- **4D**: Interaction between SNR and SI/GV in predicting rD, visualized with slope bar plots per SNR level.

---

## 📎 Dependencies

- MATLAB R2021a or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## 📌 Usage

1. Open `main.m` in MATLAB.
2. Ensure the two `.mat` files are in the same directory.
3. Run the script to generate all figures.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📬 Contact

For questions or collaborations, please contact xh2369@columbia.edu or open an issue.
