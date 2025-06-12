# Stress EEG Analyzer

This folder contains a minimal example for training a stress vs. relaxation classifier
from EEG data stored in `.mat` files. The dataset referenced in `stress_eeg_analysis.py`
corresponds to the [SAM-40 dataset](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090).

Place the extracted `.mat` files inside a directory and run the script:

```bash
python3 stress_eeg_analysis.py /path/to/dataset
```

The script loads each file, computes band power features (delta, theta, alpha, beta, gamma) using Welch's
method, and trains a `RandomForestClassifier` to predict stress labels. The dataset
structure may differ, so adjust the key names in `StressEEGAnalyzer._load_mat_file` to
match the variables stored in the `.mat` files.

This example is meant for educational purposes and requires `numpy`, `scipy` and
`scikit-learn` installed in your Python environment.
