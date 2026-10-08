# HAR70+ activity recognition in older adults

Jialu Wei — DATAX504 Deep Learning — individual final project.

This project evaluates a small Keras/TensorFlow 1D CNN for four activities:
lying, sitting, standing and walking. A sensor ablation compares both sensors,
lower back only, and thigh only. Each configuration uses seeds 42, 123 and 2026.
A three-fold grouped robustness check uses only the original training people.

## Files

- `har70_final.ipynb`: readable notebook with recorded outputs and plots.
- `train.py`: the identical training/evaluation code extracted from the notebook.
- `verify_results.py`: recomputes recorded metrics from prediction CSVs without training.
- `requirements.txt`: exact direct packages for the recorded Linux CPU run.
- `requirements-macos.txt`: optional, untested Mac compatibility environment.
- `results/`: recorded result tables, predictions, models, histories and figures.
- `data/har70plus.zip`: the supplied original data, included in the download package.
- `HAR70_Final_Report.pdf`: final report describing the recorded experiment.
- `DATA_LICENSE.md`: source attribution and data licence.

The report describes the recorded experiment and its limitations. The notebook
and script implement the same preprocessing, training and evaluation procedure.

To quickly verify the saved result tables without TensorFlow or retraining:

```bash
python verify_results.py
```

This requires numpy, pandas and scikit-learn. It checks prediction labels,
probability sums, confusion matrices and all reported mean/SD values.

## Reproduce the recorded Linux CPU experiment

Use Python 3.12 on Linux. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

This runs all 12 models: nine sensor/seed fits and three grouped folds.
New outputs go to `results_reproduced/`, preserving the recorded `results/`.
Compare the new `protocol.json`, `test_runs.csv`, `test_summary.csv` and
`grouped_cv.csv` with the recorded versions. Runtime depends on the hardware.
The accompanying report is tied to the recorded run; if a new environment
changes results, update report and slides together rather than mix runs.

The recorded run used `HAR70_OUTPUT_DIR=results` with TensorFlow 2.20.0,
Keras 3.15.1, numpy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and matplotlib 3.10.8.
CPU threads are limited and deterministic operations requested. Exact agreement
is not guaranteed across hardware, operating systems or package versions.

## Run in Jupyter

In the Linux environment above:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name har70-final --display-name "HAR70 Final"
jupyter lab
```

Open `har70_final.ipynb`, select the **HAR70 Final** kernel and choose
**Restart Kernel and Run All Cells**. Keep `data/har70plus.zip` in place.
Every submitted code cell is already executed; the outputs can be read without
retraining. The script is an alternative to Jupyter, not an additional experiment.

For macOS, use an existing working course TensorFlow environment first.
If a new environment is needed, `requirements-macos.txt` offers CPU-compatible
versions by architecture; they were not used for the recorded experiment.
TensorFlow's installation documentation is at https://www.tensorflow.org/install/pip.

## Data location and downloading

The separate complete download package contains the original archive. This GitHub upload folder excludes the data archive; download it using the instructions below before retraining.
If the GitHub upload excludes the data ZIP, download the HAR70+ archive from:
https://archive.ics.uci.edu/dataset/780/har70
and save it as `data/har70plus.zip`. The code reads CSVs directly from the ZIP.
An extracted `data/har70plus/` containing the 18 CSVs is also supported.
The recorded `results/protocol.json` contains the archive SHA-256 fingerprint.

## Evaluation protocol

- Train participants: 501–509, 513, 515–518 (14 people).
- Validation: 510 and 514. Test: 511 and 512.
- Four labels are retained; shuffling and stairs are excluded, not re-labelled.
- Windows remain inside one person, one activity, and one continuous segment.
- A gap greater than 30 ms starts a new segment; 100 samples, stride 50.
- Mean/std are estimated from training windows only.
- Same windows, layer configuration and training settings for sensor ablations.
- Adam 0.001, batch size 64, maximum 30 epochs, early stopping patience 5.
- Best weights are restored using validation loss.
- Macro F1 is primary; accuracy, class metrics and person-specific results are reported.
- Seed 42 is selected in advance as the example for confusion matrices and error analysis.
- Grouped CV never includes original validation/test participants.

## Limits and AI disclosure

Only 18 independent people are represented, with two in the final test set.
Overlapping windows are correlated observations. High accuracy on these stable
windows is not evidence of clinical benefit or population-wide validity.
Earlier exploratory test outputs were seen before preprocessing was corrected;
this is not blind external validation. No corrected test score is used for tuning.

ChatGPT/Codex assisted with planning, code generation and correction, execution of
experiments, preparation of figures, and report drafting/editing. Sources were
checked. Metrics are computed by the Python pipeline. The student must review
the work, disclose assistance, and be able to explain the submitted code.

## Main references

Logacjov, A., & Ustad, A. (2023). HAR70+ [Data set]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CW3D

Ustad, A., et al. (2023). Validation of an activity type recognition model
classifying daily physical behavior in older adults: The HAR70+ model.
Sensors, 23(5), 2368. https://doi.org/10.3390/s23052368

Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural
networks for multimodal wearable activity recognition. Sensors, 16(1), 115.
https://doi.org/10.3390/s16010115
