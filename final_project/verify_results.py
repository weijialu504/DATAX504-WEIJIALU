"""Recalculate recorded metrics from prediction CSVs without retraining models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

results = Path('results')
protocol = json.loads((results / 'protocol.json').read_text())
recorded = pd.read_csv(results / 'test_runs.csv')
class_names = protocol['class_names']
test_people = set(protocol['test_subjects'])
recomputed = []

for _, row in recorded.iterrows():
    name = f"{row['input']}_seed{int(row['seed'])}"
    p = pd.read_csv(results / f'predictions_{name}.csv', dtype={'subject': str})
    assert len(p) == 4172
    assert set(p['subject']) == test_people
    assert np.all(p['max_interval_seconds'] <= protocol['max_interval_seconds'])
    accuracy = accuracy_score(p['true'], p['predicted'])
    macro_f1 = f1_score(p['true'], p['predicted'], labels=range(4),
                        average='macro', zero_division=0)
    np.testing.assert_allclose([accuracy, macro_f1], [row['accuracy'], row['macro_f1']], atol=1e-12)
    probabilities = p[[f'prob_{c}' for c in class_names]].to_numpy()
    np.testing.assert_array_equal(probabilities.argmax(axis=1), p['predicted'].to_numpy())
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    if int(row['seed']) == 42:
        cm = confusion_matrix(p['true'], p['predicted'], labels=range(4))
        saved_cm = np.loadtxt(results / f"confusion_{row['input']}_seed42.csv", delimiter=',', dtype=int)
        np.testing.assert_array_equal(cm, saved_cm)
    recomputed.append({'input':row['input'], 'seed':int(row['seed']),
                       'accuracy':accuracy, 'macro_f1':macro_f1})

new_summary = pd.DataFrame(recomputed).groupby('input').agg(
    accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
    macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
old_summary = pd.read_csv(results / 'test_summary.csv').set_index('input').sort_index()
np.testing.assert_allclose(new_summary.to_numpy(),old_summary.to_numpy(),atol=1e-12)

cv = pd.read_csv(results / 'grouped_cv.csv')
for _, row in cv.iterrows():
    p = pd.read_csv(results / f"grouped_fold{int(row['fold'])}_predictions.csv",dtype={'subject':str})
    np.testing.assert_allclose(
        [accuracy_score(p['true'],p['predicted']),
         f1_score(p['true'],p['predicted'],labels=range(4),average='macro',zero_division=0)],
        [row['accuracy'],row['macro_f1']],atol=1e-12)
    assert set(p['subject']).isdisjoint(test_people | set(protocol['validation_subjects']))

print('Verified: all 9 test runs, 3 confusion matrices, 3 grouped folds and summary statistics.')
print(new_summary.to_string())
