"""
Evaluate the updated hybrid TF-IDF+LR app on the same labeled dataset.
Since the model is trained on these 100 samples (in-sample), this measures
training fit — a separate cross-validation result is in evaluate.py.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import classify_text, _train_time_ms  # import new app
from dataset import LABELED_SAMPLES
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ["positive", "neutral", "negative"]
SEP = "=" * 60

y_true, y_pred, scores = [], [], []
total_infer = 0.0

for text, label in LABELED_SAMPLES:
    t0 = time.perf_counter()
    pred_label, score = classify_text(text)
    total_infer += time.perf_counter() - t0
    y_true.append(label)
    y_pred.append(pred_label)
    scores.append(score)

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
avg_ms = (total_infer / len(LABELED_SAMPLES)) * 1000

print(f"\n{SEP}")
print("  Hybrid TF-IDF+LR App — In-Sample Evaluation")
print(SEP)
print(f"  Model train time     : {_train_time_ms:.2f} ms (at startup, once)")
print(f"  Accuracy             : {acc:.4f}")
print(f"  F1 Macro             : {f1_macro:.4f}")
print(f"  F1 Weighted          : {f1_weighted:.4f}")
print(f"  Avg inference latency: {avg_ms:.4f} ms / sample")
print()
print("  Classification Report:")
for line in report.splitlines():
    print("    " + line)
print()
print("  Confusion Matrix  [pos / neu / neg]:")
for i, row in enumerate(cm):
    print(f"    {LABELS[i]:8s}: {row.tolist()}")
print()

# Count how many predictions used ML vs lexicon fallback
n_ml = sum(1 for s in scores if s >= 0.55)
n_lex = len(scores) - n_ml
print(f"  Routing: ML model used for {n_ml}/{len(scores)} samples, "
      f"lexicon fallback for {n_lex}/{len(scores)} samples")
print()
