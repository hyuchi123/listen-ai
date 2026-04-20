"""
Evaluation script: compares Lexicon-based vs TF-IDF + Logistic Regression
sentiment classifiers on the labeled dataset.

Usage:
    python evaluate.py
"""

import sys
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

# Allow importing the existing NLP app from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import classify_text  # noqa: E402

from dataset import LABELED_SAMPLES  # noqa: E402

LABELS = ["positive", "neutral", "negative"]


# ─── 1. Lexicon-based evaluation ────────────────────────────────────────────

def evaluate_lexicon(samples):
    y_true, y_pred = [], []
    total_time = 0.0

    for text, label in samples:
        t0 = time.perf_counter()
        pred_label, _ = classify_text(text)
        total_time += time.perf_counter() - t0
        y_true.append(label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    avg_ms = (total_time / len(samples)) * 1000

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": cm,
        "avg_latency_ms": avg_ms,
        "total_time_s": total_time,
    }


# ─── 2. TF-IDF + Logistic Regression evaluation (5-fold CV) ─────────────────

def evaluate_tfidf_logreg(samples):
    texts = [t for t, _ in samples]
    labels = [l for _, l in samples]

    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    inv_map = {v: k for k, v in label_map.items()}
    y = np.array([label_map[l] for l in labels])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1_macros, f1_weighteds = [], [], []
    all_true, all_pred = [], []
    train_times, infer_times = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        X_train = [texts[i] for i in train_idx]
        X_test  = [texts[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Train
        t0 = time.perf_counter()
        vec = TfidfVectorizer(
            analyzer="char_wb",   # character n-grams handle both EN and ZH
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        X_tr = vec.fit_transform(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_tr, y_train)
        train_times.append(time.perf_counter() - t0)

        # Infer
        t0 = time.perf_counter()
        X_te = vec.transform(X_test)
        y_pred = clf.predict(X_te)
        infer_times.append(time.perf_counter() - t0)

        accs.append(accuracy_score(y_test, y_pred))
        f1_macros.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighteds.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    all_true_labels = [inv_map[v] for v in all_true]
    all_pred_labels = [inv_map[v] for v in all_pred]
    report = classification_report(all_true_labels, all_pred_labels, labels=LABELS, zero_division=0)
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=LABELS)

    n_test = len(samples) - len(samples) // skf.n_splits
    avg_infer_ms = (np.mean(infer_times) / n_test) * 1000

    return {
        "accuracy": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_macro": np.mean(f1_macros),
        "f1_macro_std": np.std(f1_macros),
        "f1_weighted": np.mean(f1_weighteds),
        "f1_weighted_std": np.std(f1_weighteds),
        "report": report,
        "confusion_matrix": cm,
        "avg_train_time_s": np.mean(train_times),
        "avg_latency_ms": avg_infer_ms,
    }


# ─── Pretty print ────────────────────────────────────────────────────────────

SEP = "=" * 60

def print_results(name, res, is_cv=False):
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)
    if is_cv:
        print(f"  Accuracy (5-fold CV) : {res['accuracy']:.4f} ± {res['accuracy_std']:.4f}")
        print(f"  F1 Macro   (CV)      : {res['f1_macro']:.4f} ± {res['f1_macro_std']:.4f}")
        print(f"  F1 Weighted (CV)     : {res['f1_weighted']:.4f} ± {res['f1_weighted_std']:.4f}")
        print(f"  Avg train time       : {res['avg_train_time_s']*1000:.2f} ms / fold")
    else:
        print(f"  Accuracy             : {res['accuracy']:.4f}")
        print(f"  F1 Macro             : {res['f1_macro']:.4f}")
        print(f"  F1 Weighted          : {res['f1_weighted']:.4f}")
    print(f"  Avg inference latency: {res['avg_latency_ms']:.4f} ms / sample")
    print()
    print("  Classification Report (aggregated):")
    for line in res["report"].splitlines():
        print("    " + line)
    print()
    print("  Confusion Matrix  (rows=true, cols=pred)  [pos / neu / neg]:")
    for i, row in enumerate(res["confusion_matrix"]):
        print(f"    {LABELS[i]:8s}: {row.tolist()}")
    print()


if __name__ == "__main__":
    print("\nListenAI NLP – Sentiment Classifier Evaluation")
    print(f"Dataset: {len(LABELED_SAMPLES)} samples  "
          f"(positive={sum(1 for _,l in LABELED_SAMPLES if l=='positive')}, "
          f"neutral={sum(1 for _,l in LABELED_SAMPLES if l=='neutral')}, "
          f"negative={sum(1 for _,l in LABELED_SAMPLES if l=='negative')})")

    print("\n[1/2] Evaluating Lexicon-based classifier...")
    lex_res = evaluate_lexicon(LABELED_SAMPLES)
    print_results("Lexicon-based (existing algorithm)", lex_res, is_cv=False)

    print("\n[2/2] Evaluating TF-IDF + Logistic Regression (5-fold CV)...")
    tfidf_res = evaluate_tfidf_logreg(LABELED_SAMPLES)
    print_results("TF-IDF + Logistic Regression (new algorithm)", tfidf_res, is_cv=True)

    print(SEP)
    print("  Summary Comparison")
    print(SEP)
    print(f"  {'Metric':<28} {'Lexicon':>10}  {'TF-IDF+LR':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<28} {lex_res['accuracy']:>10.4f}  {tfidf_res['accuracy']:>10.4f}")
    print(f"  {'F1 Macro':<28} {lex_res['f1_macro']:>10.4f}  {tfidf_res['f1_macro']:>10.4f}")
    print(f"  {'F1 Weighted':<28} {lex_res['f1_weighted']:>10.4f}  {tfidf_res['f1_weighted']:>10.4f}")
    print(f"  {'Avg latency (ms/sample)':<28} {lex_res['avg_latency_ms']:>10.4f}  {tfidf_res['avg_latency_ms']:>10.4f}")
    print()
