"""
Evaluate a pre-trained multilingual sentiment model on the labeled dataset.
Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Multilingual (EN + ZH both supported)
  - Outputs: positive / neutral / negative directly
  - ~278 MB download on first run (cached afterwards)

Usage:
    python evaluate_pretrained.py
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LABELED_SAMPLES

LABELS = ["positive", "neutral", "negative"]
# nlptown model outputs 1-5 star ratings (multilingual, ~700 MB)
# Stars 4-5 → positive, 3 → neutral, 1-2 → negative
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
SEP = "=" * 60


def evaluate_pretrained(samples):
    print(f"  Loading model: {MODEL_NAME}")
    print("  (first run will download ~700 MB; subsequent runs use cache)")

    t0 = time.perf_counter()
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=1,
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f} s")

    def star_to_label(star_label: str) -> str:
        """Map '1 star'–'5 stars' to positive/neutral/negative."""
        stars = int(star_label.split()[0])
        if stars >= 4:
            return "positive"
        if stars == 3:
            return "neutral"
        return "negative"

    y_true, y_pred = [], []
    total_infer = 0.0

    for text, label in samples:
        t0 = time.perf_counter()
        result = classifier(text, truncation=True, max_length=512)[0]
        total_infer += time.perf_counter() - t0
        # top_k=1 returns [[{label, score}]]; after [0] → [{label, score}]
        top = result[0] if isinstance(result, list) else result
        pred_label = star_to_label(top["label"])
        y_true.append(label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    avg_ms = (total_infer / len(samples)) * 1000

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": cm,
        "avg_latency_ms": avg_ms,
        "load_time_s": load_time,
    }


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  XLM-RoBERTa Pre-trained Sentiment Model Evaluation")
    print(f"{SEP}")
    res = evaluate_pretrained(LABELED_SAMPLES)

    print(f"\n  Accuracy             : {res['accuracy']:.4f}")
    print(f"  F1 Macro             : {res['f1_macro']:.4f}")
    print(f"  F1 Weighted          : {res['f1_weighted']:.4f}")
    print(f"  Model load time      : {res['load_time_s']:.2f} s")
    print(f"  Avg inference latency: {res['avg_latency_ms']:.2f} ms / sample")
    print()
    print("  Classification Report:")
    for line in res["report"].splitlines():
        print("    " + line)
    print()
    print("  Confusion Matrix  [pos / neu / neg]:")
    for i, row in enumerate(res["confusion_matrix"]):
        print(f"    {LABELS[i]:8s}: {row.tolist()}")
    print()
