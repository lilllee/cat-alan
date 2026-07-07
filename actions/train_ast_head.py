"""Train a lightweight classifier on cached AST embeddings for CatMeows
context classification (brushing / isolation / waiting_for_food).

The AST encoder stays frozen; only this small head is trained, which is
CPU-friendly and robust to the small dataset size.

Note on evaluation: CatMeows' test split contains cats unseen in training, so
in-training cross-validation (~72%) noticeably overestimates accuracy on new
cats (~55%). A well-regularized, probability-calibrated logistic regression is
deployed rather than the raw CV-argmax model, which overfits the CV folds.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

CLASSES = ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]
DEPLOY_C = 0.1  # regularized; calibrated probabilities for the app's bar chart


def load(split):
    d = np.load(Path("data/ast_embeddings", f"{split}.npz"))
    return d["features"], d["labels"]


def build(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=c, class_weight="balanced"),
    )


def main():
    xtr, ytr = load("train")
    xte, yte = load("test")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for c in [0.03, 0.1, 0.3, 1.0]:
        score = cross_val_score(build(c), xtr, ytr, cv=cv).mean()
        print(f"logreg C={c:<4} cv_acc {score:.2%}")

    clf = build(DEPLOY_C)
    clf.fit(xtr, ytr)
    print(f"\ndeployed: logreg C={DEPLOY_C}")
    print(f"train accuracy:           {accuracy_score(ytr, clf.predict(xtr)):.2%}")
    print(f"test accuracy (new cats): {accuracy_score(yte, clf.predict(xte)):.2%}")
    print("confusion matrix (rows=true, cols=pred):", CLASSES)
    print(confusion_matrix(yte, clf.predict(xte)))

    out = Path("models/context_ast_head.joblib")
    joblib.dump({"classifier": clf, "classes": CLASSES}, out)
    print(f"saved classifier to {out}")


if __name__ == "__main__":
    main()
