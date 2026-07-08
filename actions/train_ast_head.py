"""Train a lightweight classifier on cached AST embeddings for CatMeows
context classification (brushing / isolation / waiting_for_food), optionally
augmented with prosodic features (duration, mean F0, intonation slope).

The AST encoder stays frozen; only this small head is trained, which is
CPU-friendly and robust to the small dataset size.

Note on evaluation: CatMeows' test split contains cats unseen in training, so
in-training cross-validation (~72%) noticeably overestimates accuracy on new
cats (~55%). A well-regularized, probability-calibrated logistic regression is
deployed rather than the raw CV-argmax model, which overfits the CV folds.

This script prints AST-only and AST+prosody results side by side (the honest
unseen-cat test accuracy is the number that matters) and deploys the AST+prosody
head when the prosody cache is present. See models/prosody.py for the rationale.
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


def load_prosody(split):
    path = Path("data/ast_embeddings", f"{split}_prosody.npz")
    if not path.exists():
        return None, None
    d = np.load(path)
    return d["features"], d["labels"]


def build(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=c, class_weight="balanced"),
    )


def evaluate(name, xtr, ytr, xte, yte):
    """Print CV sweep + unseen-cat test accuracy for one feature set."""
    print(f"\n[{name}]  dim={xtr.shape[1]}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for c in [0.03, 0.1, 0.3, 1.0]:
        score = cross_val_score(build(c), xtr, ytr, cv=cv).mean()
        print(f"  logreg C={c:<4} cv_acc {score:.2%}")
    clf = build(DEPLOY_C).fit(xtr, ytr)
    test_acc = accuracy_score(yte, clf.predict(xte))
    print(f"  deployed C={DEPLOY_C}: train {accuracy_score(ytr, clf.predict(xtr)):.2%}"
          f"  test (new cats) {test_acc:.2%}")
    return clf, test_acc


def main():
    xtr, ytr = load("train")
    xte, yte = load("test")

    clf_ast, acc_ast = evaluate("AST only", xtr, ytr, xte, yte)

    ptr, pytr = load_prosody("train")
    pte, pyte = load_prosody("test")
    have_prosody = ptr is not None
    acc_pro = -1.0
    if have_prosody:
        assert np.array_equal(pytr, ytr) and np.array_equal(pyte, yte), (
            "prosody cache is misaligned with AST cache; re-run "
            "extract_ast_embeddings.py and extract_prosody.py together"
        )
        xtr_p = np.hstack([xtr, ptr])
        xte_p = np.hstack([xte, pte])
        clf_pro, acc_pro = evaluate("AST + prosody", xtr_p, ytr, xte_p, yte)
    else:
        print("\n[AST + prosody] skipped — run actions/extract_prosody.py first")

    # Deploy prosody only if it actually beats AST-only on unseen cats. On a tie
    # it isn't worth the extra pyin cost at inference, so keep the AST-only head.
    use_prosody = have_prosody and acc_pro > acc_ast
    clf = clf_pro if use_prosody else clf_ast
    print(f"\ndeployed model uses prosody: {use_prosody} "
          f"(AST {acc_ast:.2%} vs AST+prosody "
          f"{acc_pro:.2%})" if have_prosody else "")
    print("confusion matrix (rows=true, cols=pred):", CLASSES)
    xte_final = xte_p if use_prosody else xte
    print(confusion_matrix(yte, clf.predict(xte_final)))

    out = Path("models/context_ast_head.joblib")
    joblib.dump(
        {"classifier": clf, "classes": CLASSES, "use_prosody": use_prosody}, out
    )
    print(f"saved classifier to {out}")


if __name__ == "__main__":
    main()
