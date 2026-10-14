"""
Pipeline ML: Stacking (XGBoost + Random Forest -> Logistic Regression)
dengan cost-sensitive learning, threshold tuning per kelas, dan kalibrasi
untuk deteksi dini prediabetes & diabetes tipe 2 (BRFSS 2015, 3 kelas).

Cara pakai:
    python pipeline_prediabetes.py --data diabetes_012_health_indicators_BRFSS2015.csv
    python pipeline_prediabetes.py --synthetic        # uji pipeline tanpa dataset asli
Tambahan: --quick (model kecil, untuk uji cepat)

Output (folder hasil/): results.csv, confusion_*.csv, pr_threshold.csv,
                        model_final.joblib, config.json
"""
import argparse, json, os, itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             balanced_accuracy_score, accuracy_score,
                             confusion_matrix, average_precision_score)
from xgboost import XGBClassifier

SEED = 42
TARGET = "Diabetes_012"
CLASSES = [0, 1, 2]  # 0 tidak diabetes, 1 prediabetes, 2 diabetes

# Fitur yang lebih "transferable" ke konteks Indonesia (tanpa Income,
# Education, AnyHealthcare, NoDocbcCost). Sesuaikan jika perlu.
SUBSET = ["HighBP", "HighChol", "BMI", "Smoker", "Stroke",
          "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
          "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth",
          "DiffWalk", "Sex", "Age"]


# ---------------------------------------------------------------- data
def load_data(path, synthetic, drop_dup):
    if synthetic:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=20000, n_features=21, n_informative=8,
                                   n_classes=3, n_clusters_per_class=1,
                                   weights=[0.84, 0.02, 0.14], class_sep=0.8,
                                   random_state=SEED)
        cols = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
        df = pd.DataFrame(X, columns=cols)
        df[TARGET] = y
    else:
        df = pd.read_csv(path)
    n0 = len(df)
    assert df.isna().sum().sum() == 0, "Ada missing value, tangani dulu"
    if drop_dup:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"Baris: {n0} -> {len(df)} (drop_duplicates={drop_dup})")
    df[TARGET] = df[TARGET].astype(int)
    print("Distribusi kelas (%):\n", (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))
    return df


# --------------------------------------------------------------- model
def build_stack(quick, seed=SEED):
    n_rf, n_xgb = (60, 80) if quick else (300, 400)
    rf = RandomForestClassifier(n_estimators=n_rf, max_depth=12, min_samples_leaf=5,
                                n_jobs=-1, random_state=seed)
    xgb = XGBClassifier(n_estimators=n_xgb, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                        eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
                        random_state=seed)
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    # Out-of-fold predictions (stratified 5-fold) -> meta-learner, anti data leakage
    return StackingClassifier(estimators=[("rf", rf), ("xgb", xgb)], final_estimator=meta,
                              stack_method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=1)


def fit_stack(model, X, y, cost_sensitive):
    """Cost-sensitive = sample_weight 'balanced' (bobot kelas minoritas lebih besar).
    sample_weight diteruskan ke RF, XGBoost, dan Logistic Regression."""
    if cost_sensitive:
        sw = compute_sample_weight("balanced", y)
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return model


# ------------------------------------------------------------- metrik
def gmean_recall(y, pred):
    r = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return float(np.prod(r) ** (1 / len(r)))


def brier_multiclass(y, proba):
    onehot = np.eye(len(CLASSES))[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(name, y, proba, pred, rows, cm_dir):
    rec = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    prec = precision_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    ap = [average_precision_score((y == c).astype(int), proba[:, i]) for i, c in enumerate(CLASSES)]
    rows.append({
        "skenario": name,
        "recall_sehat": rec[0], "recall_prediabetes": rec[1], "recall_diabetes": rec[2],
        "precision_prediabetes": prec[1], "f1_prediabetes": f1[1],
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "g_mean": gmean_recall(y, pred),
        "pr_auc_prediabetes": ap[1], "pr_auc_diabetes": ap[2],
        "brier": brier_multiclass(y, proba),
        "accuracy": accuracy_score(y, pred),
    })
    pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                 index=[f"aktual_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        os.path.join(cm_dir, f"confusion_{name}.csv"))


# ----------------------------------------- threshold tuning + kalibrasi
def apply_weights(proba, w):
    return np.argmax(proba * np.asarray(w), axis=1)


def tune_class_weights(y_val, proba_val, min_precision=0.0, grid=None):
    """Cari pengali probabilitas (w1 prediabetes, w2 diabetes) pada data VALIDASI.
    Tujuan: recall prediabetes maksimal dengan precision prediabetes >= min_precision.
    Jika min_precision=0 -> memaksimalkan macro F1."""
    grid = grid if grid is not None else np.round(np.concatenate([np.arange(1, 10, 0.5), np.arange(10, 41, 2.5)]), 2)
    best, best_score, table = (1.0, 1.0), -1, []
    for w1, w2 in itertools.product(grid, [0.5, 1, 1.5, 2, 3, 5]):
        pred = apply_weights(proba_val, [1, w1, w2])
        r1 = recall_score(y_val, pred, labels=[1], average=None, zero_division=0)[0]
        p1 = precision_score(y_val, pred, labels=[1], average=None, zero_division=0)[0]
        mf1 = f1_score(y_val, pred, average="macro")
        table.append({"w_prediabetes": w1, "w_diabetes": w2, "recall_prediabetes": r1,
                      "precision_prediabetes": p1, "macro_f1": mf1})
        score = mf1 if min_precision == 0 else (r1 if p1 >= min_precision else -1)
        if score > best_score:
            best, best_score = (w1, w2), score
    return [1.0, *best], pd.DataFrame(table)


class IsotonicOvR:
    """Kalibrasi isotonic one-vs-rest, lalu dinormalisasi agar jumlah = 1."""
    def fit(self, proba, y):
        self.cal = [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                    .fit(proba[:, i], (y == c).astype(int)) for i, c in enumerate(CLASSES)]
        return self

    def transform(self, proba):
        out = np.column_stack([c.predict(proba[:, i]) for i, c in enumerate(self.cal)])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="diabetes_012_health_indicators_BRFSS2015.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="default: duplikat dibuang. Laporkan pilihan ini di laporan.")
    ap.add_argument("--min-precision", type=float, default=0.0,
                    help="mis. 0.25 -> recall prediabetes maksimal dgn precision >= 0.25")
    ap.add_argument("--out", default="hasil")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_data(a.data, a.synthetic, drop_dup=not a.keep_duplicates)

    # 1) Split stratified 80/20 (test tidak disentuh sampai evaluasi akhir)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    # 2) Dari train, sisihkan 20% sebagai validasi untuk threshold tuning & kalibrasi
    fit_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[TARGET], random_state=SEED)

    rows, probas = [], {}
    feature_sets = {"full21": [c for c in df.columns if c != TARGET],
                    "subset": [c for c in SUBSET if c in df.columns]}

    for fs_name, feats in feature_sets.items():
        for csl in [False, True]:
            tag = f"{fs_name}_{'CSL' if csl else 'baseline'}"
            print(f"\n=== Melatih {tag} ({len(feats)} fitur) ===")
            m = fit_stack(build_stack(a.quick), fit_df[feats], fit_df[TARGET], csl)
            p_val, p_test = m.predict_proba(val_df[feats]), m.predict_proba(test_df[feats])
            y_val, y_test = val_df[TARGET].values, test_df[TARGET].values
            evaluate(tag, y_test, p_test, np.argmax(p_test, axis=1), rows, a.out)
            probas[tag] = (m, feats, p_val, p_test)

            if csl:
                # Threshold tuning per kelas (dipelajari di validasi, dievaluasi di test)
                w, table = tune_class_weights(y_val, p_val, a.min_precision)
                table.to_csv(os.path.join(a.out, f"pr_threshold_{tag}.csv"), index=False)
                print("Bobot keputusan terpilih [sehat, pre, diab]:", w)
                evaluate(tag + "_thr", y_test, p_test, apply_weights(p_test, w), rows, a.out)

                # Kalibrasi isotonic (fit di validasi) + threshold tuning ulang
                cal = IsotonicOvR().fit(p_val, y_val)
                pc_val, pc_test = cal.transform(p_val), cal.transform(p_test)
                w2, _ = tune_class_weights(y_val, pc_val, a.min_precision)
                evaluate(tag + "_cal_thr", y_test, pc_test, apply_weights(pc_test, w2), rows, a.out)
                probas[tag + "_final"] = (m, feats, cal, w2, w)

    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\n=== HASIL (data uji) ===")
    print(res[["skenario", "recall_prediabetes", "precision_prediabetes", "recall_diabetes",
               "macro_f1", "balanced_acc", "g_mean", "pr_auc_prediabetes", "brier"]].to_string(index=False))

    # Simpan model final: subset fitur + CSL + kalibrasi + threshold (untuk FastAPI)
    best_key = "subset_CSL_final"
    m, feats, cal, w2, _ = probas[best_key]
    joblib.dump({"model": m, "calibrator": cal, "class_weights": w2, "features": feats},
                os.path.join(a.out, "model_final.joblib"))
    json.dump({"features": feats, "class_weights": w2, "classes": CLASSES,
               "labels": ["tidak_diabetes", "prediabetes", "diabetes"]},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"\nModel final disimpan ({best_key}). Ganti best_key bila hasil evaluasi menunjuk skenario lain.")


if __name__ == "__main__":
    main()
