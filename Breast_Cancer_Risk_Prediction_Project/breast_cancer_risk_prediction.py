# final_premium_pipeline.py
# Full pipeline: 6 ML models, 4 DL models, greedy forward feature selection, comparisons, plots, saves.

import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
import joblib

# ---------------- CONFIG ----------------
DATA_PATH = r"D:\INTRO TO PYTHON\Breast_Cancer_Risk_Prediction_Project\breast_cancer_risk_dataset_100k.csv"
TARGET_COL = "Risk_Label"   # confirmed
RESULTS_DIR = "results"
RANDOM_STATE = 42
GREEDY_K = 10   # how many features to choose in greedy analysis (you can set to X.shape[1])
DL_EPOCHS = 15
DL_BATCH = 64
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

# ---------------- PREPARE X,y ----------------
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns.")
y = df[TARGET_COL].copy()
X = df.drop(columns=[TARGET_COL]).copy()

# if any non-numeric columns, one-hot encode (keeps all features)
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("One-hot encoding non-numeric columns:", non_numeric)
    X = pd.get_dummies(X, drop_first=False)

# encode label if needed
if y.dtype == object or y.dtype == bool:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Label classes:", le.classes_)

print("Final feature count:", X.shape[1])

# ---------------- SPLIT & SCALE ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.joblib"))

# ---------------- DEFINE MODELS ----------------
ml_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB()
}

# ---------------- GREEDY FORWARD FEATURE SELECTION (analysis only) ----------------
def greedy_forward_selection(X_df, y_arr, estimator, k=10, cv=3):
    features = list(X_df.columns)
    remaining = features.copy()
    selected = []
    scores = []
    print(f"\nRunning greedy forward selection (k={k}) with evaluator={estimator.__class__.__name__} ...")
    for step in range(min(k, len(features))):
        best_feat, best_score = None, -np.inf
        for feat in remaining:
            cand = selected + [feat]
            # evaluate with cross_val on original data (not scaled here) but estimator expects numeric -> it's fine
            score = cross_val_score(clone(estimator), X_df[cand], y_arr, cv=cv, scoring='accuracy').mean()
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        remaining.remove(best_feat)
        scores.append(best_score)
        print(f"Step {step+1}: Added '{best_feat}' | CV accuracy = {best_score:.4f}")
    return selected, scores

# Use RandomForest as the greedy evaluator (fast & robust)
greedy_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
selected_feats, feat_scores = greedy_forward_selection(X, y, greedy_estimator, k=GREEDY_K, cv=4)
pd.DataFrame({"feature": selected_feats, "cv_score": feat_scores}).to_csv(os.path.join(RESULTS_DIR, "greedy_selected_features.csv"), index=False)

# If greedy selected fewer than all features, we will still keep ALL features for 'all-features' run.
print("\nGreedy-selected features (top):", selected_feats)

# ---------------- TRAIN & EVALUATE ML MODELS (ALL FEATURES) ----------------
ml_results_all = []
for name, model in ml_models.items():
    print(f"\nTraining ML model (ALL features): {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    ml_results_all.append({"Model": name, "Accuracy": acc, "Precision_Recall_F1": None})
    # save model
    joblib.dump(model, os.path.join(RESULTS_DIR, f"{name}_all.joblib"))
    # save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(os.path.join(RESULTS_DIR, f"{name}_all_classif_report.csv"))

ml_all_df = pd.DataFrame(ml_results_all).sort_values(by="Accuracy", ascending=False)
ml_all_df["Accuracy_pct"] = (ml_all_df["Accuracy"]*100).round(2)
ml_all_df.to_csv(os.path.join(RESULTS_DIR, "ml_results_all_features.csv"), index=False)
print("\nML results (ALL features):")
print(ml_all_df[["Model","Accuracy_pct"]])

# ---------------- TRAIN & EVALUATE ML MODELS (GREEDY FEATURES) ----------------
# prepare scaled data for greedy features
Xg = X[selected_feats].copy()
Xg_train, Xg_test = Xg.loc[X_train.index], Xg.loc[X_test.index]
scaler_g = StandardScaler().fit(Xg_train)
Xg_train_s = scaler_g.transform(Xg_train)
Xg_test_s = scaler_g.transform(Xg_test)
joblib.dump(scaler_g, os.path.join(RESULTS_DIR, "scaler_greedy.joblib"))

ml_results_greedy = []
for name, model in ml_models.items():
    print(f"\nTraining ML model (GREEDY features): {name}")
    model.fit(Xg_train_s, y_train)
    y_pred = model.predict(Xg_test_s)
    acc = accuracy_score(y_test, y_pred)
    ml_results_greedy.append({"Model": name, "Accuracy": acc})
    joblib.dump(model, os.path.join(RESULTS_DIR, f"{name}_greedy.joblib"))
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).to_csv(os.path.join(RESULTS_DIR, f"{name}_greedy_classif_report.csv"))

ml_g_df = pd.DataFrame(ml_results_greedy).sort_values(by="Accuracy", ascending=False)
ml_g_df["Accuracy_pct"] = (ml_g_df["Accuracy"]*100).round(2)
ml_g_df.to_csv(os.path.join(RESULTS_DIR, "ml_results_greedy_features.csv"), index=False)
print("\nML results (GREEDY features):")
print(ml_g_df[["Model","Accuracy_pct"]])

# ---------------- DEEP LEARNING MODELS ----------------
tf_available = False
try:
    import tensorflow as tf
    from tensorflow import keras
    tf_available = True
    print("\nTensorFlow available — using tf.keras for DL models.")
except Exception as e:
    print("\nTensorFlow NOT available. Falling back to sklearn MLP variants as DL models.")

dl_results_all = []
dl_results_greedy = []

if tf_available:
    input_dim_all = X_train_scaled.shape[1]
    input_dim_g = Xg_train_s.shape[1]

    # Helper to evaluate Keras model
    def train_eval_keras(builder, Xtr, ytr, Xte, yte, epochs=DL_EPOCHS, batch=DL_BATCH):
        model = builder()
        model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, validation_split=0.1, verbose=0)
        yproba = model.predict(Xte)
        ypred = (yproba > 0.5).astype(int).reshape(-1)
        acc = accuracy_score(yte, ypred)
        return model, acc, ypred

    # builders
    def build_ann_small():
        m = keras.Sequential([keras.layers.Input(shape=(input_dim_all,)),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); return m

    def build_ann_deep():
        m = keras.Sequential([keras.layers.Input(shape=(input_dim_all,)),
                              keras.layers.Dense(256, activation='relu'),
                              keras.layers.Dense(128, activation='relu'),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); return m

    def build_cnn1d_all():
        m = keras.Sequential([keras.layers.Input(shape=(input_dim_all,1)),
                              keras.layers.Conv1D(32,3,activation='relu'),
                              keras.layers.MaxPool1D(2),
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); return m

    def build_lstm_all():
        m = keras.Sequential([keras.layers.Input(shape=(input_dim_all,1)),
                              keras.layers.LSTM(64),
                              keras.layers.Dense(32, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); return m

    # Train DL on ALL features (reshape where needed)
    Xtr_all = X_train_scaled
    Xte_all = X_test_scaled

    # ANN small
    m1, a1, ypred1 = train_eval_keras(build_ann_small, Xtr_all, y_train, Xte_all, y_test)
    m1.save(os.path.join(RESULTS_DIR, "DL_ANN_small_all.h5"))
    dl_results_all.append({"Model":"ANN_small","Accuracy":a1})

    # ANN deep
    m2, a2, ypred2 = train_eval_keras(build_ann_deep, Xtr_all, y_train, Xte_all, y_test)
    m2.save(os.path.join(RESULTS_DIR, "DL_ANN_deep_all.h5"))
    dl_results_all.append({"Model":"ANN_deep","Accuracy":a2})

    # CNN1D (reshape)
    Xtr_cnn = Xtr_all.reshape((Xtr_all.shape[0], Xtr_all.shape[1], 1))
    Xte_cnn = Xte_all.reshape((Xte_all.shape[0], Xte_all.shape[1], 1))
    m3, a3, ypred3 = train_eval_keras(build_cnn1d_all, Xtr_cnn, y_train, Xte_cnn, y_test)
    m3.save(os.path.join(RESULTS_DIR, "DL_CNN1D_all.h5"))
    dl_results_all.append({"Model":"CNN1D","Accuracy":a3})

    # LSTM (reshape)
    m4, a4, ypred4 = train_eval_keras(build_lstm_all, Xtr_cnn, y_train, Xte_cnn, y_test)
    m4.save(os.path.join(RESULTS_DIR, "DL_LSTM_all.h5"))
    dl_results_all.append({"Model":"LSTM","Accuracy":a4})

    # Now DL on GREEDY features (if any)
    if len(selected_feats) > 0:
        Xtr_g, Xte_g = Xg_train_s, Xg_test_s
        # adjust builders for greedy dim
        def build_ann_small_g():
            return keras.Sequential([keras.layers.Input(shape=(Xtr_g.shape[1],)),
                                     keras.layers.Dense(64, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')])
        def build_ann_deep_g():
            return keras.Sequential([keras.layers.Input(shape=(Xtr_g.shape[1],)),
                                     keras.layers.Dense(128, activation='relu'),
                                     keras.layers.Dense(64, activation='relu'),
                                     keras.layers.Dense(32, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')])
        def build_cnn1d_g():
            return keras.Sequential([keras.layers.Input(shape=(Xtr_g.shape[1],1)),
                                     keras.layers.Conv1D(32,3,activation='relu'),
                                     keras.layers.MaxPool1D(2),
                                     keras.layers.Flatten(),
                                     keras.layers.Dense(32, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')])
        def build_lstm_g():
            return keras.Sequential([keras.layers.Input(shape=(Xtr_g.shape[1],1)),
                                     keras.layers.LSTM(32),
                                     keras.layers.Dense(16, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')])
        # train
        mg1, ag1, _ = train_eval_keras(build_ann_small_g, Xtr_g, y_train, Xte_g, y_test)
        mg1.save(os.path.join(RESULTS_DIR, "DL_ANN_small_greedy.h5")); dl_results_greedy.append({"Model":"ANN_small_g","Accuracy":ag1})
        mg2, ag2, _ = train_eval_keras(build_ann_deep_g, Xtr_g, y_train, Xte_g, y_test)
        mg2.save(os.path.join(RESULTS_DIR, "DL_ANN_deep_greedy.h5")); dl_results_greedy.append({"Model":"ANN_deep_g","Accuracy":ag2})
        Xtrg_c = Xtr_g.reshape((Xtr_g.shape[0], Xtr_g.shape[1], 1))
        Xteg_c = Xte_g.reshape((Xte_g.shape[0], Xte_g.shape[1], 1))
        mg3, ag3, _ = train_eval_keras(build_cnn1d_g, Xtrg_c, y_train, Xteg_c, y_test)
        mg3.save(os.path.join(RESULTS_DIR, "DL_CNN1D_greedy.h5")); dl_results_greedy.append({"Model":"CNN1D_g","Accuracy":ag3})
        mg4, ag4, _ = train_eval_keras(build_lstm_g, Xtrg_c, y_train, Xteg_c, y_test)
        mg4.save(os.path.join(RESULTS_DIR, "DL_LSTM_greedy.h5")); dl_results_greedy.append({"Model":"LSTM_g","Accuracy":ag4})

    dl_all_df = pd.DataFrame(dl_results_all).sort_values(by="Accuracy", ascending=False)
    dl_g_df = pd.DataFrame(dl_results_greedy).sort_values(by="Accuracy", ascending=False) if dl_results_greedy else pd.DataFrame()
    dl_all_df.to_csv(os.path.join(RESULTS_DIR, "dl_results_all_features.csv"), index=False)
    if not dl_g_df.empty: dl_g_df.to_csv(os.path.join(RESULTS_DIR, "dl_results_greedy_features.csv"), index=False)

else:
    # TF not available -> fallback to sklearn MLP variants for DL experiments
    print("\nUsing sklearn MLP fallbacks for DL experiments.")
    mlp_variants = [
        ("MLP_large", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=800, random_state=RANDOM_STATE)),
        ("MLP_medium", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=800, random_state=RANDOM_STATE)),
        ("MLP_small", MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=RANDOM_STATE)),
        ("MLP_deep", MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=800, random_state=RANDOM_STATE))
    ]
    for name, clf in mlp_variants:
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        dl_results_all.append({"Model":name,"Accuracy":acc})
        joblib.dump(clf, os.path.join(RESULTS_DIR, f"{name}_all.joblib"))
    # greedy DL: train on greedy features as well
    if len(selected_feats)>0:
        for name, clf in mlp_variants:
            clf.fit(Xg_train_s, y_train)
            acc = accuracy_score(y_test, clf.predict(Xg_test_s))
            dl_results_greedy.append({"Model": name+"_g", "Accuracy": acc})
            joblib.dump(clf, os.path.join(RESULTS_DIR, f"{name}_greedy.joblib"))
    dl_all_df = pd.DataFrame(dl_results_all).sort_values(by="Accuracy", ascending=False)
    dl_g_df = pd.DataFrame(dl_results_greedy).sort_values(by="Accuracy", ascending=False) if dl_results_greedy else pd.DataFrame()
    dl_all_df.to_csv(os.path.join(RESULTS_DIR, "dl_results_all_features.csv"), index=False)
    if not dl_g_df.empty: dl_g_df.to_csv(os.path.join(RESULTS_DIR, "dl_results_greedy_features.csv"), index=False)

# ---------------- HYBRID MODELS ----------------
# Hybrid A: features from best DL (if TF available) or PCA -> SVM
hybrid_results = []
# PCA->SVM hybrid (works always)
pca = PCA(n_components=min(50, X_train_scaled.shape[1]), random_state=RANDOM_STATE)
Xp_train = pca.fit_transform(X_train_scaled); Xp_test = pca.transform(X_test_scaled)
svm_pca = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
svm_pca.fit(Xp_train, y_train)
acc_h_pca = accuracy_score(y_test, svm_pca.predict(Xp_test))
hybrid_results.append({"Model":"PCA->SVM","Accuracy":acc_h_pca})
joblib.dump(svm_pca, os.path.join(RESULTS_DIR, "hybrid_PCA_SVM.joblib"))

# If TF built a model and you extracted features, do DNN->SVM hybrid (we attempted earlier if TF available)
# For simplicity, we'll use PCA features as second hybrid too (already done). Add stacking hybrid:
from sklearn.ensemble import StackingClassifier
stack = StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier())],
                          final_estimator=LogisticRegression(), cv=5)
stack.fit(X_train_scaled, y_train)
acc_stack = accuracy_score(y_test, stack.predict(X_test_scaled))
hybrid_results.append({"Model":"StackingHybrid","Accuracy":acc_stack})
joblib.dump(stack, os.path.join(RESULTS_DIR, "hybrid_stacking.joblib"))

hyb_df = pd.DataFrame(hybrid_results).sort_values(by="Accuracy", ascending=False)
hyb_df.to_csv(os.path.join(RESULTS_DIR, "hybrid_results.csv"), index=False)
print("\nHybrid results:")
print(hyb_df)

# ---------------- AGGREGATE & SAVE SUMMARIES ----------------
# Read ML & DL results tables saved above, combine for final ranking
ml_all = pd.read_csv(os.path.join(RESULTS_DIR, "ml_results_all_features.csv"))
ml_g = pd.read_csv(os.path.join(RESULTS_DIR, "ml_results_greedy_features.csv"))
dl_all = pd.read_csv(os.path.join(RESULTS_DIR, "dl_results_all_features.csv"))
dl_g = pd.read_csv(os.path.join(RESULTS_DIR, "dl_results_greedy_features.csv")) if os.path.exists(os.path.join(RESULTS_DIR, "dl_results_greedy_features.csv")) else pd.DataFrame()

# normalize column names and combine
ml_all["Type"]="ML_all"; ml_g["Type"]="ML_greedy"
dl_all["Type"]="DL_all"; 
if not dl_g.empty: dl_g["Type"]="DL_greedy"
hyb_df["Type"]="Hybrid"

combined = pd.concat([ml_all[["Model","Accuracy","Type"]], ml_g[["Model","Accuracy","Type"]],
                      dl_all[["Model","Accuracy","Type"]], dl_g[["Model","Accuracy","Type"]], hyb_df[["Model","Accuracy","Type"]]], ignore_index=True, sort=False)
combined.to_csv(os.path.join(RESULTS_DIR, "combined_model_ranking.csv"), index=False)

# ---------------- PLOTS ----------------
# Bar chart: ML all vs ML greedy
plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="Accuracy", data=ml_all)
plt.xticks(rotation=45); plt.title("ML Models - Accuracy (All features)")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "ml_all_accuracy.png")); plt.close()

plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="Accuracy", data=ml_g)
plt.xticks(rotation=45); plt.title("ML Models - Accuracy (Greedy features)")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "ml_greedy_accuracy.png")); plt.close()

if not dl_all.empty:
    plt.figure(figsize=(10,5))
    sns.barplot(x="Model", y="Accuracy", data=dl_all)
    plt.xticks(rotation=45); plt.title("DL Models - Accuracy (All features)")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "dl_all_accuracy.png")); plt.close()

# Confusion matrix for best ML (all)
best_ml_name = ml_all.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
best_ml_model = joblib.load(os.path.join(RESULTS_DIR, f"{best_ml_name}_all.joblib"))
y_pred_best = best_ml_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title(f"Confusion - {best_ml_name} (All)"); plt.savefig(os.path.join(RESULTS_DIR, "best_ml_confusion.png")); plt.close()

# Save combined summary to excel/csv
combined.to_csv(os.path.join(RESULTS_DIR, "final_combined_results.csv"), index=False)

print("\nAll done. Results saved in folder:", RESULTS_DIR)
print("Files:", os.listdir(RESULTS_DIR))
