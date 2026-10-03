"""
Machine Learning - Homework 1 (University of Ioannina, 2021-22)
Monster type recognition  (Kaggle: "Ghouls, Goblins, and Ghosts... Boo!")

Goal: predict whether a "monster" is a Ghoul, a Goblin or a Ghost from 5
features. This single file implements ALL the methods required by the
assignment:

    Method 1: k-NN (Euclidean distance), k = 1, 3, 5, 10
    Method 2: Neural networks (sigmoid, softmax output layer, SGD)
              - 1 hidden layer  : K = 50, 100, 200
              - 2 hidden layers : (K1, K2) = (50, 25), (100, 50), (200, 100)
    Method 3: SVM (one-versus-rest) with a linear and a Gaussian RBF kernel
    Method 4: Naive Bayes (normal distribution for the 4 continuous features,
              multinomial distribution for the color)

Structure of train.csv:
    id | bone_length | rotting_flesh | hair_length | has_soul | color | type

How the program works:
    * The true labels of test.csv are hidden (only Kaggle knows them), so
      every model is evaluated LOCALLY on a part (20%) of train.csv that we
      keep aside (validation set).
    * Then the model is trained again on the WHOLE of train.csv and writes a
      submission file (id,type) that can be uploaded to Kaggle to obtain the
      official accuracy.

Run:   python src/monster_classifiers.py
"""

import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# =============================================================================
# SETTINGS (change them here, at the top of the file)
# =============================================================================
DATA_DIR = "data"              # folder containing train.csv and test.csv
OUT_DIR = "submissions"        # folder where the Kaggle submission files are written
VAL_SIZE = 0.2                 # fraction of the train set kept for validation
SEED = 42                      # fixed seed so that the results are reproducible
EPOCHS = 50                    # training epochs of the neural networks
RUN_NEURAL_NETWORKS = True     # set to False to skip the (slower) neural networks

# =============================================================================
# STEP 1: Load and prepare the data
# =============================================================================
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# The first 4 features are continuous numbers in [0, 1]. The 5th one (color)
# is text with 6 possible values. As suggested by the assignment, we replace
# it with {1/6, 2/6, ..., 1} so that it also lies in [0, 1].
COLOR_VALUES = {
    "clear": 1 / 6,
    "white": 2 / 6,
    "green": 3 / 6,
    "blue": 4 / 6,
    "black": 5 / 6,
    "blood": 6 / 6,
}
train["color"] = train["color"].map(COLOR_VALUES)
test["color"] = test["color"].map(COLOR_VALUES)

# If a color was not recognised, map() would have produced a missing value (NaN).
if train["color"].isna().any() or test["color"].isna().any():
    raise ValueError("Unknown color found in the data.")

# Select the 5 feature columns. The id column is NOT used for training (it is
# just an identification number), only in the output files.
FEATURES = ["bone_length", "rotting_flesh", "hair_length", "has_soul", "color"]

X_train_full = train[FEATURES].to_numpy()   # train inputs (all rows)
y_train_full = train["type"].to_numpy()     # train labels: Ghoul/Goblin/Ghost
X_test = test[FEATURES].to_numpy()          # test inputs (no labels)
test_ids = test["id"].to_numpy()            # test ids, used in the submission files

# Split the train set into "training" (80%) and "validation" (20%) parts.
# stratify keeps the same Ghoul/Goblin/Ghost proportions in both parts.
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SIZE, stratify=y_train_full, random_state=SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# STEP 2: The 4 classification methods
# Each method is a function that trains on (X_fit, y_fit) and returns its
# predictions for the samples X_pred.
# =============================================================================

# ---------------------------- Method 1: k-NN ---------------------------------
def knn_predict(k, X_fit, y_fit, X_pred):
    """k nearest neighbors with Euclidean distance.
    weights='distance': closer neighbors vote with a larger weight."""
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
    model.fit(X_fit, y_fit)
    return model.predict(X_pred)


# ------------------------ Method 2: Neural networks --------------------------
def nn_predict(hidden_layers, X_fit, y_fit, X_pred):
    """Neural network (Keras).
    hidden_layers: list with the number of neurons of each hidden layer,
                   e.g. [100] = 1 layer with 100 neurons, [100, 50] = 2 layers."""
    from tensorflow import keras   # imported here because it is slow to load

    keras.utils.set_random_seed(SEED)

    # The network works with numbers, not text: Ghost/Ghoul/Goblin -> 0/1/2.
    # We keep the encoder to convert numbers back to names later.
    encoder = LabelEncoder()
    y_numbers = encoder.fit_transform(y_fit)

    # Architecture: 5 inputs -> hidden layers (sigmoid) -> 3 outputs (softmax).
    # Softmax gives, for each monster, the probability of belonging to each class.
    layers = [keras.Input(shape=(5,))]
    for neurons in hidden_layers:
        layers.append(keras.layers.Dense(neurons, activation="sigmoid"))
    layers.append(keras.layers.Dense(3, activation="softmax"))
    model = keras.Sequential(layers)

    # Training with Stochastic Gradient Descent (optimizer="sgd").
    # sparse_categorical_crossentropy works with integer labels 0, 1, 2.
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    model.fit(X_fit.astype("float32"), y_numbers, epochs=EPOCHS, verbose=0)

    # For each monster keep the class with the highest probability (argmax)
    # and convert it back to its name (inverse_transform).
    probabilities = model.predict(X_pred.astype("float32"), verbose=0)
    return encoder.inverse_transform(np.argmax(probabilities, axis=1))


# ------------------------------ Method 3: SVM --------------------------------
def svm_predict(kernel, gamma, X_fit, y_fit, X_pred):
    """Support Vector Machine with the one-versus-rest (OVR) strategy.
    An SVM separates only 2 classes by nature; for 3 classes, 3 models are
    trained (each class against all the others)."""
    if kernel == "linear":
        # Linear kernel. C=10: larger penalty for misclassified samples.
        base = SVC(kernel="linear", C=10, class_weight="balanced", random_state=SEED)
    else:
        # Gaussian RBF kernel. The gamma parameter controls how "narrow" the
        # kernel is (large gamma = more complex decision boundary).
        base = SVC(kernel="rbf", gamma=gamma, class_weight="balanced", random_state=SEED)
    model = OneVsRestClassifier(base)
    model.fit(X_fit, y_fit)
    return model.predict(X_pred)


# -------------------------- Method 4: Naive Bayes ----------------------------
def naive_bayes_predict(X_fit, y_fit, X_pred):
    """Naive Bayes implemented "by hand".

    Bayes rule:          P(class | x)  ~  P(class) * P(x | class)
    "Naive" assumption: the features are independent of each other, so
        P(x | class) = P(x1|class) * P(x2|class) * ... * P(x5|class)
    - x1..x4 (continuous): normal distribution with a mean and a variance per class
    - x5 (color)         : multinomial distribution (how frequent each color is)
    For numerical stability we work with logarithms: products of
    probabilities become sums of logarithms."""
    classes = np.unique(y_fit)              # ["Ghost", "Ghoul", "Goblin"]
    n_colors = len(COLOR_VALUES)            # 6 colors

    # The color is stored as {1/6,...,1}. Convert it back to an index 0..5 so
    # that it can be used as a position in a table of counts.
    color_fit = np.rint(X_fit[:, 4] * n_colors).astype(int) - 1
    color_pred = np.rint(X_pred[:, 4] * n_colors).astype(int) - 1

    scores = np.zeros((len(X_pred), len(classes)))   # score of each class

    for i, c in enumerate(classes):
        rows = X_fit[y_fit == c]                     # the samples of this class
        colors_c = color_fit[y_fit == c]

        # (a) Prior probability P(class): the fraction of samples in this class.
        log_prior = np.log(len(rows) / len(X_fit))

        # (b) Gaussian for the 4 continuous features: mean and variance
        # (ddof=1: divide by n-1, as in the formula of the report).
        mean = rows[:, :4].mean(axis=0)
        var = rows[:, :4].var(axis=0, ddof=1)
        x = X_pred[:, :4]
        log_gauss = -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)

        # (c) Multinomial for the color: count how many times each color
        # appears in the class. The +1 (Laplace smoothing) avoids a zero
        # probability for a color that never appeared in this class.
        counts = np.bincount(colors_c, minlength=n_colors)
        color_prob = (counts + 1) / (len(rows) + n_colors)
        log_color = np.log(color_prob)[color_pred]

        # Total score = log(prior) + log(Gaussians) + log(color)
        scores[:, i] = log_prior + log_gauss.sum(axis=1) + log_color

    # The predicted class is the one with the highest score.
    return classes[np.argmax(scores, axis=1)]


# =============================================================================
# STEP 3: Evaluate every model and create the Kaggle submission files
# =============================================================================
results = []   # the results of all the models are collected here


def run_model(name, file_name, predict_function):
    """Evaluates a model and writes its submission file.
    predict_function(X_fit, y_fit, X_pred) -> predictions."""
    # (1) Train on 80% of the train set, test on the other 20% (validation).
    y_val_pred = predict_function(X_tr, y_tr, X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    # F1/precision/recall are computed per class and then averaged with
    # weights (average="weighted"), because we have 3 classes.
    f1 = f1_score(y_val, y_val_pred, average="weighted")
    precision = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_val_pred, average="weighted")

    # (2) Train again on the WHOLE train set, predict the Kaggle test set.
    y_test_pred = predict_function(X_train_full, y_train_full, X_test)
    submission = pd.DataFrame({"id": test_ids, "type": y_test_pred})
    submission.to_csv(os.path.join(OUT_DIR, file_name), index=False)

    print(f"{name:<34} accuracy={accuracy:.4f}  F1={f1:.4f}  "
          f"precision={precision:.4f}  recall={recall:.4f}")
    results.append({"method": name, "accuracy": accuracy, "f1": f1,
                    "precision": precision, "recall": recall})


print(f"Train: {len(X_tr)} samples | Validation: {len(X_val)} | "
      f"Test (Kaggle): {len(X_test)}\n")

# partial(f, a) = "the function f with its first argument already set to a".
# This way every model gets the form predict_function(X_fit, y_fit, X_pred).

print("--- Method 1: k-NN ---")
for k in (1, 3, 5, 10):
    run_model(f"k-NN (k={k})", f"submission_knn_k{k}.csv", partial(knn_predict, k))

if RUN_NEURAL_NETWORKS:
    print("\n--- Method 2: Neural networks ---")
    try:
        import tensorflow  # check that it is installed
        # One hidden layer: K = 50, 100, 200
        for k in (50, 100, 200):
            run_model(f"Neural network K={k}", f"submission_nn_{k}.csv",
                      partial(nn_predict, [k]))
        # Two hidden layers: (K1, K2)
        for k1, k2 in ((50, 25), (100, 50), (200, 100)):
            run_model(f"Neural network (K1,K2)=({k1},{k2})",
                      f"submission_nn_{k1}-{k2}.csv", partial(nn_predict, [k1, k2]))
    except ImportError:
        print("TensorFlow is not installed: pip install tensorflow")

print("\n--- Method 3: SVM (one-versus-rest) ---")
run_model("SVM linear kernel (C=10)", "submission_svm_linear.csv",
          partial(svm_predict, "linear", None))
for gamma in ("scale", 0.1, 1, 10):   # try several values of gamma
    run_model(f"SVM RBF (gamma={gamma})", f"submission_svm_rbf_gamma_{gamma}.csv",
              partial(svm_predict, "rbf", gamma))

print("\n--- Method 4: Naive Bayes ---")
run_model("Naive Bayes (Gaussian + multinomial)", "submission_naive_bayes.csv",
          naive_bayes_predict)

# =============================================================================
# STEP 4: Compare the methods and find the best one
# =============================================================================
summary = pd.DataFrame(results).sort_values("accuracy", ascending=False)
summary.to_csv(os.path.join(OUT_DIR, "validation_summary.csv"), index=False)

print("\n=== Method comparison (validation, best first) ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = summary.iloc[0]
print(f"\nBest method on the validation set: {best['method']} "
      f"(accuracy = {best['accuracy']:.4f})")
print(f"The files for Kaggle are in the '{OUT_DIR}/' folder. "
      "Upload them to get the official accuracy.")
