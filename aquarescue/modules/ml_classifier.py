import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SonarClassifier:
    """
    Trains on synthetically generated feature vectors
    with ground-truth labels, then predicts on new data.
    """
    MODEL_MAP = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":               SVC(probability=True, kernel="rbf", C=2.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }

    def __init__(self, model_name="Random Forest"):
        self.model_name = model_name
        self.model      = self.MODEL_MAP.get(model_name, self.MODEL_MAP["Random Forest"])
        self.scaler     = StandardScaler()
        self._trained   = False
        self.last_training_report = None

    def _synthetic_training_data(self, n_samples=2000):
        """Generate labelled feature vectors for supervised training."""
        X, y = [], []
        for label in [0, 1]:   # 0 = debris, 1 = human
            for _ in range(n_samples // 2):
                if label == 1:
                    height   = np.random.uniform(0.8, 2.5)
                    width    = np.random.uniform(0.3, 0.9)
                    density  = np.random.uniform(10, 300)
                    intmean  = np.random.uniform(0.55, 0.95)
                    dopvar   = np.random.uniform(0.4, 3.5)
                    sym      = np.random.uniform(0.50, 0.95)
                else:
                    height   = np.random.uniform(0.2, 5.0)
                    width    = np.random.uniform(0.2, 4.0)
                    density  = np.random.uniform(1,  200)
                    intmean  = np.random.uniform(0.2, 0.75)
                    dopvar   = np.random.uniform(0.0, 0.5)
                    sym      = np.random.uniform(0.1, 0.70)
                X.append([height, width, density, intmean, dopvar, sym])
                y.append(label)
        return np.array(X), np.array(y)

    def train(self, n_samples=2000, test_size=0.25, random_state=42):
        X, y = self._synthetic_training_data(n_samples=n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)
        self.model.fit(X_train_sc, y_train)

        y_pred = self.model.predict(X_test_sc)
        y_proba = self.model.predict_proba(X_test_sc)
        self.last_training_report = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Debris", "Human"],
                output_dict=True,
                zero_division=0,
            ),
            "avg_confidence": float(np.mean(np.max(y_proba, axis=1))),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "model": self.model_name,
        }
        self._trained = True

        return self.last_training_report

    def predict(self, features):
        if not self._trained:
            self.train()
        vec  = [[features["height"], features["width"], features["density"],
                  features["intensity_mean"], features["doppler_variance"], features["symmetry_score"]]]
        vec_sc = self.scaler.transform(vec)
        pred   = self.model.predict(vec_sc)[0]
        proba  = self.model.predict_proba(vec_sc)[0]
        label  = "Human" if pred == 1 else "Debris"
        conf   = float(max(proba))
        return {"class": label, "confidence": round(conf, 4), "model": self.model_name}