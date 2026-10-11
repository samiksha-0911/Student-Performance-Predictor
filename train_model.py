"""Train a demo student performance model on synthetic data."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

FEATURES = ["attendance", "study_hours", "previous_score", "assignment_completion", "sleep_hours", "extracurricular_hours"]
rng = np.random.default_rng(42)
n = 6000
attendance = rng.uniform(40, 100, n)
study_hours = rng.uniform(0, 8, n)
previous_score = rng.uniform(25, 100, n)
assignment_completion = rng.uniform(0, 100, n)
sleep_hours = rng.uniform(3, 10, n)
extracurricular_hours = rng.uniform(0, 5, n)
noise = rng.normal(0, 5, n)
score = (0.22 * attendance + 2.8 * study_hours + 0.36 * previous_score + 0.12 * assignment_completion + 1.8 * sleep_hours - 0.9 * extracurricular_hours + noise)
score = np.clip(score, 0, 100)
df = pd.DataFrame({"attendance": attendance, "study_hours": study_hours, "previous_score": previous_score, "assignment_completion": assignment_completion, "sleep_hours": sleep_hours, "extracurricular_hours": extracurricular_hours, "final_score": score})
df.to_csv("synthetic_student_data.csv", index=False)
X = df[FEATURES]
y = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
joblib.dump({"model": model, "features": FEATURES, "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)), "demo": True}, "student_performance_model.joblib")
print(f"Saved model. Demo holdout MAE={mean_absolute_error(y_test,pred):.2f}; R2={r2_score(y_test,pred):.3f}")
