# Student Performance Prediction using Machine Learning

A small Streamlit web app that estimates a student's score using a Random Forest regression model.

## Run on Windows
1. Install Python 3.10 or newer and tick **Add Python to PATH** during installation.
2. Extract this folder.
3. Double-click `run_app.bat`, or open a terminal in this folder and run:
   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
4. Open the local URL shown in the terminal (usually http://localhost:8501).

## Files
- `app.py`: interactive website
- `student_performance_model.joblib`: trained model pipeline
- `train_model.py`: reproducible training script
- `synthetic_student_data.csv`: 6,000 generated sample records
- `requirements.txt`: Python dependencies
- `run_app.bat`: Windows one-click launcher

## Important limitation
The included training dataset is synthetic and generated for demonstration. Predictions are illustrative, not validated for real students, grading, admissions, or other high-stakes decisions. For a meaningful model, replace it with a properly consented, representative dataset and evaluate it before use.
