# File Description
1. "main.py": main function to train/evaluate models
2. "model.py": model impelementation
3. "load_data.py": data loading
4. "shapley_value.py": caluculate shapley values with given inputs and models

# How to run
1. Change the variable "MODEL" in "main.py", and run "python main.py" to get the prediction errors for the corresponding MODEL.
2. Run 'python shapley_value.py' to calculate the shapley value for a given state (in variable "TARGET_STATE") with the saved MODEL.

# Dependencies
pip install shap

# Data
us_data.csv
