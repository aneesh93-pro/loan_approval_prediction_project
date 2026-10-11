# Trained model

`decision_tree_project.pkl` is the pickle supplied with this project.

Python pickle/joblib model files are version-sensitive. Load it only in a trusted environment. The supplied model could not be loaded in the packaging environment because of a scikit-learn internal compatibility error. The reproducible fix is to install the compatible scikit-learn version in `requirements.txt` or retrain it with `python src/train.py`; that script overwrites the model with a freshly trained pipeline.
