import joblib
import numpy as np
import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline     import Pipeline

class StackedXGBFNN:
    """
    Full end-to-end predictor:
      raw input  ─►  SK-learn Pipeline (scaling + OHE)  ─►  XGBoost  ─►  FNN  ─►  prediction
    """
    def __init__(self, preprocess_pipe, xgb_model, fnn_model):
        self.preprocess_pipe = preprocess_pipe   # your scikit-learn Pipeline
        self.xgb_model       = xgb_model         # fitted XGBoost
        self.fnn_model       = fnn_model         # fitted Keras FNN

    # ---------- public inference APIs ----------
    def predict_proba(self, X_raw):
        X_proc   = self.preprocess_pipe.transform(X_raw)                           # 1. preprocessing
        xgb_prob = self.xgb_model.predict_proba(X_proc)[:, 1].reshape(-1, 1)       # 2. XGB probs
        xgb_prob_stacked = np.hstack([X_proc, xgb_prob])                           # 3. Stack XGBoost Output onto Original Features
        return self.fnn_model.predict(xgb_prob_stacked)                            # 4. final XGBoost + FNN probs

    def predict(self, X_raw):
        return (self.predict_proba(X_raw) > 0.5).astype(int)  # Dichotomic prediction

    # ---------- persistence ----------
    def save(self, path):
        joblib.dump(self.preprocess_pipe, f"{path}/prep.pkl")
        joblib.dump(self.xgb_model,       f"{path}/xgb.pkl")
        self.fnn_model.save(f"{path}/fnn.keras")

    @staticmethod
    def load(path):
        prep = joblib.load(f"{path}/prep.pkl")
        xgbm = joblib.load(f"{path}/xgb.pkl")
        fnn  = tf.keras.models.load_model(f"{path}/fnn.keras")
        return StackedXGBFNN(prep, xgbm, fnn)
    