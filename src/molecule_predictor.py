# src/molecule_predictor.py
import numpy as np
import torch


class MoleculePredictor:
    """Predict molecular properties from formula or manual input"""

    def __init__(self, feature_scaler, target_scaler, angle_model, amp_model):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.angle_model = angle_model
        self.amp_model = amp_model

    def manual_features_to_array(self, features_dict):
        """Convert manual input features to array"""
        feature_order = ['MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors',
                         'NumRotatableBonds', 'NumHeteroatoms', 'HeavyAtomCount',
                         'NumSaturatedHeterocycles']

        features = [features_dict.get(f, 0) for f in feature_order]
        return np.array([features[:len(self.feature_scaler.feature_names_in_)]])

    def predict_from_features(self, features_array):
        """Predict solubility from feature array"""
        features_scaled = self.feature_scaler.transform(features_array)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            angle_pred_scaled = self.angle_model(features_tensor).numpy()
            amp_pred_scaled = self.amp_model(features_tensor).numpy()

        angle_pred = self.target_scaler.inverse_transform(angle_pred_scaled.reshape(-1, 1)).flatten()[0]
        amp_pred = self.target_scaler.inverse_transform(amp_pred_scaled.reshape(-1, 1)).flatten()[0]

        return angle_pred, amp_pred
