# models/demand_models.py
"""
Demand Models for Multi-Room Hotel Pricing.

This module contains:
1. SingleRoomDemandModel: Logistic regression for one room type
2. MultiRoomDemandModel: Collection of demand models

Each model predicts: P(booking | daily_context, price)

UPDATED: Better handling of rooms with insufficient data
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Minimum positive samples required to train a model
MIN_POSITIVE_SAMPLES = 10


class SingleRoomDemandModel:
    """
    Demand model for a single room type.
    
    Uses logistic regression to predict P(booking | context, price).
    
    Args:
        room_type: Room type identifier (A, D, E, etc.)
        adr_ref: Reference ADR for this room type
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        room_type: str,
        adr_ref: float,
        seed: int = 42
    ):
        self.room_type = room_type
        self.adr_ref = adr_ref
        self.seed = seed
        
        self.model = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_context_features = None
        self.use_fallback = False  # If True, use simple heuristic instead of trained model
        
        # Metrics
        self.train_accuracy = None
        self.train_auc = None
        self.price_sensitivity = None
        self.n_positive = 0
        self.n_negative = 0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        price_col_idx: int = -1
    ) -> 'SingleRoomDemandModel':
        """
        Fit the demand model.
        
        Args:
            X: Feature matrix (context + price_ratio)
            y: Labels (1 = booked, 0 = not booked)
            price_col_idx: Index of price_ratio column (default: last)
        """
        self.price_col_idx = price_col_idx
        self.n_context_features = X.shape[1] - 1
        self.n_positive = int(y.sum())
        self.n_negative = int(len(y) - y.sum())
        
        # Check if we have enough samples of both classes
        if self.n_positive < MIN_POSITIVE_SAMPLES:
            print(f"  Warning: Room {self.room_type} has only {self.n_positive} positive samples.")
            print(f"           Using fallback heuristic model.")
            self.use_fallback = True
            self.is_fitted = True
            self.price_sensitivity = -1.0  # Default negative sensitivity
            self.train_auc = 0.5
            self.train_accuracy = 0.5
            
            # Still fit scaler for consistency
            self.scaler.fit(X)
            return self
        
        # Check if we have both classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"  Warning: Room {self.room_type} has only class {unique_classes[0]}.")
            print(f"           Using fallback heuristic model.")
            self.use_fallback = True
            self.is_fitted = True
            self.price_sensitivity = -1.0
            self.train_auc = 0.5
            self.train_accuracy = 0.5
            self.scaler.fit(X)
            return self
        
        # Normal training
        self.use_fallback = False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Compute metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        self.train_accuracy = accuracy_score(y, y_pred)
        self.train_auc = roc_auc_score(y, y_proba)
        
        # Get price sensitivity (coefficient of price_ratio)
        self.price_sensitivity = self.model.coef_[0][price_col_idx]
        
        return self
    
    def predict_proba(
        self,
        context: np.ndarray,
        price: float
    ) -> float:
        """
        Predict booking probability given context and price.
        
        Args:
            context: Context features
            price: Absolute price
            
        Returns:
            Probability of booking
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute price ratio
        price_ratio = price / self.adr_ref
        
        # If using fallback, return simple heuristic
        if self.use_fallback:
            # Simple sigmoid based on price ratio
            # P = sigmoid(-2 * (price_ratio - 1))
            # At price_ratio=1.0: P=0.5
            # At price_ratio=0.8: P=0.73
            # At price_ratio=1.2: P=0.27
            z = -2.0 * (price_ratio - 1.0)
            return 1.0 / (1.0 + np.exp(-z))
        
        # Normal prediction
        if context.ndim == 1:
            context = context.reshape(1, -1)
        
        # Truncate or pad context to expected size
        if context.shape[1] > self.n_context_features:
            context = context[:, :self.n_context_features]
        elif context.shape[1] < self.n_context_features:
            padding = np.zeros((context.shape[0], self.n_context_features - context.shape[1]))
            context = np.hstack([context, padding])
        
        # Append price ratio to context
        X = np.column_stack([context, np.full(len(context), price_ratio)])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        return proba[0] if len(proba) == 1 else proba
    
    def get_summary(self) -> Dict:
        """Get model summary statistics."""
        return {
            'room_type': self.room_type,
            'adr_ref': self.adr_ref,
            'train_accuracy': self.train_accuracy,
            'train_auc': self.train_auc,
            'price_sensitivity': self.price_sensitivity,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
            'use_fallback': self.use_fallback
        }


class MultiRoomDemandModel:
    """
    Collection of demand models for all room types.
    
    Manages separate demand models, one per room type.
    """
    
    def __init__(
        self,
        room_types: List[str],
        adr_refs: Dict[str, float],
        seed: int = 42
    ):
        self.room_types = room_types
        self.adr_refs = adr_refs
        self.seed = seed
        
        # Create individual models
        self.models: Dict[str, SingleRoomDemandModel] = {}
        for room in room_types:
            self.models[room] = SingleRoomDemandModel(
                room_type=room,
                adr_ref=adr_refs[room],
                seed=seed
            )
    
    def fit_all(
        self,
        training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        verbose: bool = True
    ) -> 'MultiRoomDemandModel':
        """
        Fit all demand models.
        
        Args:
            training_data: Dict mapping room_type -> (X, y)
            verbose: Print training progress
        """
        if verbose:
            print("\n" + "=" * 50)
            print("Training Demand Models (Phase 1)")
            print("=" * 50)
        
        for room in self.room_types:
            if room not in training_data:
                print(f"Warning: No training data for room {room}")
                continue
            
            X, y = training_data[room]
            self.models[room].fit(X, y)
            
            if verbose:
                summary = self.models[room].get_summary()
                status = " (fallback)" if summary['use_fallback'] else ""
                print(f"Room {room}: AUC={summary['train_auc']:.4f}, "
                      f"Price Sens={summary['price_sensitivity']:.4f}{status}")
        
        return self
    
    def predict_proba(
        self,
        room_type: str,
        context: np.ndarray,
        price: float
    ) -> float:
        """Predict booking probability for a specific room type."""
        if room_type not in self.models:
            raise ValueError(f"Unknown room type: {room_type}")
        return self.models[room_type].predict_proba(context, price)
    
    def predict_all_rooms(
        self,
        context: np.ndarray,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict booking probabilities for all room types.
        """
        probas = {}
        for room in self.room_types:
            if room in prices:
                probas[room] = self.predict_proba(room, context, prices[room])
        return probas
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all models as DataFrame."""
        summaries = [self.models[room].get_summary() for room in self.room_types]
        return pd.DataFrame(summaries)
    
    def verify_price_sensitivity(self, verbose: bool = True) -> bool:
        """
        Verify that all models have negative price sensitivity.
        """
        all_negative = True
        for room in self.room_types:
            sens = self.models[room].price_sensitivity
            if sens is not None and sens >= 0:
                all_negative = False
                if verbose:
                    print(f"Warning: Room {room} has non-negative price sensitivity: {sens:.4f}")
        
        if all_negative and verbose:
            print("✓ All models have negative price sensitivity (expected)")
        
        return all_negative


def prepare_and_train_demand_models(
    hotel_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    room_types: List[str],
    adr_refs: Dict[str, float],
    seed: int = 42
) -> MultiRoomDemandModel:
    """
    Convenience function to prepare data and train all demand models.
    
    Args:
        hotel_df: Raw hotel booking data
        daily_df: Daily features DataFrame
        room_types: List of room types to model
        adr_refs: Reference ADR for each room type
        seed: Random seed
        
    Returns:
        Trained MultiRoomDemandModel
    """
    from data.load_data import get_feature_columns
    
    numerical_features, categorical_features = get_feature_columns(room_types)
    feature_cols = numerical_features + categorical_features
    
    # Prepare training data for each room type
    training_data = {}
    
    for room in room_types:
        samples = []
        labels = []
        
        # Get room-specific bookings
        room_df = hotel_df[hotel_df['reserved_room_type'] == room].copy()
        
        # Positive samples: actual bookings
        for _, row in room_df.iterrows():
            date = row['arrival_date']
            date_features = daily_df[daily_df['date'] == date]
            
            if len(date_features) == 0:
                continue
            
            features = date_features[feature_cols].values[0].tolist()
            price_ratio = row['adr'] / adr_refs[room]
            features.append(price_ratio)
            
            samples.append(features)
            labels.append(1)
        
        # Negative samples: synthetic high-price rejections
        for _, row in daily_df.iterrows():
            features_base = row[feature_cols].values.tolist()
            
            # Add negative samples at various high prices
            for price_ratio in [1.25, 1.35, 1.45]:
                features = features_base + [price_ratio]
                samples.append(features)
                labels.append(0)
        
        X = np.array(samples)
        y = np.array(labels)
        
        training_data[room] = (X, y)
        print(f"Room {room}: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    
    # Create and train multi-room model
    model = MultiRoomDemandModel(room_types, adr_refs, seed)
    model.fit_all(training_data, verbose=True)
    
    return model


if __name__ == '__main__':
    # Test demand models
    from data.load_data import load_and_prepare_all
    
    data = load_and_prepare_all()
    
    model = prepare_and_train_demand_models(
        hotel_df=data['hotel_df'],
        daily_df=data['train_df'],
        room_types=data['room_types'],
        adr_refs=data['adr_refs']
    )
    
    print("\n" + "=" * 50)
    print("Model Summary:")
    print(model.get_summary())
    
    print("\n" + "=" * 50)
    model.verify_price_sensitivity()
