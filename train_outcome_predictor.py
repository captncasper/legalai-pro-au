#!/usr/bin/env python3
"""Train ML model to predict case outcomes"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from load_real_aussie_corpus import corpus

class OutcomePredictor:
    def __init__(self):
        corpus.load_corpus()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = []
        
    def extract_features(self, case):
        """Extract ML features from case"""
        features = []
        
        # Text length features
        text_length = len(case['text'].split())
        features.append(text_length)
        
        # Year feature
        features.append(case['year'])
        
        # Court type features (one-hot encoding)
        courts = ['NSWDC', 'NSWLEC', 'FCA', 'HCA', 'Other']
        for court in courts:
            features.append(1 if court in case['court'] else 0)
        
        # Keyword features
        keywords = ['negligence', 'contract', 'breach', 'damages', 'appeal', 
                   'immigration', 'criminal', 'property', 'employment']
        text_lower = case['text'].lower()
        for keyword in keywords:
            features.append(1 if keyword in text_lower else 0)
        
        # Precedent count (if available)
        precedent_count = len([p for p in corpus.precedent_network 
                              if p['citing'] == case['citation']])
        features.append(precedent_count)
        
        return features
    
    def prepare_data(self):
        """Prepare training data"""
        X = []
        y = []
        
        for case in corpus.cases:
            features = self.extract_features(case)
            X.append(features)
            
            # Encode outcome
            outcome = case['outcome']
            if outcome == 'applicant_won':
                y.append(2)
            elif outcome == 'settled':
                y.append(1)
            else:  # applicant_lost
                y.append(0)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Train the model"""
        print("Preparing training data...")
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training on {len(X_train)} cases...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, 
              target_names=['Lost', 'Settled', 'Won']))
        
        # Save model
        with open('outcome_predictor.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("✅ Model saved to outcome_predictor.pkl")
        
        # Feature importance
        print("\n🎯 Top Feature Importance:")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        feature_names = ['text_length', 'year'] + \
                       [f'court_{c}' for c in ['NSWDC', 'NSWLEC', 'FCA', 'HCA', 'Other']] + \
                       [f'keyword_{k}' for k in ['negligence', 'contract', 'breach', 'damages', 
                        'appeal', 'immigration', 'criminal', 'property', 'employment']] + \
                       ['precedent_count']
        
        for i in indices:
            print(f"  - {feature_names[i]}: {importances[i]:.3f}")
    
    def predict_new_case(self, case_description):
        """Predict outcome for new case"""
        # Create fake case for feature extraction
        fake_case = {
            'text': case_description,
            'year': 2024,
            'court': 'Unknown',
            'citation': 'New Case'
        }
        
        features = self.extract_features(fake_case)
        probabilities = self.model.predict_proba([features])[0]
        
        return {
            'applicant_loses': float(probabilities[0]),
            'settles': float(probabilities[1]),
            'applicant_wins': float(probabilities[2]),
            'predicted_outcome': ['Lost', 'Settled', 'Won'][np.argmax(probabilities)]
        }

if __name__ == "__main__":
    predictor = OutcomePredictor()
    predictor.train()
    
    # Test prediction
    test_case = "negligence case involving personal injury and damages claim"
    result = predictor.predict_new_case(test_case)
    print(f"\n🔮 Prediction for: '{test_case}'")
    print(f"  Predicted outcome: {result['predicted_outcome']}")
    print(f"  Probabilities: Win={result['applicant_wins']:.1%}, "
          f"Settle={result['settles']:.1%}, Lose={result['applicant_loses']:.1%}")
