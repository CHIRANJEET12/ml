import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from models.linear_model.logistic_regression import LogisticRegression

def test_logistic_regression_on_noisy_data():
    """
    Tests Logistic Regression on a realistic, non-linearly separable dataset.
    Validates metrics on both training and unseen testing partitions.
    """
    print("\n--- Running Logistic Regression on Realistic Dataset ---")
    
    # 1. Generate a realistic binary dataset with noise and class overlap
    X, y = make_classification(
        n_samples=150,      # More data points
        n_features=4,       # More complex dimensional space
        n_informative=3,   # 3 features hold real patterns
        n_redundant=1,     # 1 feature is random noise correlated to others
        random_state=42,   # Ensures identical data every time you run it
        class_sep=1.1      # Controls overlap (lower means more overlap/harder)
    )

    # 2. Partition into separate training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Instantiate and train your custom model
    # We slightly lower the learning rate to ensure smooth descent over complex spaces
    model = LogisticRegression(learning_rate=0.05, epochs=1000)
    model.fit(X_train, y_train)

    print("\n--- Post-Training Evaluations ---")

    # 4. Verify parameter updates
    assert model.weights_ is not None, "Weights failed to initialize."
    assert model.weights_.shape == (4,), f"Expected shape (4,), got {model.weights_.shape}"
    print(f"Learned Weights vector : {np.round(model.weights_, 4)}")
    print(f"Learned Bias scalar    : {model.bias_:.4f}")
    print(f"Final Training Loss    : {model.loss:.4f}")

    # 5. Verify threshold-independent probabilities on unseen test set
    probs = model.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 2), "Probability matrix dimension mismatch."
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probability distributions must sum to 1.0 per row."
    
    # Print sample predictions to inspect confidence ranges
    print(f"Sample prob row (Class 0, Class 1) for index 0: {np.round(probs[0], 4)}")

    # 6. Evaluate accuracy metrics
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Final Train Accuracy   : {train_acc * 100:.2f}%")
    print(f"Final Test Accuracy    : {test_acc * 100:.2f}%")

    # 7. Core behavioral assertions
    # On a standard 4-feature classification set, the model should out-perform a random guess (50%)
    assert train_acc > 0.70, f"Training accuracy too low ({train_acc}). Check learning rate/gradients."
    assert test_acc > 0.70, f"Testing accuracy too low ({test_acc}). Model failing to generalize."
    
    print("✅ Realistic dataset evaluation test completed successfully!")

if __name__ == "__main__":
    test_logistic_regression_on_noisy_data()
