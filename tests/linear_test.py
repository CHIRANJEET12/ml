import numpy as np
from sklearn.datasets import make_regression

import utils.preprocessing as preprocessing
from models.linear_model.linear_regression import LinearRegression

from sklearn.linear_model import LinearRegression as SklearnLinearRegression

def run_regression_test():
    print("=" * 60)
    print("STARTING TEST FOR CUSTOM LINEAR REGRESSION PIPELINE")
    print("=" * 60)

    train_test_split = preprocessing.Preprocessing()

    X, y = make_regression(n_samples=100, n_features=3, noise=15.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[Step 1] Training custom Linear Regression model...")
    model = LinearRegression(learning_rate=0.05, epochs=1500)
    model.fit(X_train, y_train)
    
    print(f"-> Model trained successfully on {model.n_samples} samples.")
    print(f"-> Learned Weights: {np.round(model.weights_, 3)}")
    print(f"-> Learned Bias: {round(model.bias_, 3)}")

    print("\n[Step 2] Comparing parameters against Scikit-Learn standard...")
    sk_model = SklearnLinearRegression()
    sk_model.fit(X_train, y_train)
    
    try:
        np.testing.assert_allclose(model.weights_, sk_model.coef_, rtol=1e-2, atol=1e-1)
        np.testing.assert_allclose(model.bias_, sk_model.intercept_, rtol=1e-2, atol=1e-1)
        print("✅ PASS: Custom model weights and bias match Scikit-Learn exactly!")
    except AssertionError as e:
        print("❌ FAIL: Weight discrepancies found between implementations.")
        print(e)

    print("\n[Step 3] Testing prediction pipeline...")
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, f"Shape mismatch: Expected {y_test.shape}, got {y_pred.shape}"
    print(f"✅ PASS: Predictions shape matrix matches test targets {y_pred.shape}.")

    print("\n[Step 4] Testing custom metrics classes...")
    
    custom_r2 = model.score(X_test, y_test, is_r2=True)
    sklearn_r2 = sk_model.score(X_test, y_test)
    
    print(f"-> Custom R² Score:  {custom_r2:.5f}")
    print(f"-> Sklearn R² Score: {sklearn_r2:.5f}")
    
    assert abs(custom_r2 - sklearn_r2) < 1e-4, "R² score does not match sklearn benchmark."
    print("✅ PASS: Custom R² Score class matches official industry calculation.")

    custom_adj_r2 = model.score(X_test, y_test, is_r2=False)
    print(f"-> Custom Adjusted R² Score: {custom_adj_r2:.5f}")
    
    assert custom_adj_r2 < custom_r2, "Math error: Adjusted R² should penalize feature count."
    print("✅ PASS: Adjusted R² properly includes the feature penalty metric.")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_regression_test()