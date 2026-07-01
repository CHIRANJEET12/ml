import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from models.non_linear.knn import KNN  # Matches your exact import path layout

def test_knn_on_large_dataset():
    """
    Validates the KNN engine on a large scale multi-class dataset.
    Tests loop infrastructure, multi-metric execution, and scoring capabilities.
    """
    print("\n--- Running Scaled KNN Functional Validation Suite ---")

    # 1. Generate a large, realistic synthetic dataset with 3 classes
    X, y = make_blobs(
        n_samples=75,       # Total points across features
        n_features=2,       # 2D coordinates for easy distance calculation
        centers=3,          # 3 separate distinct physical clusters
        cluster_std=0.60,   # Sizable distribution dispersion (creates overlapping points)
        random_state=42     # Identical deterministic values every iteration
    )

    # 2. Segment partitions out into separate Train and Test slices (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Instantiate model and feed constraints
    knn = KNN()
    knn.fit(X_train, y_train, k=5) # K=5 allows handling noise overlaps effectively

    print(f"Total training data size: {X_train.shape[0]} samples")
    print(f"Total validation test size: {X_test.shape[0]} samples")
    print(f"Identified Unique Clusters: {knn.unique_classes}")

    # 4. Cycle test vectors across your metrics suite array configurations
    metric_map = {1: "Euclidean", 2: "Manhattan"}

    for metric_val, metric_name in metric_map.items():
        predictions = []
        
        # Iterates sample-by-sample matching your internal loop counters structure
        for sample in X_test:
            sample_2d = sample.reshape(1, -1)
            pred = knn.predict(sample_2d, val=metric_val)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # Calculate raw accuracy performance calculation step manually
        accuracy = np.mean(predictions == y_test)
        print(f"\n[{metric_name} Distance Performance]")
        print(f"-> Predicted Array Output : {predictions}")
        print(f"-> Real Target Ground Truth: {y_test}")
        print(f"-> Evaluated Metric Score : {accuracy * 100:.2f}%")

        # Basic baseline sanity verification check
        assert accuracy > 0.80, f"KNN baseline performance dropped too low ({accuracy:.2f}) using {metric_name}."

    # 5. Check probability output array formats across the dataset array size
    print("\n[Probability Validation Checks]")
    for sample in X_test[:3]: # Sample the first three points
        sample_2d = sample.reshape(1, -1)
        probs = knn.predict_proba(sample_2d, val=1)
        
        # Mathematical rule checking checks
        assert isinstance(probs, list), "Expected list container output type framework."
        assert len(probs) == 3, f"Expected 3 target distribution slots, got: {len(probs)}"
        assert np.allclose(sum(probs), 1.0), f"Violated Probability distribution law! Array sum: {sum(probs)}"
        print(f"-> Sample Probs vector layout: {np.round(probs, 2)}")

    print("\n✅ All large-scale dataset KNN validations executed successfully!")

if __name__ == "__main__":
    test_knn_on_large_dataset()
