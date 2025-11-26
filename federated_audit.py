import numpy as np


def detect_federated_anomalies(updates, threshold_percentile=95):  
    average_update = np.mean(updates, axis=0)
    anomaly_scores = [np.linalg.norm(update - average_update) for update in updates]

    threshold = np.percentile(anomaly_scores, threshold_percentile)
    flagged_updates = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    return flagged_updates, anomaly_scores


if __name__ == "__main__":
    fake_updates = np.random.randn(100, 10)  # Simulated 100 model updates
    flagged, scores = detect_federated_anomalies(fake_updates)
    print(f"Flagged Federated Updates (top 5% anomalies): {flagged}")
    # Optionally print some scores for inspection
    print(f"Sample anomaly scores: {scores[:5]}")