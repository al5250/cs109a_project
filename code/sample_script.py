import numpy as np
from anomaly_detector import AnomalyDetector

non_anomalous = np.random.uniform(0, 1, 60)
anomalous = np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)])

det = AnomalyDetector('AverageDistance', 'Power')
det.
