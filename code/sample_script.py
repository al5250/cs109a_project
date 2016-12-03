import numpy as np
from anomaly_detector import AnomalyDetector

det = AnomalyDetector('AverageDistance', 'Power')
det.analyze(np.random.uniform(0, 1, 60))
det.analyze(np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)]))