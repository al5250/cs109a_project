import numpy as np
import argparse
from anomaly_detector import AnomalyDetector

non_anomalous = np.random.uniform(0, 1, 60)
anomalous = np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)])

det = AnomalyDetector('AverageDistance', 'Power')
det2 = AnomalyDetector('RangePercentile', 'Plugin')

line1 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
line2 = [10 * i - 270 + np.random.uniform(-1, 1) for i in range(30)]
det2.analyze(np.hstack([line1, line2]))