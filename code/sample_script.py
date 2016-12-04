import numpy as np
import argparse
from anomaly_detector import AnomalyDetector

det = AnomalyDetector('RangePercentile', 'Plugin')
det.analyze(np.random.uniform(0, 1, 60))
det.analyze(np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)]))

line1 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
line2 = [10 * i - 270 + np.random.uniform(-1, 1) for i in range(30)]
det.analyze(np.hstack([line1, line2]))
