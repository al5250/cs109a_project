import numpy as np
import argparse
from anomaly_detector import AnomalyDetector

norm = np.random.normal(0, 1, 60)
ex1 = np.hstack([range(30) + np.random.normal(0, 1, 30), 
                [-261 + 10 * i for i in range(30, 60)] + np.random.normal(0, 1, 30)])
ex2 = np.hstack([range(30) + np.random.normal(0, 1, 30), 
                [-261 + 10 * i for i in range(30, 60)] + np.random.normal(0, 1, 30), 
                [-851 + 20 * i for i in range(60, 90)] + np.random.normal(0, 1, 30)])
ex3 = np.hstack([range(30) + np.random.normal(0, 1, 30), 
                [-551 + 20 * i for i in range(30, 60)] + np.random.normal(0, 1, 30), 
                [39 + 10 * i for i in range(60, 90)] + np.random.normal(0, 1, 30)])
ex4 = np.hstack([np.random.normal(0, 1, 30), 10, np.random.normal(0, 1, 30)])

det2 = AnomalyDetector('OLSResidual', 'Power')
det2.analyze(ex1)
det2.analyze(ex2)
det2.analyze(ex3)
det2.analyze(ex4)

# det = AnomalyDetector('AverageDistance', 'Power')
# det.analyze(np.random.normal(0, 1, 60))
# det.analyze(np.hstack([np.random.normal(0, 1, 30), np.random.normal(10, 1, 30)]))
# det.analyze(np.hstack([np.random.normal(-10, 1, 20), np.random.normal(0, 1, 20), np.random.normal(10, 1, 20)]))

# det = AnomalyDetector('OLSTrend', 'Power')
# det.analyze(np.random.normal(0, 1, 50))
# det.analyze(np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 5), np.random.uniform(0, 1, 30)]))
# det.analyze(np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30), np.random.uniform(0, 1, 30)]))

# line1 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
# line2 = [10 * i - 270 + np.random.uniform(-1, 1) for i in range(30)]
# line3 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
# det.analyze(np.hstack([line1, line2, line3]))

# non_anomalous = np.random.uniform(0, 1, 60)
# anomalous = np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)])
#
# det = AnomalyDetector('AverageDistance', 'Power')
# det2 = AnomalyDetector('RangePercentile', 'Plugin')
#
# line1 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
# line2 = [10 * i - 270 + np.random.uniform(-1, 1) for i in range(30)]
# det2.analyze(np.hstack([line1, line2]))
