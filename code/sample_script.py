import numpy as np
import argparse
from anomaly_detector import AnomalyDetector

non_anomalous = np.random.uniform(0, 1, 60)
anomalous = np.hstack([np.random.uniform(0, 1, 30), np.random.uniform(10, 11, 30)])

det = AnomalyDetector('RangePercentile', 'Power')
det.analyze(non_anomalous)
pvals = det.get_p_vals(anomalous)
print pvals
mgales = det.get_mgales(pvals)
print mgales
det.plot_mgales(mgales)

# det2 = AnomalyDetector('OLSTrend', 'Power', mgale_params={'epsilon': 0.8})
# line1 = [1 * i + np.random.uniform(-1, 1) for i in range(30)]
# line2 = [10 * i - 270 + np.random.uniform(-1, 1) for i in range(30)]
# det2.analyze(anomalous)