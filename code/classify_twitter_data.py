from __future__ import division
import numpy as np
from anomaly_detector import AnomalyDetector
import gen_data
import matplotlib
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import json
import time
from time import mktime
from datetime import datetime
import time

def to_date(str_date, float_flag=False):
    str_format = '%Y-%m-%d %H:%M:%S'
    if float_flag:
        str_format += '.%f'
    return datetime.strptime(str_date, str_format).timetuple()

company = 'AMZN'
start_index = 0
end_index = 2500

df = pd.read_csv('../data/realTweets/Twitter_volume_'+company+'.csv', delimiter = ',')
data = df['value'].values[start_index:end_index]

# get anomaly labels for data
json_f = open('../data/anomalies.json')
json_str = json_f.read()
labels_dict = json.loads(json_str)
labels = labels_dict['realTweets/Twitter_volume_'+company+'.csv']
labels = [(to_date(l, True), to_date(u, True)) for l, u in labels]

df['timestamp'] = df['timestamp'].apply(to_date)
df['label'] = 0
df['label'] = df['timestamp'].apply(lambda t: np.sum([l <= t <= u for l, u in labels]) > 0)
df['timestamp'] = df['timestamp'].apply(lambda t:
    matplotlib.dates.date2num(datetime.fromtimestamp(mktime(t))))

anomalies = [i for i in np.where(df['label'].values)[0] - start_index if i >= 0]
det = AnomalyDetector('OLSResidual', 'Power', mgale_params={'epsilon': 0.8})

fig, ax = plt.subplots(1, 1)
ax = gen_data.plot_data(ax, data, [anomalies[0]])

start = time.time()
fig, ax = plt.subplots(1, 1)
ax = det.analyze(ax, data, [anomalies[0]])

print "Process completed in {:.3} minutes".format((time.time() - start) / 60)
plt.show()
