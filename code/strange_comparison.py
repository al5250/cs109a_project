data = {
    'Non-anomalous': gen_data.gen_data(),
    'Slope Change': gen_data.gen_data('Slope'),
    'Random Outliers': gen_data.gen_data('Peak'),
    'Sudden Gap': gen_data.gen_data('Plateau')
}

strangeness_functions = ['OLSTrend', 'OLSResidual', 'AverageDistance', 'RangePercentile']
betting_functions = ['Power']
# , 'Plugin', 'SimpleMixture'
detectors = itertools.product(strangeness_functions, betting_functions)
detectors = [AnomalyDetector(*det, mgale_params={'epsilon': 0.8}) for det in detectors]
n_det = len(detectors)

# for dat in data:
#     fig, ax = plt.subplots(n_det + 1, 1, figsize=(10, 5), sharex = True, sharey = True)
#     ax[0] = gen_data.plot_data(ax[0], *dat)
#     for i, det in enumerate(detectors):
#         ax[i+1] = det.analyze(ax[i+1], *dat)
# plt.show()

for name, dat in data.iteritems():
    fig, ax = plt.subplots(1, 1)
    ax = gen_data.plot_data(ax, *dat)
    for det in detectors:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('{} Data'.format(name), fontsize=14)
        _, ax = det.analyze(ax, *dat)
plt.show()
