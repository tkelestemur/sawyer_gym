from baselines.common import plot_util as pu
results = pu.load_results('results/sawyer')

import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
plt.show()