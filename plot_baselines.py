from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np


def plot():
    results = pu.read_csv('results/baselines/ppo_50m.monitor.csv')

    plt.plot(np.cumsum(results.l), pu.smooth(results.r, radius=100))
    plt.show()


if __name__ == '__main__':
    plot()