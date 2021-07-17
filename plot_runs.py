import numpy as np
import pickle as pkl
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def smooth(Y, w=10):
    r = np.zeros_like(Y)
    for i in range(len(Y)):
        r[i] = np.mean(Y[max(0, i - w):i + 1])
    return r


def chart_loss(losses, agent, axes):
    data = []
    train_mean, train_std = [], []
    valid_mean, valid_std = [], []
    for key in losses:
        if len(losses[key]) == 0:
            continue
        train = [run[1] for run in losses[key]]
        valid = [run[0] for run in losses[key]]
        train_mean.append(np.mean(train))
        train_std.append(np.std(train))
        valid_mean.append(np.mean(valid))
        valid_std.append(np.std(valid))
        data.append(key)
    train_mean, train_std = np.array(train_mean), np.array(train_std)
    train_mean = smooth(train_mean)
    axes.plot(data, train_mean, label=agent)
    axes.fill_between(data, train_mean + train_std, train_mean - train_std, alpha=0.5)


def chart_r2(results, agent, axes):
    R_mean, R_std = [], []
    data = []
    for key in results:
        X = [run[1] for run in results[key]]
        Y = [run[0] for run in results[key]]
        X, Y = np.array(X), np.array(Y)
        r = np.array([stats.linregress(X[i], Y[i])[2] ** 2 for i in range(len(X))])
        R_mean.append(np.mean(r))
        R_std.append(np.std(r))
        data.append(key)

    R_mean, R_std = np.array(R_mean), np.array(R_std)
    R_mean = smooth(R_mean)
    axes.plot(data, R_mean, label=agent)
    axes.fill_between(data, R_mean - R_std, R_mean + R_std, alpha=0.5)


def chart_scores(results, agent, axes):
    scores_mean, scores_std = [], []
    data = []
    for key in results:
        X = [run[1] for run in results[key]]
        Y = [run[0] for run in results[key]]
        X, Y = np.array(X), np.array(Y)
        score = np.mean(np.sum(np.array(np.split(Y, 10, 1)), -1), -1)
        scores_mean.append(np.mean(score))
        scores_std.append(np.std(score))
        data.append(key)
    scores_mean, scores_std = np.array(scores_mean), np.array(scores_std)
    scores_mean = smooth(scores_mean)
    axes.plot(data, scores_mean, label=agent)
    axes.fill_between(data, scores_mean - scores_std, scores_mean + scores_std, alpha=0.5)


def chart_all():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

    score_plot = axes[0]
    r2_plot = axes[1]
    loss_plot = axes[2]

    chart_scores(pkl.load(open('MPC_results.pkl', 'rb')), 'MPC', score_plot)
    score_plot.set_xlabel('Datapoints')
    score_plot.set_ylabel('Score')
    score_plot.grid()
    score_plot.legend()

    chart_r2(pkl.load(open('MPC_results.pkl', 'rb')), 'MPC', r2_plot)
    r2_plot.set_xlabel('Datapoints')
    r2_plot.set_ylabel('r^2')
    r2_plot.legend()
    r2_plot.grid()

    chart_loss(pkl.load(open('losses.pkl', 'rb')), 'MPC', loss_plot)
    loss_plot.set_xlabel('Datapoints')
    loss_plot.set_ylabel('Loss')
    loss_plot.legend()
    loss_plot.grid()

    plt.show()


if __name__ == '__main__':
    # Command line usage 'python graph_logs.py --agents rl sup
    chart_all()
