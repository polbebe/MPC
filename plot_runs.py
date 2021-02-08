import numpy as np
import pickle as pkl
from scipy import stats
import matplotlib.pyplot as plt

# results = np.load('new_MPC_results.npy')
# results = np.load('MPC_results.npy')
# for j in range(1,len(results)):
# data = []
# R = []
# data, R, train_L, val_L, scores = [], [], [], [], []
# results = pkl.load(open('big_MPC_results.pkl', 'rb'))
# losses = pkl.load(open('big_losses.pkl', 'rb'))
results = pkl.load(open('MPC_results.pkl', 'rb'))
losses = pkl.load(open('losses.pkl', 'rb'))

R_mean, R_std = [], []
scores_mean, scores_std = [], []
data = []
for key in results:
    X = [run[1] for run in results[key]]
    Y = [run[0] for run in results[key]]
    X, Y = np.array(X), np.array(Y)
    score = np.mean(np.sum(np.array(np.split(Y, 10, 1)), -1), -1)
    scores_mean.append(np.mean(score))
    scores_std.append(np.std(score))
    # Sum over episode
    # Y = np.sum(np.array(np.split(Y, 10)), 1)
    # X = np.sum(np.array(np.split(X, 10)), 1)
    r = np.array([stats.linregress(X[i], Y[i])[2]**2 for i in range(len(X))])
    R_mean.append(np.mean(r))
    R_std.append(np.std(r))
    data.append(key)

R_mean, R_std = np.array(R_mean), np.array(R_std)
plt.plot(data, R_mean)
plt.fill_between(data, R_mean-R_std, R_mean+R_std, alpha=0.5)
# plt.ylim(0, 1.1)
plt.xlabel('Datapoints')
plt.ylabel('R^2')
# plt.xscale('log')
# plt.grid()
plt.show()

scores_mean, scores_std = np.array(scores_mean), np.array(scores_std)
plt.plot(data, scores_mean)
plt.fill_between(data, scores_mean-scores_std, scores_mean+scores_std, alpha=0.5)
plt.xlabel('Datapoints')
plt.ylabel('Score')
# plt.xscale('log')
plt.show()

data = []
train_mean, train_std = [],[]
valid_mean, valid_std = [],[]
for key in losses:
    train = [run[1] for run in losses[key]]
    valid = [run[0] for run in losses[key]]
    train_mean.append(np.mean(train))
    train_std.append(np.std(train))
    valid_mean.append(np.mean(valid))
    valid_std.append(np.std(valid))
    data.append(key)
train_mean, train_std = np.array(train_mean), np.array(train_std)
valid_mean, valid_std = np.array(valid_mean), np.array(valid_std)
plt.plot(data, train_mean, label='Training')
plt.fill_between(data, train_mean+train_std, train_mean-train_std, alpha=0.5)
# plt.plot(data, valid_mean, label='Validation')
# plt.fill_between(data, valid_mean+valid_std, valid_mean-valid_std, alpha=0.5)
plt.xlabel('Datapoints')
plt.ylabel('Loss')
# plt.xscale('log')
plt.show()

# for key in results:
#     X = [run[1] for run in results[key]]
#     Y = [run[0] for run in results[key]]
#     X, Y = np.array(X[0]), np.array(Y[0])
#     # Sum over episode
#     # Y = np.sum(np.array(np.split(Y, 10)), 1)
#     # X = np.sum(np.array(np.split(X, 10)), 1)
#     plt.scatter(X, Y)
#     m, b, r, p, std_err = stats.linregress(X, Y)
#     print(str(key)+': '+str(r**2))
#     plt.title('Trained on '+str(key)+' datapoints')
#     plt.plot(X, m*X+b, label=str(key))
#     # plt.legend()
#     plt.xlabel('Pred Scores')
#     plt.ylabel('Real Scores')
#     plt.show()

# for i in range(len(results)):
# for i in range(1):
#     row = results[i,1:]
#     X = row[1::2]
#     Y = row[0::2]
#     X, Y = X[X != 0], Y[Y != 0]
#     plt.scatter(X, Y)
#     line = plt.plot(np.unique(X), np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)), label=str(int(results[i,0])))
# plt.legend()
# plt.xlabel('Pred Scores')
# plt.ylabel('Real Scores')
# plt.show()