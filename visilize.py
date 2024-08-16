import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from utils import *
from nntli import *

# load dataset
file = "/W_best_naval.pkl"
path = os.getcwd()+file
with open(path, 'rb') as f:
    W1s, Wcs, Wds, a, b, tl1, tl2, Spatial = pickle.load(f)

file = "/network_best_naval.pkl"
path = os.getcwd()+file
with open(path, 'rb') as f:
    Formula, conjunc, disjunc, clip = pickle.load(f)

file = '/dataset/naval_dataset.pkl'
path = os.getcwd()+file
with open(path, 'rb') as f:
    train_data, train_label, val_data, val_label = pickle.load(f)

train_data = torch.tensor(train_data, requires_grad=False)
train_label = torch.tensor(train_label, requires_grad=False)
val_data = torch.tensor(val_data, requires_grad=False)
val_label = torch.tensor(val_label, requires_grad=False)

# get false data
false_data, false_label, acc = validation_accuracy(train_data,train_label,Formula,conjunc,disjunc,clip,W1s,Wcs,Wds)
false_data = np.array(false_data)
print('accuracy={acc}'.format(acc=acc))


# get data
print('get formulas ------------------')
f = []
d1 = torch.where(Wds==1)[0]
y_F_paras, y_G_paras  = [], []
x_F_paras, x_G_paras = [], []
const_F_paras, const_G_paras = [], []
time_F_intervals, time_G_intervals = [], []
time_interval_list = []
nplot_max = 0

for i in d1:
    f1 = []
    wc = Wcs[i,:]
    c1 = torch.where(wc==1)[0] # get formula when wc=1
    y_F_para, y_G_para  = [], []
    x_F_para, x_G_para = [], []
    const_F_para, const_G_para = [], []
    time_F_interval, time_G_interval = [], []
    for j in c1:
        # time parameters
        t1, t2= get_t1_t2(W1s[j,:])
        # predicates
        a1 = a[j,0,0] # 1d
        a2 = a[j,0,1]
        b1 = b[j,0]
        # get the function parameters
        if Spatial[j] == 'F':
            x_F_para.append(-a1.item())
            y_F_para.append(a2.item())
            const_F_para.append(b1.item())
            time_F_interval.append([t1, t2])
        if Spatial[j] == 'G':
            x_G_para.append(-a1.item())
            y_G_para.append(a2.item())
            const_G_para.append(b1.item())
            time_G_interval.append([t1,t2])
        # get the formula
        symbol = ">"
        if a2 >=0:
            add = "+"
        else:
            add = ""
        if b1 >=0:
            badd = b1
            addb = "-"
        else:
            badd = -b1
            addb = "+"
        f1.append(Spatial[j]+"["+str(t1)+","+str(t2)+"]"+"{:.2f}".format(a1)+"x"+add+"{:.2f}".format(a2)+"y"+addb+"{:.2f}".format(badd)+symbol+"0")
    f.append(f1)
    x_F_paras.append(x_F_para)
    y_F_paras.append(y_F_para)
    const_F_paras.append(const_F_para)
    time_F_intervals.append(time_F_interval)
    x_G_paras.append(x_G_para)
    y_G_paras.append(y_G_para)
    const_G_paras.append(const_G_para)
    time_G_intervals.append(time_G_interval)
    # get unique time intervals
    time_interval_list = time_F_interval + time_G_interval
    time_interval = np.array(time_interval_list)
    time_interval = np.unique(time_interval,axis=0)
    nplot = time_interval.shape[0]
    # maximum subplot
    if nplot>=nplot_max:
        nplot_max = nplot

# write down the formula
for fi in f:
    print(fi)

# plot data
fd = int(sum(Wds==1)) # disjunctions
fig, axes = plt.subplots(fd,nplot_max)
fig.set_size_inches(25, 10)
for i in range(fd):
    x_F_para = x_F_paras[i]
    y_F_para = y_F_paras[i]
    const_F_para = const_F_paras[i]
    time_F_interval = time_F_intervals[i]
    x_G_para = x_G_paras[i]
    y_G_para = y_G_paras[i]
    const_G_para = const_G_paras[i]
    time_G_interval = time_G_intervals[i]

    time_interval = time_F_interval + time_G_interval
    time_interval = np.array(time_interval)
    time_interval = np.unique(time_interval,axis=0)

    time_F_interval = np.array(time_F_interval)
    time_G_interval = np.array(time_G_interval)

    for id, ti in enumerate(time_interval):
        # position of subplot
        if fd==1 and nplot_max==1:
            ax = axes
        elif len(axes.shape)==1:
            ax = axes[id]
        else:
            ax = axes[i,id]
        # ax.set_xlim((0,25))
        # ax.set_ylim((0,50))
        nsample = 100
        plot_timed_data(file, ti, ax, nsample)
        # plot_false_data(false_data, false_label, ti, ax, nsample)
        for index, tf in enumerate(time_F_interval):
            if np.array_equal(tf,ti):
                a = x_F_para[index]
                b = y_F_para[index]
                c = const_F_para[index]
                plot_function(ax, a, b, c, 'orange')

        for index, tf in enumerate(time_G_interval):
            if np.array_equal(tf,ti):
                a = x_G_para[index]
                b = y_G_para[index]
                c = const_G_para[index]
                plot_function(ax, a, b, c, 'green')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_title(str(ti),fontsize=20)
figname = "classification.png"
plt.savefig(figname)
plt.show()