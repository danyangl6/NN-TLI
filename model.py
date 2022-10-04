import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pickle
from time import time
from tlnn import *
from utils import *

torch.set_default_dtype(torch.float64)

def tlnn_train(train_data, train_label, val_data, val_label, dataname):
    nsample = train_data.shape[0]
    dim = train_data.shape[1]
    length = train_data.shape[-1]
    val_nsample = val_data.shape[0]

    # initialization
    STE= STEstimator.apply
    clip = Clip.apply

    f_num = 8
    f_conj = 1

    t1 = np.zeros((f_num,1))
    t1 = torch.tensor(t1, requires_grad=True)
    t2 = np.ones((f_num,1))*(length-1)
    t2 = torch.tensor(t2, requires_grad=True)

    Wc = torch.ones((f_conj,f_num), requires_grad=True)
    Wd = torch.ones(f_conj, requires_grad=False)

    a = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,0],[-1,0],[0,1],[0,-1]]).reshape(f_num,1,dim)
    a = torch.tensor(a, dtype=torch.float64, requires_grad=False)
    b = torch.rand((f_num,1), requires_grad=True)

    at = torch.tensor(1, requires_grad=False)
    W = RMinTimeWeight(at,t1,t2)
    W1 = torch.tensor(range(length), requires_grad=False)

    Formula = []
    Spatial = []
    tl1 = 0
    tl2 = length-1

    j = 0
    beta = 1
    am = 0
    scale = 2.5
    fn = int(f_num/2)
    for i in range(fn):
        Formula.append(Eventually(a[j],b[j],tl1,tl2))
        Formula[j].init_relumax(beta,am,scale,2)
        Spatial.append('F')
        j += 1
    for i in range(fn):
        Formula.append(Always(a[j],b[j],tl1,tl2))
        Formula[j].init_relumax(beta,am,scale,2)
        Spatial.append('G')
        j += 1

    beta = 1
    am = 0
    scale = 1.1
    conjunc = Conjunction()
    conjunc.init_relumax(beta,am,scale,1)
    beta = 1
    am = 0
    scale = 1
    disjunc = Disjunction()
    disjunc.init_relumax(beta,am,scale,1)

    optimizer1 = torch.optim.Adam([Wc], lr=0.1)
    optimizer2 = torch.optim.Adam([b], lr=0.1)
    optimizer3 = torch.optim.Adam([t1,t2], lr=0.1)
    n_iters = 10000
    batch_size = 10
    delta = 1
    acc = 0
    acc_best = 0
    log = []

    W1s = W.get_weight(W1)
    Wcs = STE(Wc)
    Wds = STE(Wd)
    x = val_data
    y = val_label
    r1o = torch.empty((val_nsample,f_num,1))
    for k, formula in enumerate(Formula):
        xo1 = formula.robustness_trace(x,W1s[k,:],val_nsample,need_trace=False)
        r1o[:,k,:] = xo1[:,0]
    r2i = torch.squeeze(r1o)
    r2o = torch.empty((val_nsample,f_conj))
    for k in range(f_conj):
        xo2 = conjunc.forward(r2i,Wcs[k,:])
        r2o[:,k] = xo2[:,0]
    R = disjunc.forward(r2o,Wds)
    Rl = clip(R)
    acc = sum(val_label==Rl[:,0])/(val_nsample)
    print('before training, accuracy = {acc}'.format(acc = acc))

    training_time = 0
    for epoch in range(1,n_iters):
        start = time()
        rand_num = rd.sample(range(0,nsample),batch_size)
        x = train_data[rand_num,:,:]
        y = train_label[rand_num]
        
        Wt = W.get_weight(W1)
        W1s = Wt
        W1s = STE(Wt)
        r1o = torch.empty((batch_size,f_num,1))
        for k, formula in enumerate(Formula):
            if sum(W1s[k,:])==0:
                with torch.no_grad():
                    Wc[:,k] = 0.4
            xo1 = formula.robustness_trace(x,W1s[k,:],batch_size,need_trace=False)
            r1o[:,k,:] = xo1[:,0]

        Wcs = STE(Wc)
        r2i = torch.squeeze(r1o)
        r2o = torch.empty((batch_size,f_conj))
        for k in range(f_conj):
            if sum(Wcs[k,:])==0:
                Wd[k] = 0
            else:
                Wd[k] = 1
            xo2 = conjunc.forward(r2i,Wcs[k,:])
            r2o[:,k] = xo2[:,0]

        Wd = torch.sum(Wcs,1)
        Wds = clip(Wd)
        R = disjunc.forward(r2o,Wds)
        Rl = clip(R)
        Rl = Rl[:,0]
        l = torch.sum(torch.exp(-delta * y * Rl)) #+ 0.1*torch.sum(Wcs)#- 0.001*torch.sum(t2-t1) #+ 0.1*torch.sum(Wcs)
        log.append(l.detach().numpy())
        l.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        
        with torch.no_grad():
            Wc[Wc<=0] = 0
            Wc[Wc>=1] = 1
        with torch.no_grad():
            t1[t1<0] = 0
            t2[t2<0] = 0
            t1[t1>tl2] = tl2
            t2[t2>tl2] = tl2
            for k, t in enumerate(t1):
                if t>t2[k]:
                    t1[k] = t2[k]-1
        
        end = time()
        training_time += end - start

        if epoch % 10 ==0:
            x = val_data
            y = val_label
            r1o = torch.empty((val_nsample,f_num,1))
            for k, formula in enumerate(Formula):
                xo1 = formula.robustness_trace(x,W1s[k,:],val_nsample,need_trace=False)
                r1o[:,k,:] = xo1[:,0]
            r2i = torch.squeeze(r1o)
            r2o = torch.empty((val_nsample,f_conj))
            for k in range(f_conj):
                xo2 = conjunc.forward(r2i,Wcs[k,:])
                r2o[:,k] = xo2[:,0]
            R = disjunc.forward(r2o,Wds)
            Rl = clip(R)
            acc_val = sum(val_label==Rl[:,0])/(val_nsample)
            acc_stl = STL_accuracy(x,y,Formula,Spatial,W1s,Wcs,Wds,clip)
            acc = acc_stl
            print('epoch_num = {epoch}, loss = {l}, accuracy_val = {acc_val}, accuracy_stl = {acc_stl}'.format(epoch=epoch,l=l, acc_val=acc_val, acc_stl=acc_stl))
            if acc>acc_best:
                best_training_time = training_time
            if acc>=acc_best:
                acc_best = acc
                Wcss, Wdss = extract_formula(train_data,train_label,Formula,conjunc,disjunc,clip,W1s,Wcs,Wds)
                f = open('W_best_'+dataname+'.pkl', 'wb')
                pickle.dump([W1s, Wcss, Wdss, a, b, t1, t2, Spatial], f)
                f.close()
                f = open('network_best_'+dataname+'.pkl', 'wb')
                pickle.dump([Formula, conjunc, disjunc, clip], f)
                f.close()
    print(acc_best)
    return acc_best, best_training_time

