import fnmatch
import torch
import os
import pickle
import numpy as np
from random import shuffle
from nntli import *


def get_t1_t2(w):
    w = w.bool()
    l = w.shape[0]
    t = []
    t12 = []
    tf = False
    for j in range(l):
        if w[j] != tf:
            if tf == True:
                t12.append(j-1)
            else:
                t12.append(j)
            tf = not tf
            if len(t12) == 2:
                break
    if tf == True:
        t12.append(l-1)
    tc1 = t12[0]
    tc2 = t12[1]
    return tc1, tc2


def print_formula(Formula, Spatial, W1s, Wcs, Wds):
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach()
    formula_T = [] # temporal operator
    formula_time = [] # time interval
    formula_const = [] # x, y
    formula_xy = []
    formula_sym = [] # >, <
    for k in range(f_num):
        formula_T.append(Spatial[k])
        t11, t12 = get_t1_t2(W1s[k])
        formula_time.append([t11,t12])
        if Formula[k].A[0,0]==1:
            formula_xy.append('x')
            formula_sym.append('>')
            formula_const.append(Formula[k].b.item())
        elif Formula[k].A[0,0]==-1:
            formula_xy.append('x')
            formula_sym.append('<')
            formula_const.append(-Formula[k].b.item())
        elif Formula[k].A[0,1]==1:
            formula_xy.append('y')
            formula_sym.append('>')
            formula_const.append(Formula[k].b.item())
        elif Formula[k].A[0,1]==-1:
            formula_xy.append('y')
            formula_sym.append('<')
            formula_const.append(-Formula[k].b.item())
    dis_index = torch.where(Wds==1)[0]
    for indexd, i in enumerate(dis_index):
        if indexd > 0 and len(dis_index)>1:
            print(' or ')
        con_index = torch.where(Wcs[i]==1)[0]
        for indexc, j in enumerate(con_index):
            if indexc > 0 and len(con_index)>1:
                print(' and ')
            print(formula_T[j]+"["+str(formula_time[j][0])+","+str(formula_time[j][1])+"]"
            +formula_xy[j]+formula_sym[j]+'{:.2f}'.format(formula_const[j]))
                

def extract_formula(x,y,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds):
    _,_,acc_val = validation_accuracy(x,y,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds)
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach()
    for i in range(f_dis):
        if Wds[i]==0:
            continue
        Wds_new = torch.clone(Wds)
        Wds_new[i] = 0
        _,_,acc_new = validation_accuracy(x,y,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds_new)
        if round(acc_new.item(), 2) == round(acc_val.item(), 2):
            Wds[i] = 0
        else:
            for j in range(f_num):
                if Wcs[i,j]==0:
                    continue
                Wcs_new = torch.clone(Wcs)
                Wcs_new[i,j] = 0
                _,_,acc_new = validation_accuracy(x,y,Formula1,conjunc,disjunc,clip,W1s,Wcs_new,Wds)
                if round(acc_new.item(), 2) == round(acc_val.item(), 2):
                    Wcs[i,j] = 0
    return Wcs, Wds


def validation_accuracy(x,y,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds):
    nsample = x.shape[0]
    f_num = W1s.shape[0]
    f_conj = Wcs.shape[0]
    r1o = torch.empty((nsample,f_num,1), dtype=torch.float)
    for k, formula in enumerate(Formula1):
        xo1 = formula.robustness_trace(x,W1s[k,:],nsample,need_trace=False)
        r1o[:,k,:] = xo1[:,0]
    r2i = torch.squeeze(r1o,dim=2)
    r2o = torch.empty((nsample,f_conj), dtype=torch.float)
    for k in range(f_conj):
        xo2 = conjunc.forward(r2i,Wcs[k,:])
        r2o[:,k] = xo2[:,0]
    R = disjunc.forward(r2o,Wds)
    Rl = clip(R)
    acc = sum(y==Rl[:,0])/(nsample)
    false_data = x[y!=Rl[:,0],:,:]
    false_label = y[y!=Rl[:,0]]
    return false_data, false_label, acc


def plot_timed_data(file, time, ax, nsample):
    path = os.getcwd()+file
    with open(path, 'rb') as f:
        train_data, train_label, _, _ = pickle.load(f)
    n = train_data.shape[0]
    ind_list = [i for i in range(n)]
    shuffle(ind_list)
    ind = ind_list[0:nsample]
    x_train = train_data[ind,:,:]
    y_train = train_label[ind]
    if time[1]-time[0] == 0:
        m = '.'
    else:
        m = ''
    for i in range(len(y_train)):
        path = x_train[i,:,time[0]:time[1]+1]
        label = y_train[i]
        if label == 1:
            p1 = ax.plot(path[0,:],path[1,:], color='red',marker=m,label='1')
        else:
            p2 = ax.plot(path[0,:],path[1,:], color='blue',marker=m,label='-1')


def plot_timed_data_1d(file, time, ax, nsample):
    path = os.getcwd()+file
    with open(path, 'rb') as f:
        train_data, train_label, _, _ = pickle.load(f)
    t = np.arange(time[0],time[1]+1, 1)
    n = train_data.shape[0]
    ind_list = [i for i in range(n)]
    shuffle(ind_list)
    ind = ind_list[0:nsample]
    x_train = train_data[ind,:,:]
    y_train = train_label[ind]
    if time[1]-time[0] == 0:
        m = '.'
    else:
        m = ''
    for i in range(len(y_train)):
        path = x_train[i,0,time[0]:time[1]+1]
        label = y_train[i]
        if label == 1:
            p1 = ax.plot(t,path, color='red',marker=m,label='1')
        else:
            p2 = ax.plot(t,path, color='blue',marker=m,label='-1')


def plot_false_data(x, y, time, ax, nsample):
    if len(y)<nsample:
        nsample = len(y)
    if time[1]-time[0] == 0:
        m = '.'
    else:
        m = ''
    for i in range(nsample):
        path = x[i,:,time[0]:time[1]+1]
        label = y[i]
        if label == 1:
            p1 = ax.plot(path[0,:],path[1,:], color='gray',marker=m,label='false negative')
        else:
            p2 = ax.plot(path[0,:],path[1,:], color='black',marker=m,label='false positive')


def plot_false_data_1d(x, y, time, ax, nsample):
    if len(y)<nsample:
        nsample = len(y)
    t = np.arange(time[0],time[1]+1, 1)
    if time[1]-time[0] == 0:
        m = '.'
    else:
        m = ''
    for i in range(len(y)):
        path = x[i,0,time[0]:time[1]+1]
        label = y[i]
        if label == 1:
            p1 = ax.plot(t, path, color='gray',marker=m,label='false negative')
        else:
            p2 = ax.plot(t, path, color='black',marker=m,label='false positive')


def plot_function(ax, x_para, y_para, cons, fcolor):
    xl = ax.get_xlim()
    l1 = xl[0]
    l2 = xl[1]
    x = np.linspace(l1, l2)
    if fcolor=='green':
        op = 'Always'
    else:
        op = 'Eventually'
    if y_para == 0:
        if x_para<0:
            lg = '>'
        else:
            lg = '<'
        y = ax.get_ylim()
        p1 = ax.axvline(x=-cons/x_para, color=fcolor,label='{op}:x{lg}{ep}'.format(op=op,lg=lg,ep=np.round(-cons/x_para,2)))
    elif x_para == 0:
        if y_para<0:
            lg = '<'
        else:
            lg = '>'
        y = (x_para*x + cons)/y_para
        p2 = ax.plot(x, y, color=fcolor,label='{op}:y{lg}{ep}'.format(op=op,lg=lg,ep=np.round(cons/y_para,2)))
    else:
        if y_para<0:
            lg = '<'
        else:
            lg = '>'
        if cons/y_para<0:
            sym = ''
        else:
            sym = '+'
        y = (x_para*x + cons)/y_para
        p3 = ax.plot(x, y, color=fcolor,label='{op}:y{lg}{a}x{sym}{c}'.format(op=op,lg=lg,sym=sym,a=np.round(x_para/y_para,2),c=np.round(cons/y_para,2)))


def STL_accuracy(x,y,Formula,Spatial,W1s,Wcs,Wds,clip):
    nsample = x.shape[0]
    f_num = W1s.shape[0]
    f_conj = Wcs.shape[0]
    r1o = torch.empty((nsample,f_num,1), dtype=torch.float)
    for k, formula in enumerate(Formula):
        xo1 = formula.initialize_robustness(x)
        xo1 = xo1[:,:,formula.t1:formula.t2+1]
        xo1 = xo1[:,:,W1s[k,:]>=1]
        if Spatial[k]=='F':
            r1o[:,k,:] = torch.max(xo1,2)[0]
        elif Spatial[k]=='G':
            r1o[:,k,:] = torch.min(xo1,2)[0]
    r2i = torch.squeeze(r1o,dim=2)
    r2o = torch.empty((nsample,f_conj), dtype=torch.float)
    for k in range(f_conj):
        xo2 = r2i[:,Wcs[k,:]>=1]
        r2o[:,k] = torch.min(xo2,1)[0]
    ro = r2o[:,Wds>=1]
    R = torch.max(ro,1)[0]
    Rl = clip(R)
    acc = sum(y==Rl)/(nsample)
    return acc

    