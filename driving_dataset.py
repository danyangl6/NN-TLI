from nturl2path import pathname2url
from re import L
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class GoForward(object):
    ''' Generate go forward data '''
    ''' initial position (x,y) '''
    ''' Go stright for l(length of signal) seconds with velocity v '''
    def __init__(self, nsample, length, gaussian=False, **kwargs):
        self.dataname = 'GoForward'
        self.nsample = nsample
        self.l = length
        self.gaussian = gaussian
        if gaussian == True:
            self.mean = kwargs['mean']
            self.std = kwargs['std']
    def generate_path(self, rand_pos=False, rand_v=False, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        v = kwargs['v']
        xl = x[0] if rand_pos==True else x
        xh = x[1] if rand_pos==True else x
        yl = y[0] if rand_pos==True else y
        yh = y[1] if rand_pos==True else y
        vl = v[0] if rand_v==True else v
        vh = v[1] if rand_v==True else v
        path = np.empty((self.nsample,2,self.l))
        for i in range(self.nsample):
            x = random.uniform(xl,xh)
            y = random.uniform(yl,yh)
            v = random.uniform(vl,vh)
            pi = np.array([[x]*self.l,[y+k*v for k in range(self.l)]])
            if self.gaussian:
                pi += np.random.normal(self.mean, self.std, size=(2,self.l))
            path[i,:,:] = pi
        return path


class StopandGo(object):
    ''' Generate stop and go data '''
    ''' initial position (x,y) '''
    ''' Go stright until position sp, then stop for st seconds, then go stright '''
    ''' The velocities for going straight are v1, v2 seperately '''
    ''' stop_pos is the position where the vehicle stops '''
    def __init__(self, nsample, length, gaussian=False, **kwargs):
        self.dataname = 'StopandGo'
        self.nsample = nsample
        self.l = length
        self.gaussian = gaussian
        if gaussian == True:
            self.mean = kwargs['mean']
            self.std = kwargs['std']
    def generate_path(self, rand_pos=False, rand_v=False, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        sp = kwargs['sp']
        st = kwargs['st']
        v1 = kwargs['v1']
        v2 = kwargs['v2']
        xl = x[0] if rand_pos==True else x
        xh = x[1] if rand_pos==True else x
        yl = y[0] if rand_pos==True else y
        yh = y[1] if rand_pos==True else y
        if sp<yh:
            raise ValueError('stop position error!')
        v1l = v1[0] if rand_v==True else v1
        v1h = v1[1] if rand_v==True else v1
        v2l = v2[0] if rand_v==True else v2
        v2h = v2[1] if rand_v==True else v2
        path = np.empty((self.nsample,2,self.l))
        stop_pos = np.empty((self.nsample,2,1))
        for i in range(self.nsample):
            x = random.uniform(xl,xh)
            y = random.uniform(yl,yh)
            v1 = random.uniform(v1l,v1h)
            v2 = random.uniform(v2l,v2h)
            t1 = int(sp//v1)
            t2 = int(self.l - st - t1)
            if t2<0:
                raise ValueError('t2 is negative!')
            p1 = np.array([[x]*(t1),[y+k*v1 for k in range(t1)]])
            p2 = np.array([[x]*st,[y+t1*v1]*st])
            p3 = np.array([[x]*t2,[y+t1*v1+k*v2 for k in range(t2)]])
            pi = np.concatenate((p1, p2, p3), axis=1)
            if self.gaussian:
                pi += np.random.normal(self.mean, self.std, size=(2,self.l))
            path[i,:,:] = pi
            stop_pos[i,:,:] = np.array([[x],[y+t1*v1]])
        return path, stop_pos


class LeftTurn(object):
    ''' Generate go left data '''
    ''' initial position (x,y) '''
    ''' Go stright until reach the lane, then enter the lane and continue going straight '''
    ''' Go stright for t1 seconds with velocity v1, turn left and then go for (l-t1) seconds with velocity v2 '''
    def __init__(self, nsample, length, gaussian=False, **kwargs):
        self.dataname = 'LeftTurn'
        self.nsample = nsample
        self.l = length
        self.gaussian = gaussian
        if gaussian == True:
            self.mean = kwargs['mean']
            self.std = kwargs['std']
    def generate_path(self, rand_pos=False, rand_v=False, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        v1 = kwargs['v1']
        v2 = kwargs['v2']
        lane = kwargs['lane']
        ll = lane[0]
        lh = lane[1]
        xl = x[0] if rand_pos==True else x
        xh = x[1] if rand_pos==True else x
        yl = y[0] if rand_pos==True else y
        yh = y[1] if rand_pos==True else y
        if ll<yh:
            raise ValueError('lane position error!')
        v1l = v1[0] if rand_v==True else v1
        v1h = v1[1] if rand_v==True else v1
        v2l = v2[0] if rand_v==True else v2
        v2h = v2[1] if rand_v==True else v2
        path = np.empty((self.nsample,2,self.l))
        for i in range(self.nsample):
            x = random.uniform(xl,xh)
            y = random.uniform(yl,yh)
            v1 = random.uniform(v1l,v1h)
            v2 = random.uniform(v2l,v2h)
            t1 = int(random.uniform(np.ceil((ll)/v1),np.ceil((lh)/v1)))
            t2 = int(self.l - t1)
            if t2<0:
                raise ValueError('t2 is negative!')
            p1 = np.array([[x]*t1,[y+k*v1 for k in range(t1)]])
            p2 = np.array([[x-k*v2 for k in range(t2)],[y+t1*v1]*t2])
            pi = np.concatenate((p1, p2), axis=1)
            if self.gaussian:
                pi += np.random.normal(self.mean, self.std, size=(2,self.l))
            path[i,:,:] = pi
        return path


class SwitchLane(object):
    ''' Generate Switch lane data '''
    ''' initial position (x,y) '''
    ''' Go stright for t1 seconds with velocity v1, change lane with x-axis velocity vx and y-axis velocity still v1 '''
    ''' until x-axis position x_positon = lane(can be consider as the center of the lane), then go straight with velocity v2 '''
    ''' variable vx and lane are fixed values '''
    def __init__(self, nsample, length, gaussian=False, **kwargs):
        self.dataname = 'SwitchLane'
        self.nsample = nsample
        self.l = length
        self.gaussian = gaussian
        if gaussian == True:
            self.mean = kwargs['mean']
            self.std = kwargs['std']
    def generate_path(self, rand_pos=False, rand_t=False, rand_v=False, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t1 = kwargs['t1']
        vx = kwargs['vx']
        lane = kwargs['lane']
        v1 = kwargs['v1']
        v2 = kwargs['v2']
        xl = x[0] if rand_pos==True else x
        xh = x[1] if rand_pos==True else x
        if xl<lane:
            raise ValueError('lane position error!')
        yl = y[0] if rand_pos==True else y
        yh = y[1] if rand_pos==True else y
        t1l = t1[0] if rand_t==True else t1
        t1h = t1[1] if rand_t==True else t1
        v1l = v1[0] if rand_v==True else v1
        v1h = v1[1] if rand_v==True else v1
        v2l = v2[0] if rand_v==True else v2
        v2h = v2[1] if rand_v==True else v2
        path = np.empty((self.nsample,2,self.l))
        for i in range(self.nsample):
            x = random.uniform(xl,xh)
            y = random.uniform(yl,yh)
            t1 = int(random.uniform(t1l,t1h))
            t2 = int((x-lane)//vx)
            t3 = int(self.l - t1 - t2)
            if t3<0:
                raise ValueError('t3 is negative!')
            v1 = random.uniform(v1l,v1h)
            v2 = random.uniform(v2l,v2h)
            p1 = np.array([[x]*t1,[y+k*v1 for k in range(t1)]])
            p2 = np.array([[x-k*vx for k in range(t2)],[y+v1*t1+k*v1 for k in range(t2)]])
            p3 = np.array([[x-t2*vx]*t3,[y+v1*(t1+t2)+k*v2 for k in range(t3)]])
            pi = np.concatenate((p1, p2, p3), axis=1)
            if self.gaussian:
                pi += np.random.normal(self.mean, self.std, size=(2,self.l))
            path[i,:,:] = pi
        return path


class Overtake(object):
    ''' Generate overtake data '''
    ''' initial position (x,y) '''
    ''' Go stright for t1 seconds with velocity v1, change lane with x-axis velocity vx and y-axis velocity still v1 '''
    ''' until x-axis position x_positon = lane(can be consider as the center of the lane), '''
    ''' then go straight with velocity v2(v2>v1 to overtake) for t2 seconds '''
    ''' change lane with x-axis velocity vx and y-axis velocity still v2, until back to the initial position x'''
    ''' finally go straight with velocity v2'''
    ''' variable vx and lane are fixed values '''
    def __init__(self, nsample, length, gaussian=False, **kwargs):
        self.dataname = 'Overtake'
        self.nsample = nsample
        self.l = length
        self.gaussian = gaussian
        if gaussian == True:
            self.mean = kwargs['mean']
            self.std = kwargs['std']
    def generate_path(self, rand_pos=False, rand_t=False, rand_v=False, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t1 = kwargs['t1']
        t2 = kwargs['t2']
        v1 = kwargs['v1']
        v2 = kwargs['v2']
        vx = kwargs['vx']
        lane = kwargs['lane']
        xl = x[0] if rand_pos==True else x
        xh = x[1] if rand_pos==True else x
        yl = y[0] if rand_pos==True else y
        yh = y[1] if rand_pos==True else y
        if xl<lane:
            raise ValueError('lane position error!')
        t1l = t1[0] if rand_t==True else t1
        t1h = t1[1] if rand_t==True else t1
        t2h = t2[1] if rand_t==True else t2
        t2l = t2[0] if rand_t==True else t2
        v1l = v1[0] if rand_v==True else v1
        v1h = v1[1] if rand_v==True else v1
        v2l = v2[0] if rand_v==True else v2
        v2h = v2[1] if rand_v==True else v2
        if v2l<v1h:
            raise ValueError('v2 need to be greater than v1!')
        path = np.empty((self.nsample,2,self.l))
        for i in range(self.nsample):
            x = random.uniform(xl,xh)
            y = random.uniform(yl,yh)
            t1 = int(random.uniform(t1l,t1h))
            t1x = int((x-lane)//vx)
            t2 = int(random.uniform(t2l,t2h))
            t2x = t1x
            t3 = self.l - t1 - t1x - t2 - t2x
            if t3<0:
                raise ValueError('t3 is negative!')
            v1 = random.uniform(v1l,v1h)
            v2 = random.uniform(v2l,v2h)
            p1 = np.array([[x]*t1,[y+k*v1 for k in range(t1)]])
            p2 = np.array([[x-k*vx for k in range(t1x)],[y+v1*t1+k*v1 for k in range(t1x)]])
            p3 = np.array([[x-t1x*vx]*t2,[y+v1*(t1+t1x)+k*v2 for k in range(t2)]])
            p4 = np.array([[x-t1x*vx+k*vx for k in range(t2x)],[y+v1*(t1+t1x)+v2*t2+k*v2 for k in range(t2x)]])
            p5 = np.array([[x-t1x*vx+t2x*vx]*t3,[y+v1*(t1+t1x)+v2*(t2+t2x)+k*v2 for k in range(t3)]])
            pi = np.concatenate((p1, p2, p3, p4, p5), axis=1)
            if self.gaussian:
                pi += np.random.normal(self.mean, self.std, size=(2,self.l))
            path[i,:,:] = pi
        return path


nplot = 100
n = 2500
l = 40
GoFor = GoForward(nsample=n, length=l, gaussian=True,mean=0,std=0)
StopGo = StopandGo(nsample=n, length=l, gaussian=True,mean=0,std=0)
LTurn = LeftTurn(nsample=n, length=l, gaussian=True,mean=0,std=0)
SLane = SwitchLane(nsample=n, length=l, gaussian=True,mean=0,std=0)
Otake = Overtake(nsample=n, length=l, gaussian=True,mean=0,std=0)
St = GoForward(nsample=n, length=l, gaussian=True,mean=0,std=0)

path1 = GoFor.generate_path(rand_pos=True,rand_v=True,x=[-2,2],y=[0,0],v=[1,1.2])
path2, stopP = StopGo.generate_path(rand_pos=True,rand_v=True,x=[-2,2],y=[0,0],sp=10,st=3,v1=[0.8,1],v2=[1,1.2])
dataname1 = GoFor.dataname
dataname2 = StopGo.dataname
for i in range(nplot):
    pi = path1[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:red',label='1')
for i in range(nplot):
    pi = path2[i,:,:]
    stopi = stopP[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:blue',label='-1')
    p2 = plt.plot(stopi[0,:],stopi[1,:], color='tab:blue',marker='.')
plt.title(dataname1+' vs '+dataname2, fontsize=16)
plt.show()
f = open(dataname1 + str('.pkl'), 'wb')
pickle.dump(path1, f)
f.close()
f = open(dataname2 + str('.pkl'), 'wb')
pickle.dump(path2, f)
f.close()

path1 = LTurn.generate_path(rand_pos=True,rand_v=True,x=[-2,2],y=[0,0],v1=[0.1,0.2],v2=[0.8,1],lane=[0,0.9])
path2 = LTurn.generate_path(rand_pos=True,rand_v=True,x=[-2,2],y=[0,0],v1=[0.1,0.2],v2=[0.8,1],lane=[1.1,2])
dataname1 = 'LeftTurn1'
dataname2 = 'LeftTurn2'
for i in range(nplot):
    pi = path1[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:red',label='1')
for i in range(nplot):
    pi = path2[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:blue',label='-1')
plt.title(dataname1+' vs '+dataname2, fontsize=16)
plt.show()
f = open(dataname1 + str('.pkl'), 'wb')
pickle.dump(path1, f)
f.close()
f = open(dataname2 + str('.pkl'), 'wb')
pickle.dump(path2, f)
f.close()

path1 = SLane.generate_path(rand_pos=True,rand_t=True,rand_v=True,x=[-2,2],y=[0,0],t1=[2,4],v1=[0.8,1],v2=[1.4,1.6],vx=0.5,lane=-4)
path2 = Otake.generate_path(rand_pos=True,rand_t=True,rand_v=True,x=[-2,2],y=[0,0],t1=[4,6],t2=[4,6],v1=[0.6,0.8],v2=[1.2,1.4],vx=0.5,lane=-4)
path3 = St.generate_path(rand_pos=True,rand_v=True,x=[-2,2],y=[0,0],v=[1,1.2])
dataname1 = SLane.dataname
dataname2 = Otake.dataname
dataname3 = St.dataname
for i in range(nplot):
    pi = path1[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:red',label='1')
for i in range(nplot):
    pi = path2[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:blue',label='-1')
for i in range(nplot):
    pi = path3[i,:,:]
    p1 = plt.plot(pi[0,:],pi[1,:], color='tab:green',label='0')
plt.title(dataname1+' vs '+dataname2+' vs '+dataname3, fontsize=16)
plt.show()
f = open(dataname1 + str('.pkl'), 'wb')
pickle.dump(path1, f)
f.close()
f = open(dataname2 + str('.pkl'), 'wb')
pickle.dump(path2, f)
f.close()
f = open(dataname3 + str('.pkl'), 'wb')
pickle.dump(path3, f)
f.close()