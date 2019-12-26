#%%
import numpy as np
import matplotlib.pyplot as plt


def noisy_sin(begin=0, end=10, spacing=0.1, scaling=1, noise_factor=1):
    x = np.arange(begin,end,spacing)
    y = np.sin(x)
    for i in np.arange(0,len(y)):
        y[i] = y[i] + noise_factor*np.random.normal()
    return x,y

def predict(curr_val, step_spacing=0.1):
    if(curr_val>1):
        curr_val=1
    if(curr_val<0):
        curr_val=0
    curr_step = np.arcsin(curr_val)
    next_val = np.sin(curr_step+step_spacing)
    return next_val

def estimate(predicted, measured,scale_factor):
    est = predicted + scale_factor*(measured-predicted)
    return est

begin=0
end=10
spacing=0.1

xx, yy = noisy_sin(begin=begin, end=end, spacing=spacing,noise_factor=0.7)

pp=[]
ee=[]

for i in np.arange(0,len(yy)):
    pred = predict(yy[i],spacing)
    pp.append(pred)
    est = estimate(pred,yy[i], 0.8)
    ee.append(est)

plt.scatter(xx,yy,label='noisy')
plt.scatter(xx,pp,label='pred')
plt.scatter(xx,ee,label='est')
plt.legend()

plt.show()


# %%
