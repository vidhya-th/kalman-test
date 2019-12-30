#%%
import numpy as np
import matplotlib.pyplot as plt


begin=0
end=10
spacing=0.1
gain_rate =0.0
gain_scale =0.2

belief = np.array([0.1])
def noisy_sin(begin=0, end=10, spacing=0.1, scaling=1, noise_factor=1):
    x = np.arange(begin,end,spacing)
    y = np.sin(x)
    y_noisy = []
    for i in np.arange(0,len(y)):
        y_noisy.append(y[i] + noise_factor*np.random.normal())
    return x,y,y_noisy

def predict(curr_val=0, step_spacing=0.1, g_rate=0):
    if(curr_val>1):
        curr_val=1
    if(curr_val<-1):
        curr_val=-1
    curr_step = np.arcsin(curr_val)
    # next_val = np.sin(curr_step+step_spacing)
    #gain_rate = np.cos(curr_step) 
    next_val = curr_val + g_rate*0.1
    return next_val

def estimate(predicted, measured, scale_factor, g_rate):
    est = predicted + scale_factor*(measured-predicted)
    gain_rate_updated = g_rate + gain_scale * (measured-predicted)/0.1
    return est,gain_rate_updated


xx,yy,yn = noisy_sin(begin=begin, end=end, spacing=spacing,noise_factor=0.7)

pp=[]
ee=[]
est =0
for i in np.arange(0,len(yn)):
    pred = predict(est,spacing,gain_rate)
    pp.append(pred)
    est, gain_rate = estimate(pred,yn[i], 0.3, gain_rate)
    ee.append(est)


plt.scatter(xx,yn,label='noisy')
plt.scatter(xx,yy,label='orig')
plt.scatter(xx,ee,label='est')
plt.legend()

plt.show()


# %%
