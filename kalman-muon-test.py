# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")

df = pd.read_csv("5-Gev-mu-minus.txt",sep="\t")
Bx = 1.5
By = 0.0
Bz = 0.0
k  = 0.29979  # GeV/c/T/m
#plt.scatter(df['posy'],df['momy'],df['momz'])


def measurements():
    #the coordinated are transformed to match with KB's Kalman filter
    y = df['posx']
    z = df['posy']
    x = df['posz']
    return x,y,z
def measurement(a):
    x,y,z = measurements()
    return x[a], y[a], z[a]
def print_slopes():
    x,y,z = measurements()
    xx =[]
    sx =[]
    sy =[]
    for i in range(meas_size()-1):
        slpx=(x[i+1]-x[i])/(z[i+1]-z[i])
        slpy=(y[i+1]-y[i])/(z[i+1]-z[i])
        print(i,slpx,slpy)
        xx.append(i)
        sx.append(slpx)
        sy.append(slpy)
    plt.plot(xx,sx)
    plt.plot(xx,sy)

def meas_size():
    return len(df['posx'])

def Sx(dz):
    return 0.5*Bx*dz**2

def Rx(dz):
    return Bx*dz

def Sy(dz):
    return 0.5*By*dz**2

def Ry(dz):
    return By*dz

def Sxx(dz):
    return (Bx**2*dz**3)/6

def Rxx(dz):
    return (Bx**2*dz**2)/2

def Sxy(dz):
    return (Bx*By*dz**3)/6

def Rxy(dz):
    return (Bx*By*dz**2)/2

def Syx(dz):
    return (Bx*By*dz**3)/6

def Ryx(dz):
    return (Bx*By*dz**2)/2

def Syy(dz):
    return (By**2*dz**3)/6

def Ryy(dz):
    return (By**2*dz**2)/2

def h(tx,ty,qP):
    return k*qP*np.sqrt(1+tx**2+ty**2)

def predict_x(x0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return x0 + tx*dz + hh*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ hh**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))

#print_slopes()
print(predict_x(2.0352,0.00846977093333,-0.014892050833,4.53,5.98))
plt.plot(df["posy"],np.sqrt(df["momx"]**2 + df["momy"]**2 + df["momz"]**2))



# %%