# %%
import sympy as sp
import numpy.matlib
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from numpy.linalg import inv
from scipy import interpolate
from scipy import constants
plt.style.use("ggplot")

#DATAFRAMES
df = pd.read_csv("5-Gev-mu-minus.csv")
me = pd.read_csv("mu-iron-energyloss.csv")
# f_size=len(df)
# f_size=3


#ARRAY FROM DATAFRAMES
df1=np.array(df)
# ke=df['tot_KE']*10**3  #MeV
mom=me['p']
moml=mom.tolist()
de_dx=me['dE/dx']
eid=df1[:,0]
en=max(eid)
# print("No. of events:",(en+1))



#CONSTANTS
c=constants.c
e=constants.e
Bx = 1.5                     #T
By = 0.0
Bz = 0.0
step_size = 96               #mm
# qp=3.6697713422062606e-23
# qp =e*c/(5*10**3)
# print("qpInitial",qp)
k_h  = 29979                 # MeV/c/T/mm
mass_mu = 105.6583755        # Mev/c^2
mass_e = 0.511               # MeV/c^2
rl_fe = 1.757                # cm    radiation length
A_fe = 55.845                # mass number
Z_fe = 26                    # atomic number
d_fe = 5.6                   # cm thickness of fe
rho_fe = 7.874               # g/cc
en_loss_fe= 1.594            # Mev c^2/g0
beta = 1
gamma = 10
f_size =0
    

#plt.scatter(df['posy'],df['momy'],df['momz'])

#LISTS
QP = []
# QP = [3.2140831407612124e-23]
# QP =[e/(mass_mu*c)]

QP_S=[]
P_S=[]

snx=[]          #strip numbers along x
sny=[]          #strip numbers along y

Ck_matrix=[]
F_matrix=[]
x_k1_fwd=[]
Qk_matrix=[]
F_r_a=[]

plot_pf=[]
plot_ps=[]


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
    return k_h*qP*np.sqrt(1+tx**2+ty**2)

def predict_x(x0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return x0 + tx*dz + hh*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ hh**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))

def predict_y(y0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return y0 + ty*dz + hh*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))+ hh**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))

def predict_tx(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return  tx + hh*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))+ hh**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))

def predict_ty(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return ty + hh*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ hh**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))

def propagator():

    tx, ty, qp, dz, x0, y0, del_x, del_y, del_tx, del_ty, del_qp, dl, T = sp.symbols('tx ty qp dz x0 y0 del_x del_y del_tx del_ty del_qp dl T')

    # x_ze = x0 + tx*dz + ((qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))  
    x_ze =x0 + tx*dz + (k_h*(qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz)) + ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((tx*(3*ty*2+1)*Sxx(dz)) - (ty*(3*tx**2+1)*Sxy(dz)) - (ty*(3*tx**2+1)*Syx(dz)) + (tx*(3*tx**2+3)*Syy(dz)))
    
    y_ze = y0 + ty*dz + (k_h*(qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz)) + ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((ty*(3*ty**2+3)*Sxx(dz)) - (tx*(3*ty**2+1)*Sxy(dz)) - (tx*(3*ty**2+1)*Syx(dz)) + (ty*(3*tx**2+1)*Syy(dz)))
    tx_ze = tx +(k_h*(qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz)) + ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((tx*(3*ty*2+1)*Rxx(dz)) - (ty*(3*tx**2+1)*Rxy(dz)) - (ty*(3*tx**2+1)*Ryx(dz)) + (tx*(3*tx**2+3)*Ryy(dz)))
    ty_ze = ty + (k_h*(qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz)) + (k_h*qp*((1+tx**2+ty**2)**0.5))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((ty*(3*ty*2+3)*Rxx(dz)) -(tx*(3*tx**2+1)*Rxy(dz)) - (tx*(3*ty**2+1)*Ryx(dz)) + (ty*(3*tx**2+1)*Ryy(dz)))
    dummy = 12
        
        
    ####list of constatnts of f(l) the function relating momentum and length travelled for 5GeV
    fl, l,fl1 ,fl2,c1, c2, c3= sp.symbols('fl l fl1 fl2 c1 c2 c3')
    c1=-3.644e-13
    c2=6.415e-10
    c3=4.437e-07
    c4=0.0001521
    c5=0.02675
    c6=2.227
        
    fl= -c1*l**5 + c2*l**4 - c3*l**3 + c4*l**2 - c5*l + c6
    fl1=sp.diff(fl,l)
    fl2=sp.diff(fl1,l)
    fl3=sp.diff(fl2,l)
    # print(fl1)
    #  print(fl2)
    #  print(fl3)
    d = {x0: sp.symbols("del_x"), y0: sp.symbols("del_y"), tx: sp.symbols("del_tx"), ty: sp.symbols("del_ty"), qp: sp.symbols("del_qp")}
    d_qp = (1+((fl2/fl1*dl)+(1/2*fl3/fl1)*dl**2))*d[qp] + ((fl1+fl2*dl)*fl*T*dl*(-Bx*d[x0]+Bz*d[y0])) + (fl1+fl2*dl)*step_size*(tx/T*d[tx]+ty/T*d[ty])
    # print(d_qp)
        
    deleqn=[]
        
    param = (x_ze, y_ze, tx_ze, ty_ze)
    for j in(param):
        PD=0
        for i in (x0,y0,tx,ty,qp):
            
            PD += sp.diff(j, i)*d[i]
            # print("PD",PD)
            # print("")
        deleqn.append(PD)
            
    deleqn.append(d_qp)
    # print("deleqn",deleqn)
    p=sp.diff(deleqn[0],del_tx)
    
        
    deno = (del_x, del_y, del_tx, del_ty, del_qp)
    
    ##################################################################################
    #part for extracting the equations and substituting the global variables in the eqn
        
        
    
    f00 = sp.diff(deleqn[0],deno[0]) ; print("f00\n",f00)    
    f01 = sp.diff(deleqn[0],deno[1]) ; print("f01\n",f01)
    f02 = sp.diff(deleqn[0],deno[2]) ; print("f02\n",f02)
    f03 = sp.diff(deleqn[0],deno[3]) ; print("f03\n",f03)
    f04 = sp.diff(deleqn[0],deno[4]) ; print("f04\n",f04)
        
    f10 = sp.diff(deleqn[1],deno[0]) ; print("f10\n",f10)
    f11 = sp.diff(deleqn[1],deno[1]) ; print("f11\n",f11)
    f12 = sp.diff(deleqn[1],deno[2]) ; print("f12\n",f12)
    f13 = sp.diff(deleqn[1],deno[3]) ; print("f13\n",f13)
    f14 = sp.diff(deleqn[1],deno[4]) ; print("f14\n",f14)
    print("\n")
        
    f20 = sp.diff(deleqn[2],deno[0]) ; print("f20\n",f20)
    f21 = sp.diff(deleqn[2],deno[1])  ; print("f21\n",f21)
    f22 = sp.diff(deleqn[2],deno[2])  ; print("f22\n",f22)
    f23 = sp.diff(deleqn[2],deno[3])  ; print("f23\n",f23)
    f24 = sp.diff(deleqn[2],deno[4])  ; print("f24\n",f24)
    print("\n")

    f30 = sp.diff(deleqn[3],deno[0])  ; print("f30\n",f30)
    f31 = sp.diff(deleqn[3],deno[1])  ; print("f31\n",f31)
    f32 = sp.diff(deleqn[3],deno[2])  ; print("f32\n",f32)
    f33 = sp.diff(deleqn[3],deno[3])  ; print("f33\n",f33)
    f34 = sp.diff(deleqn[3],deno[4])  ; print("f34\n",f34)
    print("\n")

    f40 = sp.diff(deleqn[4],deno[0])  ; print("f40\n",f40)
    f41 = sp.diff(deleqn[4],deno[1])  ; print("f41\n",f41)
    f42 = sp.diff(deleqn[4],deno[2])  ; print("f42\n",f42)
    f43 = sp.diff(deleqn[4],deno[3])  ; print("f43\n",f43)
    f44 = sp.diff(deleqn[4],deno[4])  ; print("f44\n",f44)
    print("\n")

        #    
        # pprint((F))

        
    return 0
        ###################################################################################
        
    
          
print(propagator())   
# %%

#         def gf00():
#             return 1
#         def gf01():
#             return 0
#         def gf02():
#             return 0.75*step_size**2*qp_p*tx0**2*ty0/np.sqrt(tx0**2 + ty0**2 + 1) + 0.75*step_size**2*qp_p*ty0*np.sqrt(tx0**2 + ty0**2 + 1) + step_size
#         def gf03():
#             return 0.75*step_size**2*qp_p*tx0*ty0**2/np.sqrt(tx0**2 + ty0**2 + 1) + 0.75*step_size**2*qp_p*tx0*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf04():  
#             return 0.75*step_size**2*tx0*ty0*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf10():
#             return 0
#         def gf11():
#             return 1
#         def gf12():
#             return 0.75*step_size**2*qp_p*tx0*(ty0**2 + 1)/np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf13():
#             return 0.75*step_size**2*qp_p*ty0*(ty0**2 + 1)/np.sqrt(tx0**2 + ty0**2 + 1) + 1.5*step_size**2*qp_p*ty0*np.sqrt(tx0**2 + ty0**2 + 1) + step_size
#         def gf14():
#             return 0.75*step_size**2*(ty0**2 + 1)*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf20():
#             return 0
#         def gf21():
#             return 0
#         def gf22():
#             return 1.5*step_size*qp_p*tx0**2*ty0/np.sqrt(tx0**2 + ty0**2 + 1) + 1.5*step_size*qp_p*ty0*np.sqrt(tx0**2 + ty0**2 + 1) + 1
#         def gf23():
#             return 1.5*step_size*qp_p*tx0*ty0**2/np.sqrt(tx0**2 + ty0**2 + 1) + 1.5*step_size*qp_p*tx0*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf24():
#             return 1.5*step_size*tx0*ty0*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf30():
#             return 0
#         def gf31():
#             return 0
#         def gf32():
#             return 1.5*step_size*qp_p*tx0*(ty0**2 + 1)/np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf33():
#             return 1.5*step_size*qp_p*ty0*(ty0**2 + 1)/np.sqrt(tx0**2 + ty0**2 + 1) + 3.0*step_size*qp_p*ty0*np.sqrt(tx0**2 + ty0**2 + 1) + 1
#         def gf34():
#             return 1.5*step_size*(ty0**2 + 1)*np.sqrt(tx0**2 + ty0**2 + 1)
#         def gf40():
#             return -144.011361674193*(9.816e-7*l**2 + 0.00066307463493294*l + 0.0212633973814044)*(3.272e-7*l**3 + 0.0002373*l**2 - 0.0243*l + 0.1715)
#         def gf41():
#             return 0
#         def gf42():
#             return -6.7068413465999e-7*l**2 - 0.00045304975320394*l - 0.0145283448173174   
#         def gf43():
#             return .00761252189256e-7*l**2 + 0.000338266182237115*l + 0.0108474791142113
#         def gf44():
#             return  96.0037871500304*(1.9632e-6*l + 0.0004746)/(9.816e-7*l**2 + 0.0004746*l - 0.0243) + 1 + 0.00904713936764081/(9.816e-7*l**2 + 0.0004746*l - 0.0243)             
#         # # print(gf40(),gf41(),gf42(),gf43(),gf44())
#         # # print(gf00(),gf01(),gf02(),gf03(),gf04(),gf10(),gf11(),gf12(),gf13(),gf14(),gf20(),gf21(),gf22(),gf23(),gf24(),gf30(),gf31(),gf32(),gf33(),gf34())

#         F=[[gf00(),gf01(),gf02(),gf03(),gf04()],[gf10(),gf11(),gf12(),gf13(),gf14()],[gf20(),gf21(),gf22(),gf23(),gf24()],[gf30(),gf31(),gf32(),gf33(),gf34()],[gf40(),gf41(),gf42(),gf43(),gf44()]]
#         # print("Deter F",np.linalg.det(F))
#         # print("matrix F",F)
#