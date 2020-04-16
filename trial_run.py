# %%
import sympy as sp
import numpy.matlib
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import math
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
step_size = 56.2               #mm
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
FNF=[]
Ckrr=[]

plot_pf=[]
plot_ps=[]
filter_state_vector=np.empty([5,1], dtype='float')

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
    return k_h*qP*math.sqrt(1+tx**2+ty**2)

def predict_x(x0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return x0 + tx*dz + hh*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ hh**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))

def predict_y(y0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return y0 + ty*dz + hh*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))+ hh**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))


def predict_tx(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return  tx + hh*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))+ hh**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))

def predict_txdummy(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return  tx + hh*(tx*ty*Rx(dz))+ hh**2*(tx*(3*ty*2+1)*Rxx(dz) )



def predict_ty(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return ty + hh*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ hh**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))
def predict_tydummy(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return ty + hh*((1+ty**2)*Rx(dz))+ hh**2*(ty*(3*ty*2+3)*Rxx(dz))    
###################  Beginning of Event  ###########################3

for n in range(en+1):
    eid_ds = df[df['eid']==n]
    f_size=len(eid_ds)      # is the number of the layers per event
    eid_a=np.array(eid_ds)
    # print("eid",eid_a[0,0],"layers:",f_size)
    # print("@@@@@@@@@@@@@@@@@@@@\nEVENT ",n,"BEGINS @@@@@@@@@@@@@@@@@@@@@@@")
    # print("No.of layers",f_size)
      #####################   PREDICTION   ###########################
    xf=[]
    yf=[]
    txf=[]
    tyf=[]
    qpf=[]

    xf1=[]
    yf1=[]
    txf1=[]
    tyf1=[]
    qpf1=[]

    xp=[]
    yp=[]
    txp=[]
    typ=[]
    qpp=[]
    

    xm=[]
    ym=[]
    txm=[]
    tym=[]
    P_F=[5*10**3]  #MeV/c
    E_inc=[5*10**3]
    E_atplane=[]
    E_atplane_S=[]
    xpf=[]
    ypf=[]
    txf=[]
    tyf=[]
    # print("\n************************Filtering***************************\n")
    for i in range(f_size-1):
        # print("-----------------")
        # print("Layer ",i,"to",i+1)
        # print("-----------------")
          
        def predict_qp_f():
            x1=[]
            y1=[]
            index=0
            for j in mom:
              #print(j)
                if((P_F[-1]/j)>1):
                    
                    # print("appended P",P_F[-1])
                    x1=[mom[index], mom[index+1]]
                    y1=[de_dx[index], de_dx[index+1]]
                    # print("index",index)         
                index+=1
            # print(x1,y1)   
            fx = interpolate.interp1d(x1, y1,kind = 'linear')
            en_loss= fx(P_F[-1])
            # print("en-loss",en_loss)
            en_incident = E_inc[-1]
            # print("EnIncident",en_incident)
            en_atplane = en_incident-(en_loss*step_size/10*rho_fe)
            # print("en_atplane",en_atplane)
            p=math.sqrt((en_atplane)**2-(mass_mu)**2)
            # print("p",p)
            # print("e/p",e/p)
            E_inc.append(en_atplane)           #for filtering
            E_atplane.append(en_atplane)       #for smoothing
            P_F.append(p)
            QP.append(e/p)
            return e/p
                    
         # plt.loglog(t,de_dx)
        if(i==0):
            x0=eid_a[i,6]
            y0=eid_a[i,4]
            tx0=((eid_a[i+1,6]-eid_a[i,6])/eid_a[i+1,5]-eid_a[i,5])
            ty0=((eid_a[i+1,4]-eid_a[i,4])/eid_a[i+1,5]-eid_a[i,5])
            qp0=0
        else:
            #filter
            # x0=xpf
            # y0=ypf
            # tx0=txf
            # ty0=tyf 
            # qp0=qpf  
            #predicted 
            x0=xpf[-1]  
            y0=ypf[-1]
            tx0=txf[-1]
            ty0=tyf[-1]
            qp0=qpf[-1] 
        T=math.sqrt(tx0**2+ty0**2+1)
        l=math.sqrt((eid_a[i+1,4]-eid_a[i,4])**2+(eid_a[i+1,5]-eid_a[i,5])**2+(eid_a[i+1,6]-eid_a[i,6])**2)
        #     # # print("l",l)
        dl=(step_size)/(1/T)
        # # print("T",T)
        # # print("1/T",1/T)
        # # print("dl",dl)
        qp_p=predict_qp_f()
        x_p=predict_x(x0,tx0,ty0,qp_p,step_size)
        y_p=predict_y(y0,tx0,ty0,qp_p,step_size)
        tx_p=predict_tx(tx0,ty0,qp_p,step_size)
        ty_p=predict_ty(tx0,ty0,qp_p,step_size)
        
        
        xpf.append(x_p)
        ypf.append(y_p)
        txf.append(tx_p)
        tyf.append(ty_p)
        qpf.append(qp_p)
        


        # print("tx",tx_p)    
        # print(tx_p," + ",h(tx_p,ty_p,qp_p)*(tx_p*ty_p*Rx(step_size)),"+", h(tx_p,ty_p,qp_p)**2*(tx_p*(3*ty_p*2+1)*Rxx(step_size) ))
        # print(tx_p + h(tx_p,ty_p,qp_p)*(tx_p*ty_p*Rx(step_size)) + h(tx_p,ty_p,qp_p)**2*(tx_p*(3*ty_p*2+1)*Rxx(step_size) ))
        txm.append((eid_a[i+1,6]-eid_a[i,6])/(eid_a[i+1,5]-eid_a[i,5]))
        # print(txm)
        tym.append((eid_a[i+1,4]-eid_a[i,4])/(eid_a[i+1,5]-eid_a[i,5]))
       
      # TERMS IN NOISE MATRIX
        
         # terms due to multiple scattering 
        D = -1      # forward direction
        lQ = step_size*T*D
        ls = rl_fe*((Z_fe+1)/Z_fe)*(289*np.power(Z_fe,1/3))/(159*np.power(Z_fe,1/2))
        CMS = ((0.015)**2/((beta)**2*((P_F[i])**2)))*lQ/ls

       # # terms due to Energy loss straggling

        Tmax = 2*mass_e*(beta)**2*(gamma)**2/(1+(2*gamma*mass_e/mass_mu)+(mass_e/mass_mu)**2)
        si = 0.1534*e**2*Z_fe/(beta**2*A_fe)*rho_fe*d_fe
        sigma_sq_E = si*Tmax*(1-(beta**2/2))

     # x00, y00, x, y, tx, ty, tx1, ty1, q, P, dz = sp.symbols('x00 y00 x y tx ty tx1 ty1 q P dz ')
        
        # x = x00 + tx*dz + (k_h*q/P*((1+tx**2+ty**2)**0.5)) *(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))
         # y=y00 + ty*dz + (k_h*q/P*((1+tx**2+ty**2)**0.5))*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))
         # tx1 = tx + (k_h*q/P*((1+tx**2+ty**2)**0.5))*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))
         # ty1=ty + (k_h*q/P*((1+tx**2+ty**2)**0.5))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))
    
        dx_dp = -0.2248425*(step_size)**2*e*tx0*ty0*(math.sqrt(tx0**2 + ty0**2 + 1))/P_F[i]**2 - 0.067405533075*(step_size)**3*e**2*tx0*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1/P_F[i]**3
        dy_dp = -0.2248425*step_size**2*e*(ty0**2 + 1)*(math.sqrt(tx0**2 + ty0**2 + 1))/P_F[i]**2 - 0.067405533075*step_size**3*e**2*ty0*(3*ty0**2 + 3)*(tx0**2 + ty0**2 + 1)**1/P_F[i]**3
        dtx1_dp =  -0.449685*step_size*e*tx0*ty0*(math.sqrt(tx0**2 + ty0**2 + 1))/P_F[i]**2 - 0.202216599225*step_size**2*e**2*tx0*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1/P_F[i]**3
        dty1_dp =  -0.449685*step_size*e*tx0*ty0*(math.sqrt(tx0**2 + ty0**2 + 1))/P_F[i]**2 - 0.202216599225*step_size**2*e**2*tx0*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1/P_F[i]**3

        
        c_xqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dx_dp)*sigma_sq_E
        c_yqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dy_dp)*sigma_sq_E
        c_txqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dtx1_dp)*sigma_sq_E
        c_tyqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dty1_dp)*sigma_sq_E
        c_qpqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*sigma_sq_E


        def Q00():
            return (1+tx0**2)*T**2*CMS*lQ**3/3
        def Q01():
            return  (tx0*ty0)*T**2*CMS*lQ**3/3
        def Q02():
            return  (1+tx0**2)*T**2*CMS*lQ**2/2*D
        def Q03():
            return  (tx0*ty0)*T**2*CMS*lQ**2/2*D
        def Q04():
            return  c_xqp
        def Q10():
            return  (tx0*ty0)*T**2*CMS*lQ**3/3
        def Q11():
            return  (1+ty0**2)*T**2*CMS*lQ**3/3  
        def Q12():
            return  (tx0*ty0)*T**2*CMS*lQ**2/2*D
        def Q13():
            return  (1+ty0**2)*T**2*CMS*lQ**2/2*D
        def Q14():
            return  c_yqp

        def Q20():
            return  (1+tx0**2)*T**2*CMS*lQ**2/2*D
        def Q21():
            return  (tx0*ty0)*T**2*CMS*lQ**2/2*D
        def Q22():
            return   (1+tx0**2)*T**2*CMS*lQ
        def Q23():
            return   (tx0*ty0)*T**2*CMS*lQ
        def Q24():
            return    c_txqp

        def Q30():
            return   (tx0*ty0)*T**2*CMS*lQ**2/2*D
        def Q31():
            return   (1+ty0**2)*T**2*CMS*lQ**2/2*D
        def Q32():
            return   (tx0*ty0)*T**2*CMS*lQ
        def Q33():
            return   (tx0*ty0)*T**2*CMS*lQ
        def Q34():
            return    c_tyqp  

        def Q40():
            return  c_xqp 
        def Q41():
            return  c_yqp
        def Q42():
            return  c_txqp
        def Q43():
            return  c_tyqp 
        def Q44():
            return  c_qpqp        


        Qk=[[Q00(),Q01(),Q02(),Q03(),Q04()],
        [Q10(),Q11(),Q12(),Q13(),Q14()],
        [Q20(),Q21(),Q22(),Q23(),Q24()],
        [Q30(),Q31(),Q32(),Q33(),Q34()],
        [Q40(),Q41(),Q42(),Q43(),Q44()]]
        Qk_matrix.append(Qk)
        # print("QK",Qk_matrix)   
                
     #Strip digitization   
        a=df1[i,6]*10**(-3)*(-256)/8
        snx.append(int(a))
        b=df1[i,4]*10**(-3)*256/8
        sny.append(int(b))

     #    ######################    FILTERING    ########################



      # PROPAGATOR MATRIX
 
        def propagator():


            tx, ty, qp, dz, x0, y0, del_x, del_y, del_tx, del_ty, del_qp = sp.symbols('tx ty qp dz x0 y0 del_x del_y del_tx del_ty del_qp')

            x_ze = x0 + tx*dz + ((qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz)) + ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((tx*(3*ty*2+1)*Sxx(dz)) - (ty*(3*tx**2+1)*Sxy(dz)) - (ty*(3*tx**2+1)*Syx(dz)) + (tx*(3*tx**2+3)*Syy(dz)))
            y_ze = y0 + ty*dz + ((qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))
            tx_ze = tx +((qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))
            ty_ze = ty + ((qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))
            # qp_ze = 0
                
                
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
            d_qp = (1+((fl2/fl1*dl)+(1/2*fl3/fl1)*dl**2))*d[qp] + ((fl1+fl2*dl)*fl*T*dl*(-Bx*d[x0]+Bz*d[y0])) + (fl1+fl2*dl)*step_size*(tx0/T*d[tx]+ty0/T*d[ty])
            # print(d_qp)
                
            deleqn=[]
                
            param = (x_ze, y_ze, tx_ze, ty_ze)
                
            for j in(param):
                PD=0
                for i in (x0,y0,tx,ty,qp):
                    PD += sp.diff(j, i)*d[i]
                    # F.fill(sp.diff(PD, k))
                    deleqn.append(PD)
                    # print(PD)
                    # print("")
                # print(deleqn[2])
            deleqn.append(d_qp)
             # print(deleqn[4])
                
            # deno = (del_x, del_y, del_tx, del_ty, del_qp)
            
            ##################################################################################
            #part for extracting the equations and substituting the global variables in the eqn
                
                
        
        # f00 = sp.diff(deleqn[0],deno[0]) ; print("f00\n",f00)    
        # f01 = sp.diff(deleqn[0],deno[1]) ; print("f01\n",f01)
        # f02 = sp.diff(deleqn[0],deno[2]) ; print("f02\n",f02)
        # f03 = sp.diff(deleqn[0],deno[3]) ; print("f03\n",f03)
        # f04 = sp.diff(deleqn[0],deno[4]) ; print("f04\n",f04)
                
        # f10 = sp.diff(deleqn[1],deno[0]) ; print("f10\n",f10)
        # f11 = sp.diff(deleqn[1],deno[1]) ; print("f11\n",f11)
        # f12 = sp.diff(deleqn[1],deno[2]) ; print("f12\n",f12)
        # f13 = sp.diff(deleqn[1],deno[3]) ; print("f13\n",f13)
        # f14 = sp.diff(deleqn[1],deno[4]) ; print("f14\n",f14)
        
        # f20 = sp.diff(deleqn[2],deno[0]) ; print("f20\n",f20)
        # f21 = sp.diff(deleqn[2],deno[1])  ; print("f21\n",f21)
        # f22 = sp.diff(deleqn[2],deno[2])  ; print("f22\n",f22)
        # f23 = sp.diff(deleqn[2],deno[3])  ; print("f23\n",f23)
        # f24 = sp.diff(deleqn[2],deno[4])  ; print("f24\n",f24)
            
        # f30 = sp.diff(deleqn[3],deno[0])  ; print("f30\n",f30)
        # f31 = sp.diff(deleqn[3],deno[1])  ; print("f31\n",f31)
        # f32 = sp.diff(deleqn[3],deno[2])  ; print("f32\n",f32)
        # f33 = sp.diff(deleqn[3],deno[3])  ; print("f33\n",f33)
        # f34 = sp.diff(deleqn[3],deno[4])  ; print("f34\n",f34)

        # f40 = sp.diff(deleqn[4],deno[0])  ; print("f40\n",f40)
        # f41 = sp.diff(deleqn[4],deno[1])  ; print("f41\n",f41)
        # f42 = sp.diff(deleqn[4],deno[2])  ; print("f42\n",f42)
        # f43 = sp.diff(deleqn[4],deno[3])  ; print("f43\n",f43)
        # f44 = sp.diff(deleqn[4],deno[4])  ; print("f44\n",f44)


                #    
                # pprint((F))

                

                ###################################################################################
                
            return  0
          
        # print(propagator())   

        def gf00():
            return 1
        def gf01():
            return 0
        def gf02():
            return 674055330.75*step_size**3*qp_p**2*tx0**2*(6*ty0 + 1) + 337027665.375*step_size**3*qp_p**2*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1 + 22484.25*step_size**2*qp_p*tx0**2*ty0/(math.sqrt(tx0**2 + ty0**2 + 1))+ 0.75*step_size**2*qp_p*ty0*(math.sqrt(tx0**2 + ty0**2 + 1)) + step_size
        def gf03():
            return 674055330.75*step_size**3*qp_p**2*tx0*ty0*(6*ty0 + 1) + 2022165992.25*step_size**3*qp_p**2*tx0*(tx0**2 + ty0**2 + 1)**1 + 22484.25*step_size**2*qp_p*tx0*ty0**2/(math.sqrt(tx0**2 + ty0**2 + 1)) + 0.75*step_size**2*qp_p*tx0*(math.sqrt(tx0**2 + ty0**2 + 1))
        def gf04():  
            return 674055330.75*step_size**3*qp_p*tx0*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1 + 22484.25*step_size**2*tx0*ty0*(math.sqrt(tx0**2 + ty0**2 + 1))
        
        def gf10():
            return 0
        def gf11():
            return 1
        def gf12():
            return 674055330.75*step_size**3*qp_p**2*tx0*ty0*(3*ty0**2 + 3) + 22484.25*step_size**2*qp_p*tx0*(ty0**2 + 1)/(math.sqrt(tx0**2 + ty0**2 + 1))
        def gf13():
            return 674055330.75*step_size**3*qp_p**2*ty0**2*(3*ty0**2 + 3) + 2022165992.25*step_size**3*qp_p**2*ty0**2*(tx0**2 + ty0**2 + 1)**1 + 337027665.375*step_size**3*qp_p**2*(3*ty0**2 + 3)*(tx0**2 + ty0**2 + 1)**1 + 22484.25*step_size**2*qp_p*ty0*(ty0**2 + 1)/(math.sqrt(tx0**2 + ty0**2 + 1) )+ 44968.5*step_size**2*qp_p*ty0*(math.sqrt(tx0**2 + ty0**2 + 1)) + step_size
        def gf14():
            return 674055330.75*step_size**3*qp_p*ty0*(3*ty0**2 + 3)*(tx0**2 + ty0**2 + 1)**1 + 22484.25*step_size**2*(ty0**2 + 1)*(math.sqrt(tx0**2 + ty0**2 + 1))
        
        def gf20():
            return 0
        def gf21():
            return 0
        def gf22():
            return 2022165992.25*step_size**2*qp_p**2*tx0**2*(6*ty0 + 1) + 1011082996.125*step_size**2*qp_p**2*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1 + 44968.5*step_size*qp_p*tx0**2*ty0/math.sqrt(tx0**2 + ty0**2 + 1) + 44968.5*step_size*qp_p*ty0*math.sqrt(tx0**2 + ty0**2 + 1) + 1
        def gf23():
            return 2022165992.25*step_size**2*qp_p**2*tx0*ty0*(6*ty0 + 1) + 6066497976.75*step_size**2*qp_p**2*tx0*(tx0**2 + ty0**2 + 1)**1+ 44968.5*step_size*qp_p*tx0*ty0**2/math.sqrt(tx0**2 + ty0**2 + 1) +44968.5*step_size*qp_p*tx0*math.sqrt(tx0**2 + ty0**2 + 1)
        def gf24():
            return 2022165992.25*step_size**2*qp_p*tx0*(6*ty0 + 1)*(tx0**2 + ty0**2 + 1)**1 + 44968.5*step_size*tx0*ty0*math.sqrt(tx0**2 + ty0**2 + 1)
        
        def gf30():
            return 0
        def gf31():
            return 0
        def gf32():
            return 2022165992.25*step_size**2*qp_p**2*tx0*ty0*(6*ty0 + 3) + 44968.5*step_size*qp_p*tx0*(ty0**2 + 1)*(1/math.sqrt(tx0**2 + ty0**2 + 1)) + 1.5*step_size*qp_p*tx0*(ty0**2 + 1)/math.sqrt(tx0**2 + ty0**2 + 1)
        def gf33():
            return 2022165992.25*step_size**2*qp_p**2*ty0**2*(6*ty0 + 3) + 6066497976.75*step_size**2*qp_p**2*ty0*(tx0**2 + ty0**2 + 1)**1 + 1011082996.125*step_size**2*qp_p**2*(6*ty0 + 3)*(tx0**2 + ty0**2 + 1)**1 + 44968.5*step_size*qp_p*ty0*(ty0**2 + 1)*(1/math.sqrt(tx0**2 + ty0**2 + 1)) + 1.5*step_size*qp_p*ty0*(ty0**2 + 1)/math.sqrt(tx0**2 + ty0**2 + 1) + 3.0*step_size*qp_p*ty0*math.sqrt(tx0**2 + ty0**2 + 1) + 89937.0*step_size*qp_p*ty0*(math.sqrt(tx0**2 + ty0**2 + 1)) + 1
        def gf34():
            return 2022165992.25*step_size**2*qp_p*ty0*(6*ty0 + 3)*(tx0**2 + ty0**2 + 1)**1 + 1.5*step_size*(ty0**2 + 1)*math.sqrt(tx0**2 + ty0**2 + 1) + 44968.5*step_size*(ty0**2 + 1)*(math.sqrt(tx0**2 + ty0**2 + 1))
        
        def gf40():
            return 144.011361674193*(1.822e-12*l**4 + 3.26567560074942e-9*l**3 - 5.92062846519066e-7*l**2 + 4.86187178491891e-5*l + 0.00245435205103926)*(3.644e-13*l**5 + 6.415e-10*l**4 - 4.437e-7*l**3 + 0.0001521*l**2 - 0.02675*l + 2.227)
        def gf41():
            return 0
        def gf42():
            return -1.24489251563824e-12*l**4 - 2.23129259817529e-9*l**3 + 4.04530519439659e-7*l**2 - 3.321903291459e-5*l - 0.00167695087765101
        def gf43():
            return 9.29489610327295e-13*l**4 + 1.66597779450929e-9*l**3 - 3.02039662184564e-7*l**2 + 2.48027404545859e-5*l + 0.00125208272860948
        def gf44():
            return  9216.72714714835*(1.0932e-11*l**2 + 7.698e-9*l - 1.3311e-6)/(1.822e-12*l**4 + 2.566e-9*l**3 - 1.3311e-6*l**2 + 0.0003042*l - 0.02675) + 96.0037871500304*(7.288e-12*l**3 + 7.698e-9*l**2 - 2.6622e-6*l + 0.0003042)/(1.822e-12*l**4 + 2.566e-9*l**3 - 1.3311e-6*l**2 + 0.0003042*l - 0.02675) + 1

         # # print(gf40(),gf41(),gf42(),gf43(),gf44())
         # # print(gf00(),gf01(),gf02(),gf03(),gf04(),gf10(),gf11(),gf12(),gf13(),gf14(),gf20(),gf21(),gf22(),gf23(),gf24(),gf30(),gf31(),gf32(),gf33(),gf34())

        F=[[gf00(),gf01(),gf02(),gf03(),gf04()],[gf10(),gf11(),gf12(),gf13(),gf14()],[gf20(),gf21(),gf22(),gf23(),gf24()],[gf30(),gf31(),gf32(),gf33(),gf34()],[gf40(),gf41(),gf42(),gf43(),gf44()]]
        
        # print("matrix F",F)
        # print("")
        F_matrix.append(F)
        # print("Deter F",np.linalg.det(F))
        
        
        H=np.matlib.eye(n = 2, M = 5, k = 0, dtype = float)
        V=[[28*28/12,0],[0,28*28/12]]
        # # print(V)
        # m_k = [[eid_a[(i+1),6]],[eid_a[(i+1),4]]]
        m_k = [[snx[i]],[sny[i]]]
        
        # xm.append(m_k[0])
        # ym.append(m_k[1])
        
        x_k0 = np.transpose([[x0,y0,tx0,ty0,qp0]])
        # print("x_k0:",x_k0)
        x_k1 = np.transpose([[x_p,y_p,tx_p,ty_p,qp_p]])
        
        
        # print("x_k1:",x_k1)
        c_k0 = (x_k1-x_k0)*(np.transpose(x_k1-x_k0))

        if(i==0):   
            # c_k1 = F*c_k0*(np.transpose(F)) + F*np.array(Qk)*np.transpose(F)
            a=np.zeros((5,5) ,float)
            c_k0=np.fill_diagonal(a,10**6)
            print(c_k0)
        else:
            c_k0 = (x_k1-x_k0)*(np.transpose(x_k1-x_k0))
        # print("c_k0",c_k0)
           
        
        FN = np.matlib.eye(n = 5, M = 5, k = 0, dtype = float)
        
        for loop in range(len(F_matrix)-1):
            FN = FN*np.array(F_matrix[loop])
        FNF.append(FN)    
            
        if(i==0):   
            # c_k1 = F*c_k0*(np.transpose(F)) + F*np.array(Qk)*np.transpose(F)
            c_k1 = F*c_k0*(np.transpose(F)) + FN*np.array(Qk)*np.transpose(FN)
         
        else:
            # c_k1 = F*np.array(Ck_matrix[-1])*(np.transpose(F)) + F*np.array(Qk)*np.transpose(F)
            c_k1 = F*np.array(Ck_matrix[-1])*(np.transpose(F)) + FN*np.array(Qk)*np.transpose(FN)
            
        # print("c_k1",c_k1)
        # print("Deter c_k1",np.linalg.det(c_k1))
        # # invc_k1 = np.linalg.inv(c_k1) 
        kg2=(H*c_k1*np.transpose(H)+V).astype(np.float64)    # kg2 was trated as obj instead of float and hence the conversion
        k_gain= c_k1*np.transpose(H)*np.linalg.inv(kg2) 
        # print("k_gain",k_gain)
        x_k = x_k1 + k_gain*(m_k-H*x_k1)
        # print("State vector of",i+1,"plane:\n",x_k)
        filter_state_vector=x_k
        
        # xpf=x_k[0,0]
        # ypf=x_k[1,0]
        # txf=x_k[2,0]
        # tyf=x_k[3,0]
        # qpf=x_k[4,0]


        
        I= np.matlib.eye(n = 5, M = 5, k = 0, dtype = float)
        Ck = (I-k_gain*H)*c_k1*(np.transpose(I-k_gain*H))+k_gain*V*(np.transpose(k_gain))
         # # # print((I-k_gain*H)*c_k1*(np.transpose(I-k_gain*H)))
        # print("error Covariance:\n",Ck) 
        Ck_matrix.append(Ck)
        # print("\neid:",eid_a[i,0])
        # print("\nq:",x_k[4]*P_F[i])
        # print("\nP from state r:",x_k[4]*P_F[i]/x_k[4])

      
     #####################    arrays for plots (Predicted vs. Filtered for diff layres)     ############################
        ##note: we use two variables for the same value, the first one is to extract the value from a matrix
        #say xf, and its object type is matrix and can't be used for computation and hence we create an array out of it
        
        xp.append(x_k1[0])
        yp.append(x_k1[1])
        xpp=np.array(xp)
        ypp=np.array(yp)
        # txp.append(x_k1[2])
        # typ.append(x_k1[3])
        # qpp.append(x_k1[4])
        # txpp=np.array(txp)
        # typp=np.array(typ)
        # qppp=np.array(qpp)

        
        xf1.append(x_k[0])
        xff1=np.array(xf)
        yf1.append(x_k[1])
        yff1=np.array(yf)
        txf1.append(x_k[2,0])
        txff1=np.array(txf)
        tyf1.append(x_k[3,0])
        tyff1=np.array(tyf)
        # qpf.append(x_k[4])        
        # qpff=np.array(qpf)
       # print("xff:",xff[0])
        # print("xf:",xf)
       # print("len",len(xff))
          
    
         



    
     #################################      SMOOTHING      ###################################
    # print(xf)
    # print("txf",txf1)
    # qpr=[]
    # yps=[]
    # txps=[]
    # typs=[]
    # qp_ps=[]


    E_atplane_S=[E_atplane[-1]]
    P_S=[P_F[-1]]
    xs=[]
    ys=[]
    txs=[]
    tys=[]
    
    xks=[]
    yks=[]
    txks=[]
    tyks=[]
    qpks=[]
    # print(E_atplane_S)

    # print(len(F_matrix))
    # print("\n********************Smoothing*******************************\n")
    # print("-----------------------------------------------")
    for i in range(f_size-1,0,-1):
        
        # if (i!=0):
            # print("-----------------")
            # print("layer",i,"to",i-1)
            # print("-----------------")
        # print("e at plane",E_atplane[i-1])
        # print("E",E_atplane_S[-1])
        # print("P",P_S[-1])   
        def predict_qp_r():
            x1=[]
            y1=[]
            for j in moml[::-1]:
                # print("j",j)
                if((P_S[-1]/j)>1):
                    # print("appended P",P_S)
                     # print("j",j)                
                    index=moml.index(j)
                    # print("index",index)
                    x1=[mom[index], mom[index+1]]
                    y1=[de_dx[index], de_dx[index+1]]
                    break
            # print(x1,y1)   
            fx = interpolate.interp1d(x1, y1,kind = 'linear')
            en_loss= fx(P_S[-1])
            # print("en-loss",en_loss)
            en_atplane = E_atplane_S[-1] 
            # print("en_atplane",en_atplane)
            en_incident = en_atplane + (en_loss*step_size/10*rho_fe)
            # print("EnIncident",en_incident)
            p=math.sqrt((en_atplane)**2+(mass_mu)**2)
            # print("p at",i-1,"layer",p)
            # print("P_forward",P_F[i-1])
           # # # print("e/p",e/p)
            E_atplane_S.append(en_incident)
            P_S.append(p)
            QP_S.append(e/p)
            # print("Filter Qp",QP[i-1])
            # print("QP_S",QP_S[-1])
            return e/p

        def predict_x_r(x0,tx,ty,qP,dz):
            hh = h(tx,ty,qP)
            return x0 - tx*dz - hh*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))- hh**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))

        def predict_y_r(y0,tx,ty,qP,dz):
            hh = h(tx,ty,qP)
            return y0 - ty*dz - hh*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))- hh**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))

        def predict_tx_r(tx,ty,qP,dz):
            hh = h(tx,ty,qP)
            return  tx - hh*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))- hh**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))

        def predict_ty_r(tx,ty,qP,dz):
            hh = h(tx,ty,qP)
            return ty - hh*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz)) - hh**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))
        
        
        qp_pr=predict_qp_r()
        
        # xn=np.array(xf[i-1])
        # yn=np.array(yf[i-1])
        # txn=np.array(txf[i-1])
        # tyn=np.array(tyf[i-1])

        if (i==f_size-1):
            
            # xn=eid_a[i,6]
            # yn=eid_a[i,4]
            # txn=((eid_a[i,6]-eid_a[i-1,6])/step_size)
            # tyn=((eid_a[i,4]-eid_a[i-1,4])/step_size)
            # qpn=0
            xn=filter_state_vector[0,0]
            yn=filter_state_vector[1,0]
            txn=filter_state_vector[2,0]
            tyn=filter_state_vector[3,0]
            qpn=filter_state_vector[4,0]
            
        else:
            xn=xkr
            yn=ykr
            txn=txkr
            tyn=tykr
            qpn= qpkr
            # xn=xs[-1]
            # yn=ys[-1]
            # txn=txs[-1]
            # tyn=tys[-1]
            # qpn=0 
        # print("xn",xn,"\nyn",yn,"\ntxn",txn,"\ntyn",tyn)
        # T=math.sqrt(txn**2+tyn**2+1)
        # l=math.sqrt((eid_a[i,4]-eid_a[i-1,4])**2+(eid_a[i,5]-eid_a[i-1,5])**2+(eid_a[i,6]-eid_a[i-1,6])**2)
        dl=(step_size)/(1/T)
            
        x_pr=predict_x_r(xn,txn,tyn,qp_pr,step_size)
        y_pr=predict_y_r(yn,txn,tyn,qp_pr,step_size)
        tx_pr=predict_tx(txn,tyn,qp_pr,step_size)
        ty_pr=predict_ty(txn,tyn,qp_pr,step_size)

        xs.append(x_pr)
        ys.append(y_pr)
        txs.append(tx_pr)
        tys.append(ty_pr)
        
        # print("xn",xn,"\nyn",yn,"\ntxn",txn,"\ntyn",tyn)
        
        # print("xn",xn,"xp",x_pr,"\nyn",yn,"yp",y_pr,"\ntxn",txn,"tx_p",tx_pr,"\ntyn",tyn,"typ",ty_pr)
        
        m_k_r = [[eid_a[(i-1),6]],[eid_a[(i-1),4]]]

        x_kn = np.transpose([[xn,yn,txn,tyn,qpn]])
        # print("xn",x_kn)
        x_kpr = np.transpose([[x_pr,y_pr,tx_pr,ty_pr,qp_pr]])
        # print("x_kpr",x_kpr)
        c_kn = (x_kpr-x_kn)*(np.transpose(x_kpr-x_kn))

        # # F_matrix is a list of 59(f_size-1) matrices
        F_matrix_r = F_matrix[i-1] 
        F_r_a.append(F_matrix_r)
        FNF_r=FNF[i-1]
        # print(FNF_r)
        
        Qk_r = np.array(Qk_matrix[i-1])
        # print((Qk_r))
        
        # D=-1 for filtering and D=+1 for smoothing......the following are the steps to negate the first four elements of the first four rows
        for j in range(len(Qk)-1):
            num1=j
            for k in range(len(Qk)-1):
                num2=k
                temp=(Qk_r[num1,num2])
                Qk_r[num1,num2]=-temp
                # print("temp",Qk_r[num1,num2])
        # # print("qk bwd:",Qk_r)    
        # 
    
        # c_kn1 =F_matrix_r*c_kn*(np.transpose(F_matrix_r)) +  FNF_r*Qk_r*np.transpose(FNF_r)
        if(i==f_size-1):   
            c_kn1 = F_matrix_r*c_kn*(np.transpose(F_matrix_r)) + FNF_r*Qk_r*np.transpose(FNF_r)
            # c_kn1 =F_matrix_r*c_kn*(np.transpose(F_matrix_r)) +  F_matrix_r*Qk_r*np.transpose(FNF_r)
         
        else:
            c_kn1 = F_matrix_r*np.array(Ckrr[-1])*(np.transpose(F_matrix_r)) +  FNF_r*Qk_r*np.transpose(FNF_r)
            # c_kn1 =F_matrix_r*np.array(Ckrr[-1]*(np.transpose(F_matrix_r)) +  F_matrix_r*Qk_r*np.transpose(FNF_r)
        # # print("x_kn:",x_kn)
        # # print("x_kpr:",x_kpr)
        # # print("c_k:",c_kn)
        # # print("c_k1:",c_kn1)

        H = np.matlib.eye(n = 2, M = 5, k = 0, dtype = float)
        V=[[28*28/12,0],[0,28*28/12]]
        Ak2=(H*c_kn1*np.transpose(H)+V).astype(np.float64)    # A_k2 was trated as obj instead of float and hence the conversion
        A_k= c_kn1*np.transpose(H)*np.linalg.inv(Ak2)
        # # print("Smoother_gain",A_k)
        x_kr = x_kpr + A_k*(m_k_r-H*x_kpr)
        # print("State vextor of ",i-1,"plane:",x_kr)   
        # print("x_kpr",x_kpr[0])
        # print("x_kr",x_kr[0])
        I= np.matlib.eye(n = 5, M = 5, k = 0, dtype = float)
        Ck_r = (I-A_k*H)*c_kn1*(np.transpose(I-A_k*H)) + A_k*V*(np.transpose(A_k))
        Ckrr.append(Ck_r)

        xkr=x_kr[0,0]
        ykr=x_kr[1,0]
        txkr=x_kr[2,0]
        tykr=x_kr[3,0]
        qpkr=x_kr[4,0]
        

        # xks.append(x_kr[0])
        # xkr=np.array(xks)
        # xkr.astype(float)
        
        # yks.append(x_kr[1])
        # ykr=np.array(yks)
        # ykr.astype(float)

        # txks.append(x_kr[2])
        # txkr=np.array(txks)
        # txkr.astype(float)
        # # txkr=np.dtype('d')
        
        # tyks.append(x_kr[3])
        # tykr=np.array(tyks)
        # tykr.astype(float)
        # # tykr=np.dtype('d')

        # qpks.append(x_kr[4])
        # qpkr=np.array(qpks)
        # qpkr.astype(float)
        # txkr=np.dtype('d')  

        # print("\neid:",eid_a[i,0])
        # print("\nq:",x_kr[4]*P_S[i])
        # print("\nP from state vector:",x_kr[4]*P_S[i]/x_kr[4]) # qpr.append(x_kr[4])
        # qprr=np.array(qpr)
    # plot_ps.append(P_S[-1])
    # plot_pf.append(P_F[0])
    
    # print("eid",eid_a[n,0])
    # print("\nq:",x_kr[4]*P_S[-1])
    # print("\nP:",P_S[-1])
# # ###############     PLOTS OF STATE VECTORS (UPDATED VS MEASURED)  ############################
# plt.hist((5000-np.array(plot_ps))/1000,bins=100)
# plt.xlabel("P true - P kalman (GeV/c)")
# plt.ylabel("frequency")
# plt.show()
# print(xffff)
# print("///////////////////////////////////////")
# print(filter_state_vector[0,0])
# print(filter_state_vector[1,0])
# print(filter_state_vector[2,0])
# print(filter_state_vector[3,0])
# print(filter_state_vector[4,0])
# print(P_S)
# %%
# print("Qk",Qk_matrix)
xr=np.arange(f_size-1,0,-1)
x=np.arange(f_size-1)

# plt.plot(x,xm,'gv',label='Measured position')
# plt.legend(loc=1)
# plt.plot(x,xpp.reshape(f_size-1),'b+',label='Predicted position')
# plt.legend(loc=1)
# plt.plot(x,xff.reshape(f_size-1),'r*',label='Filtered position')
# plt.legend(loc=1)
# plt.plot(xr,xkr.reshape(f_size-1),'cP',label='Smoothed position')
# plt.legend(loc=1)
# plt.xlabel("layers")
# plt.title("X-hit")
# plt.savefig("x-hit 5 gev")
# plt.show()

# plt.plot(x,ym,'gv',label='Measured position')
# plt.legend(loc=4)
# plt.plot(x,ypp.reshape(f_size-1),'r*',label='Predicted position')
# plt.legend(loc=4)
# plt.plot(x,yff.reshape(f_size-1),'b+',label='Filtered position')
# plt.legend(loc=4)
# plt.plot(xr,ykr.reshape(f_size-1),'cP',label='Smoothed position')
# plt.legend(loc=4)
# plt.xlabel("layers")
# plt.title("Y-hit")
# plt.savefig("y-hit 5 gev")
# plt.show()

# print("ym",ym,"yf",yff.reshape(f_size-1),"yp",ypp.reshape(f_size-1),"ys",ysmf.reshape(f_size-1))

# plt.plot(x,txm,'gv',label='Measured position')
# plt.legend(loc=1)
# plt.plot(x,txf1,'b+',label='Filtered position')
# plt.legend(loc=1)
# plt.plot(x,txkr.reshape(f_size-1),'r*',label='Smoothed position')
# plt.legend(loc=1)
# plt.plot(x,txf,'r*',label='Predicted position')
# plt.legend(loc=1)
# plt.xlabel("layers")
# plt.title("tx")
# plt.show()

# plt.plot(x,tym,'gv',label='Measured position')
# plt.legend(loc=3)
# plt.plot(x,tyf1,'b+',label='Filtered position')
# plt.legend(loc=2)
# plt.plot(x,tykr.reshape(f_size-1),'r*',label='Smoothed position')
# plt.legend(loc=2)
# plt.plot(x,tyf,'r*',label='Predicted position')
# plt.legend(loc=3)
# plt.xlabel("layers")
# plt.title("ty")
# plt.show()

# plt.plot(x,qpff.reshape(f_size-1),'r*',label='Filtered position')
# plt.legend(loc=1)
# plt.plot(x,qppp.reshape(f_size-1),'b+',label='Predicted position')
# plt.legend(loc=1)
# plt.xlabel("layers")
# plt.title("qp")
# # plt.savefig("qp")
# plt.show()


# print("X Strip Numbers:",snx)
# print("Y strip numbers:",sny)

#Smoothed cs
# xr=np.arange(f_size-1,0,-1)
# plt.plot(x,qpff.reshape(f_size-1),'r*',label='Filtered position')
# plt.legend(loc=1)
# plt.plot(xr,qprr.reshape(f_size-1),'b+',label='Smoothed position')
# plt.legend(loc=1)
# plt.xlabel("layers")
# plt.title("qp")
# # plt.savefig("qp_14GeV")
# plt.show()


# QP_SR=QP_S[::-1]
# print(QP,QP_SR)
# num=0
# for i in range(f_size-1):
#     if(round(QP[i],10)==round(QP_SR[i],10)):
#         num+=1
# print(num)        
    
# for i in range(f_size-2,-1,-1):
#     print("QP_S",QP_S[i])        
    
    # a=df1[i,6]*10**(-3)*(-256)/8
    # snx.append(int(a))
    # b=df1[i,4]*10**(-3)*256/8
    # sny.append(int(b))



# %% 


