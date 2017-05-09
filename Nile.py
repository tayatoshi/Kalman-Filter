
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

nile=pd.read_csv('Nile.csv',index_col=0,parse_dates=True)
y=np.array(nile)
y_train = y[:80]
plt.plot(nile)
plt.title('Nile')
plt.show()


# In[ ]:

class Calculation1:
    def __init__(self):
        pass
    
    def kf(y_train, a0, p0, e, h, Z, T):
        length=len(y_train)
        a = a0
        p = p0
        P_t = np.zeros(length)
        A_t = np.zeros(length)
        vero = np.zeros(length)
        for i in range(length):
            u = y_train[i] - Z*a
            F = Z*p*Z + e
            K = p*Z/F
            L = 1 - K
            A_t[i] = a + K * u
            P_t[i] = p -K*F*K
            a = T*A_t[i]
            p = T*P_t[i]*T + h
            vero[i] = np.log(F) + (u ** 2)/F
        vero_fit = -0.5*sum(vero)
        return({'A_t':A_t, 'P_t':P_t, 'vero_fit':vero_fit, 'y_hat':Z*A_t})


# In[ ]:

class Calculation2:
    def __init__(self):
        pass
    
    #pars(np.array([y,e,h,Z,T]))
    def pars(p):
        y = p[0]
        e = np.exp(p[1])
        h = np.exp(p[2])
        Z = p[3]
        T = p[4]
        result = Calculation1.kf(y,0.0,10.0**6,e,h,Z,T)
        return(result)


# In[ ]:

class KF_estimate(Calculation1,Calculation2):
    def __init__(self,data,init_e,init_h,init_z,init_t):
        self.data = data
        self.length = len(data)
        re = []
        L=None
        for e in np.arange(init_e[0],init_e[1],init_e[2]):
            for h in np.arange(init_h[0],init_h[1],init_h[2]):
                for z in np.arange(init_z[0],init_z[1],init_z[2]):
                    for t in np.arange(init_t[0],init_t[1],init_t[2]):
                        l=Calculation2.pars(np.array([data,e,h,z,t]))['vero_fit']
                        re.append(l)
                        print('(',e,h,z,t,')',l)
                        if(L==None or l>L):
                            L=l
                            ep = e
                            eta = h
                            Z=z
                            T=t
        print('\n')
        print('===========================================')
        print('log Likelihood = ', L)
        #print('===========================================')
        print('-------------------------------------------')
        print('e = ', ep, 'h = ', eta, 'Z = ', Z, 'T = ', T)
        self.estimation={"L":L, "e":ep, "h":eta, "Z":Z, "T":T}
        self.prediction=Calculation2.pars(np.array([data,self.estimation['e'],self.estimation['h'],self.estimation['Z'],self.estimation['T']]))
    
    def predict(self):
        return(self.prediction)
    
    def forecast(self, n = 20):
        self.prediction
        forecast_a = np.zeros(n)
        forecast_A = self.prediction['A_t'][-1]
        for i in range(n):
            forecast_a[i] = self.estimation['Z']*forecast_A
            forecast_A = forecast_a[i]
        forecast_a = pd.DataFrame(forecast_a)
        forecast_a.index = forecast_a.index + self.length
        return(forecast_a)
    
    def smoothing(self):
        smoothe=np.zeros(self.length)
        for i in range(self.length):
            smooth[i]=1
        return()


# In[ ]:

init_e=np.array([9,10,0.01])
init_h=np.array([7,8,0.01])
init_z=np.array([1,2,1])
init_t=np.array([1,2,1])
kfe=KF_estimate(y_train,init_e,init_h,init_z,init_t)


# In[ ]:

pre=kfe.predict()
fore=kfe.forecast()


# In[ ]:

plt.plot(y,label='observation')
plt.plot(pre['A_t'],label='predict')
plt.plot(fore,label='forecast')
plt.legend()
plt.title('Flow of River Nile')
plt.show()


# In[ ]:

plt.plot(pre['P_t'])
plt.show()


# In[ ]:

np.dot(2,2)


# In[ ]:



