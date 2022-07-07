import numpy as np
from MLDA_dwt import pred,tproddwt,tinv,ttransx,teigdwt,kernel2_G_dwt,kernel2_test_data_dwt,Class_scatters_dwt,Class_scatters2_dwt
from MLDA_dft import fronorm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pywt
from scipy.fftpack import  idct
data = pd.read_excel('fea.xlsx')
data_array  = data.to_numpy()
A = np.double(data_array)
out = np.zeros(A.shape, np.double)
normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
# data labels Y
Y = np.zeros((400,1))
k = 0
for i in range(40):
    Y[0+k:10+k] = i+1
    k = k + 10
iik =0
eig=5
#fig,axs = plt.subplots(2,3)

for ii in range(1):
 X_train, X_test, y_train, y_test = train_test_split(normalized, Y , test_size=0.3,  random_state=ii)

 Tensor_train = np.zeros((32,32,280))
 (k,m,n) = Tensor_train.shape

 Tensor_test = np.zeros((32,32,120))
 (k1,m1,n1) = Tensor_test.shape


 for i in range(n):
     Tensor_train[:,:,i] = np.transpose(np.reshape(X_train[i,:],(k,m)))

 for i in range(n1):
     Tensor_test[:,:,i] = np.transpose(np.reshape(X_test[i,:],(k1,m1)))


 train_kernel = kernel2_G_dwt(Tensor_train)
 test_kernel = kernel2_test_data_dwt(Tensor_train, Tensor_test)

 #Coeffs = pywt.wavedec(train_kernel,'haar',level=1, axis=1)
 #Coef = np.concatenate((Coeffs[0],Coeffs[1]),axis=1)
 #train_kernel = pywt.idwt(Coeffs[0][:,:70,:],Coeffs[0][:,70:140,:],'haar', axis=1)
 #fro = np.zeros((280))
 #for i in range(280):
 # fro[i] = (fronorm(Coef[:,i,:]))

 #Coeffs = pywt.wavedec(test_kernel, 'haar', level=1, axis=1)
 #Coef = np.concatenate((Coeffs[0], Coeffs[1]), axis=1)
 #test_kernel = pywt.idwt(Coeffs[0][:, :70, :], Coeffs[0][:, 70:140, :], 'haar', axis=1)
 Sw,Sb = Class_scatters2_dwt(40, train_kernel, y_train)
 II = np.eye(280,280)
 III = np.zeros((32,280,280))
 for i in range(32):
  III[i,:,:] = II*0.1
 III = pywt.idwt(III[:16,:,:],III[16:32,:,:], 'haar', axis=0)
 Sww = Sw + (III)
 S = tproddwt(tinv(Sww), Sb)
 SS, UU = teigdwt(S)
 Ac = np.zeros((39))
 for iii in range(39):
  u = UU[:, :, :iii]
  pro_df_trn = tproddwt(ttransx(u), train_kernel)
  pro_df_tst = tproddwt(ttransx(u), test_kernel)
  test_pre, accuracy = pred(pro_df_trn, pro_df_tst, y_test, y_train)
  Ac[iii] = accuracy
################
 Sw2, Sb2 = Class_scatters_dwt(40, Tensor_train, y_train)
 S2 = tproddwt(tinv(Sw2), Sb2)
 SS2, UU2 = teigdwt(S2)
 Ac2 = np.zeros((32))
 for iii in range(32):
  u2 = UU2[:, :, :iii]
  pro_df_trn2 = tproddwt(ttransx(u2), Tensor_train)
  pro_df_tst2 = tproddwt(ttransx(u2), Tensor_test)
  test_pre2, accuracy2 = pred(pro_df_trn2, pro_df_tst2, y_test, y_train)
  Ac2[iii] = accuracy2


x1 = np.arange(2,40)
x2 = np.arange(2,33)
plt.figure()
plt.plot(x1,Ac[1:39],"--b",label="KMLDA")
plt.plot(x2,Ac2[1:32],"-*r",label="MLDA")
plt.title("ORL",fontsize=16)
plt.xlabel("Feature Dimension",fontsize=12)
plt.ylabel("Accuracy rate (%)",fontsize=12)
plt.ylim([60,100])
plt.legend(fontsize=14)