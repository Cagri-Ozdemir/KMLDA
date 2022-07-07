import numpy as np
import pywt
from scipy import linalg

def tproddwt(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    C = np.zeros((n0, n1, nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    coeffsB = pywt.dwt(B, 'haar', axis=0)
    cAA, cDD = coeffsB
    Bhat = np.concatenate((cAA, cDD))
    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])
    cC1 = C[0:z2, :, :]; cC2 = C[z2:z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return Cx
def tproddb4(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    D = np.concatenate((coeffsA[0],coeffsA[1]), axis=0)
    coeffsB = pywt.dwt(B, 'db4', axis=0,mode='periodization')
    Bhat = np.concatenate((coeffsB[0], coeffsB[1]), axis=0)
    (z1,z2,z3) = Bhat.shape
    C = np.zeros((z1,n1,nc))
    for i in range(z1):
        C[i, :, :] = np.matmul(D[i, :, :], Bhat[i, :, :])
    cC1 = C[0:int(z1/2), :, :];    cC2 = C[int(z1/2):z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return Cx

def ttransx(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b))

    for j in range(a):
        B[j,:,:] = np.transpose(A[j,:,:])
    return B

def tinv(A):
    (n0, n1, n2) = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z2, :, :]
    cC2 = D2[z2:z1, :, :]

    D3 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return D3
def tinvdb4(A):
    coeffs = pywt.dwt(A, 'db4', axis=0, mode='periodization')
    D = np.concatenate((coeffs[0], coeffs[1]), axis=0)
    (n0, n1, n2) = D.shape
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    z2 = int(n0 / 2)
    cC1 = D2[0:z2, :, :]
    cC2 = D2[z2:n0, :, :]
    D3 = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return D3






def pred(U_tr, U_tst, test_labels, train_labels):

    (l,m,n) = U_tr.shape

    (l1,m1,n1) = U_tst.shape

    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.linalg.norm(U_tst[:, :, i] - U_tr[:, :, j], ord='fro')
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy

def pred22(U_tr,U_tr1, U_tst, U_tst1, test_labels, train_labels):

    (l,m,n) = U_tr.shape
    (ll, mm, nn) = U_tr1.shape
    (l1,m1,n1) = U_tst.shape
    (ll1, mm1, nn1) = U_tst1.shape
    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.linalg.norm(U_tst[:, :, i] - U_tr[:, :, j], ord='fro')+np.linalg.norm(U_tst1[:, :, i] - U_tr1[:, :, j], ord='fro')
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy

def Class_scatters_dwt(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class))
    Sw =  np.zeros((n1,n2,n2))
    Sb = np.zeros((n1, n2, n2))
    a = np.zeros((n1,n2,1))
    b = np.zeros((n1, n2, 1))
    Mean_tensor = np.zeros((n1, n2, 1))
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    for i in range(num_class):
      Sa = np.zeros((n1, n2, n2))
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      for j in idx:
          a[:,:,0] = Tensor_train[:,:,j]-mean_tensor_train[:,:,i]
          Sa = Sa + tproddwt(a,ttransx(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddwt(b,ttransx(b)))*occurrences

    return Sw,Sb
def Class_scatters_db4(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class))
    Sw =  np.zeros((n1,n2,n2))
    Sb = np.zeros((n1, n2, n2))
    a = np.zeros((n1,n2,1))
    b = np.zeros((n1, n2, 1))
    Mean_tensor = np.zeros((n1, n2, 1))
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    for i in range(num_class):
      Sa = np.zeros((n1, n2, n2))
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      for j in idx:
          a[:,:,0] = Tensor_train[:,:,j]-mean_tensor_train[:,:,i]
          Sa = Sa + tproddb4(a,ttransx(a))
      Sw = Sw+Sa
    for i in range(num_class):
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddb4(b,ttransx(b)))*num_class

    return Sw,Sb

def teigdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n2))
    S1 = np.zeros((n0, n1, n2))

    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    arr = np.concatenate((cA, cD))
    for i in range(n0):
      M = arr[i, 0:]
      s, u = np.linalg.eig(M)
      idx = np.argsort(s)
      idx = idx[::-1][:n2]
      s = s[idx]
      u = u[:, idx]
      #s, u = linalg.cdf2rdf(S, U)
      np.fill_diagonal(S1[i,:,:],s.real)
      U1[i,:,:] = u.real
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    return S1,U1

def teigdb4(A):
    coeffs = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    arr = np.concatenate((coeffs[0], coeffs[1]), axis=0)
    (n0,n1,n2) =arr.shape
    U1 = np.zeros((n0, n1, n2))
    S1 = np.zeros((n0, n1, n2))
    for i in range(n0):
        M = arr[i, 0:]
        S, U = np.linalg.eig(M)
        s, u = linalg.cdf2rdf(S, U)
        S1[i, :, :] = s
        U1[i, :, :] = u
    z2 = int(n0 / 2)
    cU1 = U1[0:z2, :, :];    cU2 = U1[z2:n0, :, :]
    cS1 = S1[0:z2, :, :];    cS2 = S1[z2:n0, :, :]
    U1 = pywt.idwt(cU1, cU2, 'db4', axis=0,mode='periodization')
    S1 = pywt.idwt(cS1, cS2, 'db4', axis=0,mode='periodization')
    return S1, U1


def tSVDdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    #cA = np.zeros((z2,n1,n2))
    arr = np.concatenate((cA, cD),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    cV1 = V1[0:z2, :, :]; cV2 = V1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    V1 = pywt.idwt(cV1,cV2, 'haar', axis=0)
    return U1, S1, V1
def tSVDdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    #cA = np.zeros((z2,n1,n2))
    arr = np.concatenate((cA, cD),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    cV1 = V1[0:z2, :, :]; cV2 = V1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    V1 = pywt.idwt(cV1,cV2, 'haar', axis=0)
    return U1, S1, V1
####################################################functions for kernels
def kernel2_G_dwt(A):
    d0,d1,d2 = A.shape
    Gbar = np.zeros((d0,d2,d2))
    a = np.zeros((d0,d1,1))
    b = np.zeros((d0, d1, 1))
    for k in range(d2):
          for j in range(d2):
            a[:,:,0] = A[:,:,j]
            b[:,:,0] = A[:,:,k]
            Gbar[:,j,k] = np.abs((tproddwt(ttransx(a),b)[:,0,0])+1)**0.8
    return Gbar

def kernel2_test_data_dwt(Train,Test):
    d0,d1,d2 = Train.shape
    d00,d11,d22 = Test.shape
    a = np.zeros((d0, d1, 1))
    b = np.zeros((d0, d1, 1))
    Gbar = np.zeros((d0,d2,d22))
    for k in range(d22):
          for j in range(d2):
              a[:, :, 0] = Train[:, :, j]
              b[:, :, 0] = Test[:, :, k]
              Gbar[:,j,k] = np.abs((tproddwt(ttransx(a),b)[:,0,0])+1)**0.8
    return Gbar

def Class_scatters2_dwt(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class))
    Sw =  np.zeros((n1,n2,n2))
    Sb = np.zeros((n1, n2, n2))
    aa = np.zeros((n1,n2,1))
    b = np.zeros((n1, n2, 1))
    Mean_tensor = np.zeros((n1, n2, 1))
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    Sa = np.zeros((n1, n2, n2))
    for i in range(num_class):


      occurrences = np.count_nonzero(y_train == i+1)
      Jn = np.ones((n1, int(occurrences), int(occurrences)))
      Jn = pywt.idwt(Jn[:int(n1/2), :, :], Jn[int(n1/2):int(n1), :, :], 'haar', axis=0)
      I = np.zeros((n1, int(occurrences), int(occurrences)))
      for ll in range(n1):
          I[ll,:,:] = np.eye(int(occurrences))
      II = pywt.idwt(I[:int(n1 / 2), :, :], I[int(n1 / 2):int(n1), :, :], 'haar', axis=0)
      Cn = II - Jn/int(occurrences)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      aa[:,:,0] = mean_tensor_train[:,:,i]
      H1 = tproddwt(Tensor_train[:,:,idx],Cn)
      H2 = tproddwt(H1,ttransx(Tensor_train[:,:,idx]))
      Sa = Sa + H2

    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddwt(b,ttransx(b)))*occurrences

    return Sa,Sb


A = np.random.rand(4,4,4)
I = np.zeros((4,4,4))
M = np.zeros((4,4,1))
for i in range(4):
    I[i,:,:] = np.eye(4)
II = pywt.idwt(I[:2,:,:],I[2:4,:,:], 'haar', axis=0)
Jn = np.ones((4,4,4))
Jn = pywt.idwt(Jn[:2,:,:],Jn[2:4,:,:], 'haar', axis=0)
Cn = II - Jn/4
mean = A.sum(axis=2)/4
M[:,:,0] = mean
AA = A-M

BB = tproddwt(Cn,Cn)
S,U = teigdwt(Cn)
CC = Cn-BB
