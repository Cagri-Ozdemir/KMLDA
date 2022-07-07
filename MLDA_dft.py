import numpy as np
from scipy import linalg

def tproddft(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    C = np.zeros((n0,n1,nc), dtype=complex)
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    D = np.fft.fft(A, axis = 0)
    Bhat = np.fft.fft(B, axis = 0)

    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])

    Cx = np.fft.ifft(C, axis=0)
    #Cx = np.real(Cx)

    return Cx.real

def ttransdft(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b),dtype='complex')
    B[0,:,:] = np.transpose(A[0,:,:])
    for j in range(a-1,0,-1):
        B[a-1-j+1,:,:] = np.transpose(A[j,:,:])
    #B = np.real(B)
    return B

def tinvdft(A):
    (a, b, c) = A.shape
    C = np.zeros((a, b, c), dtype=complex)
    D = np.fft.fft(A, axis=0)
    for j in range(a):
        C[j,:,:]=np.linalg.inv(D[j,:,:])
    C1 = np.fft.ifft(C,axis=0)
    #C1 = np.real(C1)
    return C1.real



def Class_scatters_dft(num_class,Tensor_train,y_train):
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
          Sa = Sa + tproddft(a,ttransdft(a))
      Sw = Sw+Sa
    for i in range(num_class):
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddft(b,ttransdft(b)))*num_class

    return Sw,Sb

def teigdft(A):
    (n0,n1,n2) = A.shape
    U1 = np.zeros((n0,n1,n2),dtype='complex')
    S1 = np.zeros((n0, n1, n2),dtype='complex')

    arr = np.fft.fft(A, axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      S, U = np.linalg.eig(M)
      np.fill_diagonal(S1[i,:,:],S)
      U1[i, :, :] = U
    U1x = np.fft.ifft(U1,axis=0)
    S1x = np.fft.ifft(S1, axis=0)
    U1x = np.real(U1x)
    S1x = np.real(S1x)
    return S1x,U1x

def tSVD(A):
    n0,n1,n2 = A.shape
    U1 = np.zeros((n0,n1,n1),dtype=complex)
    S1 = np.zeros((n0, n1, n2),dtype=complex)
    V1 = np.zeros((n0,n2,n2),dtype=complex)
    A = np.fft.fft(A, axis=0)

    for i in range(n0):
      (U, S, Vt) = np.linalg.svd(A[i,:,:],full_matrices='true')
      np.fill_diagonal(S1[i,:,:],S)
      U1[i,:,:] = U
      Vc = np.conj(Vt)
      V1[i,:,:] = Vc.T

    U1x = (np.fft.ifft(U1, axis=0))
    S1x = (np.fft.ifft(S1, axis=0))
    V1x = (np.fft.ifft(V1, axis=0))


    return U1x.real, S1x.real, V1x.real

def tsvd_op(Sw,Sb):
    A = tproddft(Sb,tinvdft(Sw))
    U,S,V = tSVD(A)
    Sww = np.fft.fft(Sw,axis=0)
    SS = np.fft.fft(S,axis=0)
    n0,n1,n2 = Sww.shape
    SL = np.zeros((n0,n1,n2),dtype=complex)
    for i in range(n0):
        SL[i,:,:] = Sww[i,:,:]*SS[i,0,0]

    SLL = np.fft.ifft(SL,axis=0)
    Uu,Ss,Vv = tSVD(Sb.real-SLL.real)
    return Uu,Ss




def trot(A):
    n1,n2,n3 = A.shape
    B = np.zeros((n2,n1,n3))
    for i in range(n3):
        B[:,:,i] = np.transpose(A[:,:,i])
    return B

def trotcomp(A):
    n1,n2,n3 = A.shape
    B = np.zeros((n2,n1,n3),dtype='complex')
    for i in range(n3):
        B[:,:,i] = np.transpose(A[:,:,i])
    return B

def Class_scatters_dftcomp(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class),dtype='complex')
    Sw =  np.zeros((n1,n2,n2),dtype='complex')
    Sb = np.zeros((n1, n2, n2),dtype='complex')
    a = np.zeros((n1,n2,1),dtype='complex')
    b = np.zeros((n1, n2, 1),dtype='complex')
    Mean_tensor = np.zeros((n1, n2, 1),dtype='complex')
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    for i in range(num_class):
      Sa = np.zeros((n1, n2, n2),dtype='complex')
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      for j in idx:
          a[:,:,0] = Tensor_train[:,:,j]-mean_tensor_train[:,:,i]
          Sa = Sa + tproddft(a,ttransdft(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddft(b,ttransdft(b)))*occurrences

    return Sw,Sb

def fronorm(A):
    tmp = A*A
    B = np.absolute(tmp)
    C = np.sum(B)
    y = np.sqrt(C)
    return y
####################################################functions for kernels
def kernel2_G_dft(A):
    d0,d1,d2 = A.shape
    Gbar = np.zeros((d0,d2,d2))
    a = np.zeros((d0,d1,1))
    b = np.zeros((d0, d1, 1))
    for k in range(d2):
          for j in range(d2):
            a[:,:,0] = A[:,:,j]
            b[:,:,0] = A[:,:,k]
            Gbar[:,j,k] = ((tproddft(ttransdft(a),b)[:,0,0])+0.5)**1.3
    return Gbar

def kernel2_test_data_dft(Train,Test):
    d0,d1,d2 = Train.shape
    d00,d11,d22 = Test.shape
    a = np.zeros((d0, d1, 1))
    b = np.zeros((d0, d1, 1))
    Gbar = np.zeros((d0,d2,d22))
    for k in range(d22):
          for j in range(d2):
              a[:, :, 0] = Train[:, :, j]
              b[:, :, 0] = Test[:, :, k]
              Gbar[:,j,k] = ((tproddft(ttransdft(a),b)[:,0,0])+0.5)**1.3
    return Gbar

def Class_scatters2_dft(num_class,Tensor_train,y_train):
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
      Jn = np.ones((n1, int(occurrences), 1))
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      aa[:,:,0] = mean_tensor_train[:,:,i]
      a = Tensor_train[:,:,idx] - tproddft(aa,ttransdft(Jn))
      Sa = Sa + tproddft(a,ttransdft(a))

    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddft(b,ttransdft(b)))*occurrences

    return Sa,Sb