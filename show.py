import numpy as np
import cPickle
import sys
import gzip
from multiprocessing import Pool
from PIL import Image as im
import numpy as np
def calculate_KL(i):
    x=x_test[i]
    z=hidden[i]
    y = t_test[i]
    [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,pi] = params
    num_model = len(pi)
    dimZ = W2.shape[0]/num_model
    h_encoder = np.tanh(np.dot(W1,x) + b1[:,0])
    KL = []
    b=[t for t in range(0,num_model)]
    j = np.random.uniform(0,1)
    pi_soft = np.exp(pi)/np.sum(np.exp(pi))
    i = 0
    for i in range(0,num_model):
        if pi_soft[i] > j :
            break
        else:
            j -= pi_soft[i]
    mu = np.dot(W2[int(i)*dimZ:(1+int(i))*dimZ],h_encoder) + b2[int(i)*dimZ:(1+int(i))*dimZ][0]
    log_sigma = (0.5*(np.dot(W3[int(i)*dimZ:(1+int(i))*dimZ],h_encoder)))+ b3[i*dimZ:(1+i)*dimZ][0]
    k = dimZ
    eps = np.random.normal(0,1,[dimZ,1])
    z = mu + np.exp(log_sigma)*eps[:,0]
    qxz = np.power(2*np.pi,-dimZ/2)*np.exp(np.sum(log_sigma))*np.exp(-0.5*np.dot((z-mu)*(z-mu),np.exp(log_sigma).T))
    pz=np.power(2*np.pi,-dimZ/2)*np.exp(-0.5*np.sum(np.square(z)))
    pz = np.exp(-0.5*np.sum(np.square(z)))
    kl = np.log((qxz+1)/(pz+1))*qxz
    KL.append(kl)
    h_decoder = np.tanh(np.dot(W4,z) + b4[:,0])
    o = 1/(1+np.exp(-np.dot(W5,h_decoder) + b5[:,0]))
    img = im.fromarray(255*(o).reshape((28,28)))
    img.show()
    img = im.fromarray(255*(x).reshape((28,28)))
    img.show()
    logpxz = np.sum(x*np.log(o)[:,0]+(1.0-x)*np.log(1.0-o)[:,0])
    return np.sum(KL)/np.sum(np.exp(pi))+logpxz
iter = sys.argv[1]
i = sys.argv[2]
param_File = file_name = 'VAE_dump'+str(iter)+'.pkl'
f = open(param_File,'r')
params = cPickle.load(f)
f.close()
param_File = file_name = 'hidden_dump'+str(iter)+'.pkl'
f = open(param_File,'r')
hidden = cPickle.load(f)
f.close()
f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()
[N,dimX] = x_test.shape
calculate_KL(int(i))
