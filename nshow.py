import numpy as np
import cPickle
import sys
import gzip
from multiprocessing import Pool
from PIL import Image as im

def calculate_KL(i):
    x=x_train[i]
    z=hidden[i]
    y = t_test[i]
    [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5] = params
    dimZ = W2.shape[0]
    h_encoder = np.tanh(np.dot(W1,x) + b1[:,0])
    KL = []
    mu = np.dot(W2*dimZ,h_encoder) + b2[0]
    log_sigma = (0.5*(np.dot(W3,h_encoder)))+ b3[0]
    k = dimZ
    qxz = (np.power(2*np.pi,-dimZ/2)*np.exp(np.sum(log_sigma))*np.exp(-0.5*np.dot((z-mu)[:,0]*(z-mu)[:,0],np.exp(log_sigma).T)))
    pz=np.power(2*np.pi,-dimZ/2)*np.exp(-0.5*np.sum(np.square(z[:,0])))
    pz = np.exp(-0.5*np.sum(np.square(z[:,0])))
    kl = -qxz
    KL.append(kl)
    h_decoder = np.tanh(np.dot(W4,z) + b4)
    o = 1/(1+np.exp(-np.dot(W5,h_decoder) + b5))
    img = im.fromarray(255*(o).reshape((28,28)))
    img.show()
    logpxz = np.sum(x*np.log(o)[:,0]+(1.0-x)*np.log(1.0-o)[:,0])
    return np.sum(KL)+logpxz
iter = sys.argv[1]
param_File = file_name = 'non_VAE_dump'+str(iter)+'.pkl'
f = open(param_File,'r')
params = cPickle.load(f)
f.close()
param_File = file_name = 'non_hidden_dump'+str(iter)+'.pkl'
f = open(param_File,'r')
hidden = cPickle.load(f)
f.close()
f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()
[N,dimX] = x_test.shape
hiddens = []
indices = [x for x in range(0,N)]
po = Pool(10)
#hiddens = po.map(calculate_KL, indices)
print calculate_KL(1)
#print np.mean(hiddens)
