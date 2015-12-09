"""
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com

License: MIT
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams
"""This class implements an auto-encoder with Variational Bayes"""
th.config.exception_verbosity='high'
th.config.optimizer='None'
class VA:
    def __init__(self, HU_decoder, HU_encoder, num_model, dimX, dimZ, batch_size, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder
        self.HU_encoder = HU_encoder

        self.num_model =num_model
        self.dimX = dimX
        self.dimZ = dimZ
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.sigmaInit = 0.01
        self.lowerbound = 0
        self.continuous = False
        self.printy = 0

    def initParams(self):
    	"""Initialize weights and biases, depending on if continuous data is modeled an extra weight matrix is created"""
        W1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,self.dimX))
        b1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.dimZ*self.num_model,self.HU_encoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.dimZ*self.num_model,1))

        
        W3 = np.random.normal(0,self.sigmaInit,(self.dimZ*self.num_model,self.HU_encoder))
        b3 = np.random.normal(0,self.sigmaInit,(self.dimZ*self.num_model,1))

        pi = np.random.normal(0,1.0/self.num_model,(self.num_model,1))
        
        W4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W5 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))

        b5 = np.random.normal(0,self.sigmaInit,(self.dimX,1))

        if self.continuous:
            W6 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
            b6 = np.random.normal(0,self.sigmaInit,(self.dimX,1))
            self.params = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6,pi]
        else:
	        self.params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,pi]

        self.h = [0.01] * len(self.params)
        

    def initH(self,miniBatch):
    	"""Compute the gradients and use this to initialize h"""
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]
    
    def createGradientFunctions(self):
        #Create the Theano variables
        W1,W2,W3,W4,W5,W6,x,eps = T.dmatrices("W1","W2","W3","W4","W5","W6","x","eps")

        #Create biases as cols so they can be broadcasted for minibatches
        b1,b2,b3,b4,b5,b6,pi = T.dcols("b1","b2","b3","b4","b5","b6","pi")
        
        if self.continuous:
            h_encoder = T.nnet.softplus(T.dot(W1,x) + b1)
        else:   
            h_encoder = T.tanh(T.dot(W1,x) + b1)
        print type(pi)    
        rng = T.shared_randomstreams.RandomStreams(seed=124)
        i = rng.choice(size=(1,), a=self.num_model, p=T.nnet.softmax(pi.T).T.flatten())

        mu_encoder = T.dot(W2[i[0]*self.dimZ:(1+i[0])*self.dimZ],h_encoder) + b2[i[0]*self.dimZ:(1+i[0])*self.dimZ]
        log_sigma_encoder = (0.5*(T.dot(W3[i[0]*self.dimZ:(1+i[0])*self.dimZ],h_encoder)))+ b3[i[0]*self.dimZ:(1+i[0])*self.dimZ]

        z = mu_encoder + T.exp(log_sigma_encoder)*eps
     
        
        prior = 0
        for i in range(self.num_model):
            prior += T.exp(pi[i][0])*0.5* T.sum(1 + 2*log_sigma_encoder[int(i)*self.dimZ:(1+int(i))*self.dimZ] - mu_encoder[int(i)*self.dimZ:(1+int(i))*self.dimZ]**2 - T.exp(2*log_sigma_encoder[int(i)*self.dimZ:(1+int(i))*self.dimZ]))
        prior /= T.sum(T.exp(pi))
        #Set up decoding layer
        if self.continuous:
            h_decoder = T.nnet.softplus(T.dot(W4,z) + b4)
            mu_decoder = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            log_sigma_decoder = 0.5*(T.dot(W6,h_decoder) + b6)
            logpxz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)
            gradvariables = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6,pi]
        else:
            h_decoder = T.tanh(T.dot(W4,z) + b4)
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,pi]


        logp = logpxz + prior

        #Compute all the gradients
        derivatives = T.grad(logp,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(logpxz)
        
        self.gradientfunction = th.function(gradvariables + [x,eps], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [x,eps], logp, on_unused_input='ignore')
        self.hiddenstatefunction = th.function(gradvariables + [x,eps], z, on_unused_input='ignore')
        
    def iterate(self, data):
       	"""Main method, slices data in minibatches and performs an iteration"""
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            totalGradients = self.getGradients(miniBatch.T)
            self.updateParams(totalGradients,N,miniBatch.shape[0])
    def getHiddenState(self,data):
        [N,dimX] = data.shape
        hiddens = []
        for i in range(len(data)):
            e = np.random.normal(0,1,[self.dimZ,1])
            hidden = self.hiddenstatefunction(*(self.params),x=data[i:i+1].T,eps=e)
            hiddens.append(hidden)
        return hiddens
     
    def getLowerBound(self,data):
    	"""Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            lowerbound += self.lowerboundfunction(*(self.params),x=miniBatch.T,eps=e)

        return lowerbound/N


    def getGradients(self,miniBatch):
    	"""Compute the gradients for one minibatch and check if these do not contain NaNs"""
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[1]])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[-1]

            for i in xrange(len(self.params)):
                totalGradients[i] += gradients[i]
 #       if self.printy ==0:
#            th.printing.pydotprint(self.KLfunction, outfile="./pics/symbolic_KL_opt.png", var_with_name_simple=True)
  #      self.printy =1
        return totalGradients

    def updateParams(self,totalGradients,N,current_batch_size):
    	"""Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 13):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))
