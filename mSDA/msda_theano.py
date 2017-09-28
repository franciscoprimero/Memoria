import numpy as np
from theano import function
import theano.tensor.slinalg
import theano.tensor as T

import timeit

rng = np.random

class mDA(object):
    def __init__(self, input):
        self.input = input
        
        self.W = T.dmatrix("W")
        self.Xh = T.dmatrix("Xh")
        
        self.output = T.tanh(T.dot(self.W, self.Xh))
        
    
    def get_mapping(self, pr):
        X = T.transpose(self.input)
        
        # X=[X;ones(1,size(X,2))];
        X = T.concatenate([X, T.ones((1, X.shape[1]))], axis=0)
        
        #d=size(X,1);
        d = X.shape[0]
        
        #q=[ones(d-1,1).*(1-p); 1];
        q = T.concatenate([T.ones((d-1,1)) * (1-pr), T.ones((1,1))], axis=0)
        
        #S=X*X';
        S = T.dot(X, X.T)
        
        #Q=S.*(q*q');
        Q = S * T.dot(q, q.T)
        
        #Q(1:d+1:end)=q.*diag(S);
        Q -= (T.eye(Q.shape[0]) * Q.diagonal())
        Q += T.eye(Q.shape[0]) * T.diagonal(q*S.diagonal())
        
        #P=S.*repmat(q',d,1);
        P = S * T.extra_ops.repeat(q.T, d, 0)
        
        #W=P(1:end-1,:)/(Q+1e-5*eye(d));
        
        A = Q + 10**-5*T.eye(d)
        B = P
        
        self.W = T.slinalg.solve(A.T,B.T)[:-1,:]
        self.Xh = T.tanh(T.dot(self.W, X)).T
        
        return self.W, self.Xh
    
    def predict(self, X, W):
        X = T.concatenate([X.T, T.ones((1, X.T.shape[1]))], axis=0)
        #return T.tanh(T.dot(mapping, X))
        pred = T.tanh(T.dot(W, X))
        return pred.T


class mSDATheano(object):
    def __init__(self, input, l, pr):
        self.input = input
        self.l = l
        self.pr = pr
        
        self.mDA_layers = []
        self.W_layers = []
        
        for i in xrange(self.l):
            if i==0:
                layer_input = self.input
            else:
                layer_input = self.mDA_layers[-1].output
        
            mda = mDA(layer_input)
            
            self.mDA_layers.append(mda)
    
    def train_fns(self):
        fns = []
        for mda in self.mDA_layers:
            W, Xh = mda.get_mapping(self.pr)
            train_mda = function(
                [mda.input],
                [W, Xh],
                allow_input_downcast = True
            )
            
            fns.append(train_mda)
        
        return fns
    
    def fit(self, X):
        training_fns = self.train_fns()
        
        start_time = timeit.default_timer()

        inputs = [X]

        i = 1
        for fn in training_fns:
            print "\tEntrenando capa ", i
            W, Xh = fn(inputs[-1])

            inputs.append(Xh)

            self.W_layers.append(W)

            i += 1

        end_time = timeit.default_timer()

        tiempo = (end_time - start_time)/60.
        print "\tEntrenado en %.2fm\n" % (tiempo)

        return tiempo
    
    def predict(self, X):
        Xhs = [X]
        
        for i in xrange(self.l):
            mda = self.mDA_layers[i]
            W = self.W_layers[i]

            x2 = T.dmatrix('x')
            W2 = T.dmatrix('w')

            y = mda.predict(x2, W2)
            
            
            pred = function(
                [x2, W2],
                y,
                allow_input_downcast = True
            )
            
            Xh = pred(Xhs[i], W)
            Xhs.append(Xh)
        
        return Xhs[-1]
