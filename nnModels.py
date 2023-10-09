import argparse
import numpy as np


import torch
import torch.nn as nn

#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint


# RNN implementation --------------------------------
class RNN(nn.Module):

    def __init__(self, obs_dim, nhidden, out_dim, nbatch=1):
        super(RNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        combined = torch.cat((x, h))
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)  
        out = out #softmax so we get an output tensor with language probabilities
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)


# GRU implementation --------------------------------
class GRU(nn.Module):

    def __init__(self, obs_dim, nhidden, out_dim, nbatch=1):
        super(GRU, self).__init__()
        self.nhidden = nhidden

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

        

    def forward(self, x, h):
        combined = torch.cat((x, h))
        z = torch.sigmoid(self.i2h(combined))
        r = torch.sigmoid(self.i2h(combined))
        combined2 = torch.cat((x, r*h))
        h_hat = torch.tanh(self.i2h(combined))
        h = z * h + (1 - z) * h_hat

        out = self.h2o(h)
        
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)



# LSTM implementation --------------------------------
class LSTM(nn.Module):

    def __init__(self, obs_dim, nhidden, out_dim, nbatch=1):
        super(LSTM, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x, h, c):
        combined = torch.cat((x, h))

        c_tilde = torch.sigmoid(self.i2h(combined))
        i = torch.sigmoid(self.i2h(combined))
        f = torch.sigmoid(self.i2h(combined))
        o = torch.sigmoid(self.i2h(combined))

        c = f * c + i * c_tilde 

        h = o * torch.tanh(c)

        out = self.h2o(h)
        
        return out, h, c

    def initHidden(self):
        return torch.zeros(self.nhidden)


# ODE FUNC implementation --------------------------------
class LatentODEfunc(nn.Module):

    def __init__(self, obs_dim, nhidden, latent_dim=0):
        super(LatentODEfunc, self).__init__()
        if latent_dim == 0: latent_dim = nhidden 
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, obs_dim+nhidden)
        self.fc2 = nn.Linear(obs_dim+nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out



# ODE-RNN implementation --------------------------------
class ODERNN(nn.Module):

    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(ODERNN, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

    def forward(self, x, h, t):
        h = odeint(self.func, h, t)[-1]
        combined = torch.cat((x, h))
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)  
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)


# ODE-GRU implementation --------------------------------
class ODEGRU(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(ODEGRU, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

        

    def forward(self, x, h, t):
        h = odeint(self.func, h, t)[-1]
        combined = torch.cat((x, h))
        z = torch.sigmoid(self.i2h(combined))
        r = torch.sigmoid(self.i2h(combined))
        combined2 = torch.cat((x, r*h))
        h_hat = torch.tanh(self.i2h(combined))
        h = z * h + (1 - z) * h_hat

        out = self.h2o(h)
        
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)


# ODE-LSTM implementation --------------------------------
class ODELSTM(nn.Module):
    def __init__(self, func, obs_dim, nhidden, out_dim, nbatch=1):
        super(ODELSTM, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

        

    def forward(self, x, h, c, t):
        combined = torch.cat((x, h))

        h = odeint(self.func, h, t)[-1]
        combined2 = torch.cat((x, h))

        
        c_tilde = torch.sigmoid(self.i2h(combined2))
        i = torch.sigmoid(self.i2h(combined))
        f = torch.sigmoid(self.i2h(combined))
        o = torch.sigmoid(self.i2h(combined))

        c = f * c + i * c_tilde 

        h = o * torch.tanh(c)

        out = self.h2o(h)
        
        return out, h, c

    def initHidden(self):
        return torch.zeros(self.nhidden)


# ODE-RNN implementation --------------------------------
class RecognitionODERNN(nn.Module):

    def __init__(self, func, latent_dim, obs_dim, nhidden, nbatch=1):
        super(RecognitionODERNN, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h, t):
        h = odeint(self.func, h, t)[-1]

        x = x.squeeze(0)

        combined = torch.cat((x, h))
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)

# ODE-GRU implementation --------------------------------
class RecognitionODEGRU(nn.Module):

    def __init__(self, func, latent_dim, obs_dim, nhidden, nbatch=1):
        super(RecognitionODEGRU, self).__init__()
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h, t):
        h = odeint(self.func, h, t)[-1]

        x = x.squeeze(0)

        combined = torch.cat((x, h))
        z = torch.sigmoid(self.i2h(combined))
        r = torch.sigmoid(self.i2h(combined))
        combined2 = torch.cat((x, r*h))
        h_hat = torch.tanh(self.i2h(combined))
        h = z * h + (1 - z) * h_hat

        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nhidden)


# VAE ODE-LSTM encoder implementation --------------------------------
class RecognitionODELSTM(nn.Module):

    def __init__(self, func, latent_dim, obs_dim, nhidden, nbatch=1):
        super(RecognitionODELSTM, self).__init__()
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)


    def forward(self, x, h, c, t):
        h = odeint(self.func, h, t)[-1]

        x = x.squeeze(0)

        combined = torch.cat((x, h))

        c_tilde = torch.sigmoid(self.i2h(combined))
        i = torch.sigmoid(self.i2h(combined))
        f = torch.sigmoid(self.i2h(combined))
        o = torch.sigmoid(self.i2h(combined))

        c = f * c + i * c_tilde 

        h = o * torch.tanh(c)

        out = self.h2o(h)
        
        return out, h, c

    def initHidden(self):
        return torch.zeros(self.nhidden)


# ODE VAE decoder implementation --------------------------------
class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out








def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float()
    const = torch.log(const).cuda()

    x = x.squeeze(0).cuda()

    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1):
    v1 = torch.exp(lv1)
    lstd1 = lv1 / 2.

    kl = - lstd1 + ((v1 + (mu1) ** 2.) / 2.) - .5
    return kl

