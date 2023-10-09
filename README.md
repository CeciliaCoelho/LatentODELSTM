# "Enhancing Continuous Time Series Modelling with a Latent ODE-LSTM Approach" Paper Repository
## C. Coelho, M. Fernanda P. Costa, L. L. Ferrás

### [Check the paper here!](https://arxiv.org/abs/2307.05126)

Latent ODE and Latent ODE-RNN models are difficult to train due to the vanishing and
exploding gradients problem. To overcome this problem, the main contribution of this paper is to propose
and illustrate a new model based on a new Latent ODE using an ODE-LSTM (Long Short-Term Memory)
network as an encoder - the Latent ODE-LSTM model. To limit the growth of the gradients the Norm
Gradient Clipping strategy was embedded on the Latent ODE-LSTM model.
The performance evaluation of the new Latent ODE-LSTM (with and without Norm Gradient Clipping)
for modelling CTS with regular and irregular sampling rates is then demonstrated. Numerical experiments
show that the new Latent ODE-LSTM performs better than Latent ODE-RNNs and can avoid the vanishing
and exploding gradients during training.

