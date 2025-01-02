# "Enhancing Continuous Time Series Modelling with a Latent ODE-LSTM Approach" Paper Repository
## C. Coelho, M. Fernanda P. Costa, L. L. Ferrás

### [Check the paper here!](https://www.sciencedirect.com/science/article/abs/pii/S0096300324001991)
[Also there's an early arxiv version available here.](https://arxiv.org/abs/2307.05126)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.svg" width="128"/>

### Abstract
Latent ODE and Latent ODE-RNN models are difficult to train due to the vanishing and
exploding gradients problem. To overcome this problem, the main contribution of this paper is to propose
and illustrate a new model based on a new Latent ODE using an ODE-LSTM (Long Short-Term Memory)
network as an encoder - the Latent ODE-LSTM model. To limit the growth of the gradients the Norm
Gradient Clipping strategy was embedded on the Latent ODE-LSTM model.
The performance evaluation of the new Latent ODE-LSTM (with and without Norm Gradient Clipping)
for modelling CTS with regular and irregular sampling rates is then demonstrated. Numerical experiments
show that the new Latent ODE-LSTM performs better than Latent ODE-RNNs and can avoid the vanishing
and exploding gradients during training.

### Examples

#### Toy-dataset: 
  ##### Learning spiral dynamics
  Originally available [here.](https://github.com/rtqichen/torchdiffeq)

  **Latent ODE-RNN**
  ```
  python latentODE_RNN_workingSpiral.py
  ```

  **Latent ODE-LSTM**
  ```
  python latentODE_LSTM_workingSpiral.py
  ```
  

#### Real datasets:
  ##### DJIA
  Download the dataset [here.](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
  
  ##### Climate
  Download the dataset [here.](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) 


If you found this resource useful in your research, please consider citing:

@article{coelho2024enhancing,
  title={Enhancing continuous time series modelling with a latent ODE-LSTM approach},
  author={Coelho, C and Costa, M Fernanda P and Ferr{\'a}s, Luis L},
  journal={Applied Mathematics and Computation},
  volume={475},
  pages={128727},
  year={2024},
  publisher={Elsevier}
}
