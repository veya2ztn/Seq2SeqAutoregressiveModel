# A sequence to sequence autoregressive project 

This repository aims to provide an efficient, user-friendly, and comprehensive framework for sequence-to-sequence machine learning applications, including but not limited to weather forecasting and time series analysis.

Our primary objective is to address the challenges associated with training large-scale machine learning models on extensive time-series datasets. For instance, training a 70B ViT model on the 721x1440 full-resolution ERA5 dataset (40T = 3G x 100k).

## Key Features

### 1. Parallel Training for Large Models

We offer robust support for large model parallel training, with different modules at various testing stages. These include:

- PyTorch-DDP (Tested)
- Huggingface Accelerate-DDP (Tested)
- Huggingface Accelerate-Deepspeed (Testing Required)
- Huggingface Accelerate-FSDP (Testing Required)
- Huggingface Accelerate-Magetron (Testing Required)

### 2. Efficient Data Loading for Large Datasets

This framework supports efficient data loading for large datasets through:

- In-Memory Data Loading (Tested)
  - Runtime Loading (Tested): Saves time during program initiation
  - Shared Dataset (Tested): Conserves memory in distributed mode
- Server Memory Data Loading: Boosted via [Ray](https://www.ray.io/)
  - Shared Dataset (TODO)
  - Asynchronous updating of in-memory datasets (TODO)
  - Runtime updating of datasets and common buffer for efficient data sampling and infinite data size (TODO)

### 3. Efficient Autoregressive Forward Mode

- High-order autoregressive computing. (Tested)

  $$
  \begin{align}
  	\left(
  	\begin{array}{cccccc}
  			X_{t}^O & X_{t+1}^I & X_{t+2}^{\text{II}} & X_{t+3}^{\text{III}} & X_{t+4}^{\text{IV}} & X_{t+5}^V            \\
  			        & X_{t+1}^O & X_{t+2}^I           & X_{t+3}^{\text{II}}  & X_{t+4}^{\text{IV}} & X_{t+5}^{\text{IV}}  \\
  			        &           & X_{t+2}^O           & X_{t+3}^I            & X_{t+4}^{\text{II}} & X_{t+5}^{\text{III}} \\
  			        &           &                     & X_{t+3}^O            & X_{t+4}^I           & X_{t+5}^{\text{II}}  \\
  			        &           &                     &                      & X_{t+4}^O           & X_{t+5}^I            \\
  			        &           &                     &                      &                     & X_{t+5}^O            \\
  		\end{array}
  	\right)
  \end{align}
  $$

- Patch wise and overlap aggregation back.  (Tested)

### 4. Autoregressive Plugins for Enhanced Performance

We offer various autoregressive plugins/tricks for training boost and performance enhancement such as:

- Pseudo-future alignment via high order loss (MLSE, MASE, and any order)  (Tested)
- Jacobian Regularization to limit:
  - Naive computing and gradient modification (Testing Required)
  - Stochastic backward via Hutchinson Method (Testing Required)

### 5. Various Large Models in PyTorch

Our framework supports various large models in PyTorch including:

- [FourCastNet](https://github.com/NVlabs/FourCastNet) (Tested)
- [GraphCast](https://github.com/google-deepmind/graphcast) (Tested)
- ViT Model Series (Tested)
- Physics-Constrained Models: Convection Model (Testing Required)
- On-stamp Prediction (Testing Required)
- And much more...

This repository is continually evolving, with new features and models being added. We welcome contributions, suggestions, and feedback to enhance its functionality and user experience.
