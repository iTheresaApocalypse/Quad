#  Quad: Harnessing Diversity for Important Data Selection in Pretraining Large Language Models

## Overview

- **Cluster Score**
![dc11eac126fba0e8fb59244f60e9a7a](https://github.com/user-attachments/assets/cf26ec21-fe17-4dc7-a161-dc4c863301b3)

- **K-FAC for attention layers**
![0d487fe3f29a0cabde565969e22afb8](https://github.com/user-attachments/assets/b2e00dbe-e771-42e0-872b-b3bac0ccb412)


## Environment Configuration

The python core library files required for Quad are as follows:

- torch                    2.4.0+cu121
- torchaudio               2.4.0
- torchvision              0.19.0
- nvcc                     11.7
- trak                     0.3.2

##  Kindly Regard: How to install TRAK?

First, make sure you have installed all the python core library files except TRAK.

> **Note**
>
> Version required:
> nvcc >= 10.0.
> If you have CUDA 11, you will need gcc with version 7.5 <= version <= 10. For CUDA 12, you will need gcc with version >= 7.5.

The PyTorch-only version of our package can be installed using

```python
pip install traker
```

To install the version of our package which contains a fast, custom CUDA kernel, use

```python
pip install traker[fast]
```

If you find traker hasn't been successfully installed, like

```bash
Quad: no matches found: traker[fast]
```

Make sure to escape the square brackets, i.e. run

```python
pip install traker\[fast\]
```

You can test your installation by running the following in a python shell:

```python
import trak
trak.test_install(use_fast_jl=False)
trak.test_install(use_fast_jl=True)  # if you're using the fast version
```

In addition, you may encounter other difficulties in installing TRAK, please refer to the official TRAK configuration website:
[https://trak.readthedocs.io/en/latest/install.html](https://trak.readthedocs.io/en/latest/install.html)

## Data Preparation

We use the entire 627B-token  [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B)  dataset as the candidate pool . In the clustering process, the [BAAI/bge-large-en-v1.5]( https://huggingface.co/BAAI/bge-large-en-v1.5 ) model is employed to generate embeddings for the input data, and approximately 600 million data points from the candidate pool are clustered into 10,000 groups using the k-means algorithm. We use [LAMBADA](https://huggingface.co/datasets/cimec/lambada) as our reference set, which is a widely used language modeling task and often serves as a validation benchmark for language model pre-training.

## Data Selection Pipeline

### Step1:

### Step2:

## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!
