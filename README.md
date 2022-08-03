# Fact2Vec

## Introduction
Fact2Vec employs a deep embedding model to convert data facts (pieces in a data story) into vector representations. It is achieved by adding two fully connected layers on top of BERT and fine-tuned based on a set of manually designed visual data stories.

## Requirements
- Python 3.7.3
- Pytorch 1.9.0
- Transformers 4.9.1
- Sentence-transformers 2.0.0

**Note**: CUDA (≥10.2) is required.

## Dataset
The dataset contains 300 triples (_dataset/storypieces.csv_), each training value is a trigram of the facts. The input format of a data fact is as follows:
```
[type]fact-type [subspace]field,value [measure]field,agg[breakdown]field [focus]value [meta]extra-info
```
## Data Preprocessing
Install the [pre-trained model](https://www.sbert.net/) to initialize the data fact into vectors (_training/preprocessing.py_).
```
pip install sentence-transformers
```

## Fine-Tuning
Train the model with the prepared dataset, using two fully connected layers on top of BERT with 768 inputs and 768 outputs. 

|Model Parameter|Value|
|-------------|-------------------|
|Batch|16|
|Epoch|10|
|Learning rate|0.01|


The model is fine-tuned with its own loss function:

<img src="https://github.com/idvxlab/fact4vec/raw/master/training/loss_function.png" alt="loss" style="width: 270px">

Go to folder _training_, execute the following command to start training model. It will save the training result in the folder.
```
python main.py
```

## How to use
**1. Install pytorch and import it**
```
import torch
```
**2. Load the model**
```
net = torch.load('fact4vec.pth')
```
**3. Convert the data fact into vector**
```
factEmbedding = net(fact)
```

If only cpu on your server, try:
```
net = torch.load('fact4vec.pth',map_location='cpu')
net = net.to(torch.device('cpu'))
torch.no_grad()
factEmbedding = net(fact)
```

## Reference
[Project Page](https://erato.idvxlab.com/project/)
```
@article{sun2022erato,
      title={Erato: Cooperative Data Story Editing via Fact Interpolation},
      author={Sun, Mengdi and Cai, Ligan and Cui, Weiwei and Wu, Yanqiu and Shi, Yang and Cao, Nan},
      journal = {IEEE Transactions on Visualization and Computer Graphics},
      year = {2022},
      publisher={IEEE}
    }
```
