# Fact4Vec

## Introduction
Fact4Vec employs a deep embedding model to convert data facts into vector representations. It is achieved by adding two fully connected layers on top of BERT and fine-tuned based on a set of manually designed visual narratives.

## Requirements
- Python 3.7.3
- Pytorch 1.9.0
- Transformers 4.9.1
- Sentence-transformers 2.0.0

**Note**: you may need to install CUDA on your own server to make the training go fast.

## Dataset
The dataset contains 300 triples (_dataset/trainingDataset.csv_), each training value is a trigram of the facts. The input format of a data fact is as follows:
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


It is fine-tuned with its own loss function:

![image](https://github.com/idvxlab/fact4vec/blob/master/training/loss_function.png)

Go to folder _training_, execute the following command to start training model. It will save the training result in the folder.
```
python main.py
```

## How to use
**First, install pytorch and import it**
```
import torch
```
**Then, load the model**
```
net = torch.load('fact4vec.pth')
```
**Convert the data fact into vector**
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
[Project Page](https://erato.idvxlab.com/Project/)
```
Mengdi Sun, Ligan Cai, Weiwei Cui, Yanqiu Wu, Yang Shi, and Nan Cao. Erato: Cooperative Data Story Editing via Fact Interpolation. IEEE Trans. Visualization & Comp. Graphics, 2022
```
