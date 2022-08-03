# Fact2Vec

## Introduction
Fact2Vec is a pretrained embeding model used for converting data facts into vector presentations that capture the facts' semantic similarity. A data fact is a story piece that is defined in the following paper: 
```
Danqing Shi, Xinyue Xu, Fuling Sun, Yang Shi, Nan Cao:
Calliope: Automatic Visual Data Story Generation from a Spreadsheet. IEEE Trans. Vis. Comput. Graph. 27(2): 453-463 (2021)
```

Fact2Vec is trained based on a set of manually designed visual data stories by fine tuning BERT using the following loss:

<img src="https://github.com/idvxlab/fact4vec/raw/master/training/loss_function.png" alt="loss" style="width: 270px">

## Website
[Project Page](https://erato.idvxlab.com/project/)

## Training Corpus 
We selected 100 high-quality data stories that were manually authored based on different datasets using the Calliope system (https://datacalliope.com). All of these stories consist of 5 data facts with diverse fact types. They were designed by following either the time-oriented narrative structure or the parallel structure. 300 fact trigrams were extracted from these stories as our training set. Each of them consisted of 3 succeeding data facts in the original story. The data is available at (_dataset/storypieces.csv_)

## Requirements
- Python 3.7.3
- Pytorch 1.9.0
- Transformers 4.9.1
- Sentence-transformers 2.0.0
- CUDA ≥10.2 (Optional)


## How to use
**1. Import pytorch**
```
import torch
```
**2. Load the pretrained model**
```
net = torch.load('fact4vec.pth')
```
**3. Convert the data fact into vector**
```
embedding = net(fact)
```

It also runs on cpu:
```
net = torch.load('fact4vec.pth',map_location='cpu')
net = net.to(torch.device('cpu'))
torch.no_grad()
embedding = net(fact)
```

## Reference
```
@article{sun2022erato,
      title={Erato: Cooperative Data Story Editing via Fact Interpolation},
      author={Sun, Mengdi and Cai, Ligan and Cui, Weiwei and Wu, Yanqiu and Shi, Yang and Cao, Nan},
      journal = {IEEE Transactions on Visualization and Computer Graphics},
      year = {2022},
      publisher={IEEE}
    }
```
