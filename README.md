## Introduction
Fact2Vec is a pretrained embeding model used for converting a data fact (i.e., a data story piece [1]) into a vector presentation that captures the semantics of the fact and those logically related data facts will close to each other in the vector space. Details about the project is described in the following paper: 

```
@article{sun2022erato,
    title={Erato: Cooperative Data Story Editing via Fact Interpolation},
    author={Sun, Mengdi and Cai, Ligan and Cui, Weiwei and Wu, Yanqiu and Shi, Yang and Cao, Nan},
    journal = {IEEE Transactions on Visualization and Computer Graphics},
    year = {2022},
    publisher={IEEE}
}
```

### Training Corpus 
The model was trained based on 100 high-quality data stories that were manually authored by experts using the Calliope system (https://datacalliope.com). All of these stories consist of 5 data facts with diverse fact types. They were designed by following either the time-oriented narrative structure or the parallel structure. 300 fact trigrams were extracted from these stories as the training set. Each of them consisted of 3 succeeding data facts in the original story. The data is available at (_dataset/storypieces.csv_)

### Requirements and Dependencies
- Python 3.7.3
- Pytorch 1.9.0
- Transformers 4.9.1
- Sentence-transformers 2.0.0
- CUDA ≥10.2 (Optional)

### How to use the pretrained model
on GPU
```
import torch
net = torch.load('fact4vec.pth')
embedding = net(fact)
```

on CPU
```
import torch
net = torch.load('fact4vec.pth',map_location='cpu')
net = net.to(torch.device('cpu'))
torch.no_grad()
embedding = net(fact)
```

### Website
https://erato.idvxlab.com/project/

