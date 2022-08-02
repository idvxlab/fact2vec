import pandas as pd
from sentence_transformers import SentenceTransformer


def load_data():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    csv_data = pd.read_csv('dataset/trainingDataset.csv')  # read file
    dataSet = []
    for _, row in csv_data.iterrows():
        rowSet = []
        corpus_embeddings = model.encode(row["sentence1"])
        rowSet.append(corpus_embeddings)
        corpus_embeddings = model.encode(row["sentence2"])
        rowSet.append(corpus_embeddings)
        corpus_embeddings = model.encode(row["sentence3"])
        rowSet.append(corpus_embeddings)
        dataSet.append(rowSet)
        print(dataSet)

    return dataSet

