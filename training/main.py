from matplotlib.pyplot import step
import torch
import numpy as np
from FactEmbedder import FactEmbedder
from customLoss import customLoss
import torch.utils.data as Data
import pandas as pd


def load_data():
    csv_data = pd.read_csv('dataset/storypieces.csv')  # read file
    dataSet = []
    for _, row in csv_data.iterrows():
        rowSet = []
        corpus_sentence =row["sentence1"]
        rowSet.append(corpus_sentence)
        corpus_sentence = row["sentence2"]
        rowSet.append(corpus_sentence)
        corpus_sentence = row["sentence3"]
        rowSet.append(corpus_sentence)
        dataSet.append(rowSet)

    return dataSet

def modelTrain(indexFolder, dataSet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MINIBATCH_SIZE = 16    # batch size
    epoch_num = 10
    # put the dataset into DataLoader
    trainLoader = Data.DataLoader(
        dataset=dataSet,
        batch_size=MINIBATCH_SIZE,
        shuffle=True
    )

    net = FactEmbedder()
    custom_criterion = customLoss()
    net.to(device)
    custom_criterion.to(device)
    optomizerAdam = torch.optim.Adam(net.parameters(), lr=0.01)
    TrainLoss = []
    for i in range(epoch_num):
        for step, batch in enumerate(trainLoader):
            optomizerAdam.zero_grad()
            output = net(batch[0])
            train_loss = custom_criterion(output)
            train_loss.backward()
            running_loss = train_loss.item()  # loss accumulation
            optomizerAdam.step()
            TrainLoss.append(running_loss/len(output))
            print("-------------loss:", train_loss)
        print("epoch finish", i)
    print("finish training")
    # save training loss
    TrainLoss0 = np.array(TrainLoss)
    np.save('Trainloss_{}'.format(indexFolder), TrainLoss0)
    # save the training result
    torch.save(
        net, 'net_{}.pth'.format(indexFolder))


if __name__ == "__main__":

    dataSet = load_data()

    # choose the fold to train
    # for i in range(0,5):
    #     modelTrain(i,dataSet)
    modelTrain(2, dataSet)
