from matplotlib.pyplot import step
import torch
import numpy as np
import torch.nn.functional as F
from FullConnectedNuralNetwork import FullConnectedNuralNetwork
from customLoss import customLoss
import torch.utils.data as Data
from preprocessing import load_data


def modelTrain(indexFolder, dataSet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainDataset1 = dataSet
    trainDataset1 = torch.as_tensor(trainDataset1, dtype=torch.float32)
    if torch.cuda.is_available():
        trainDataset1 = trainDataset1.cuda()
    else:
        trainDataset1 = trainDataset1
    trainDataset = Data.TensorDataset(trainDataset1)

    MINIBATCH_SIZE = 16    # batch size
    epoch_num = 10
    # put the dataset into DataLoader
    trainLoader = Data.DataLoader(
        dataset=trainDataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True
    )

    net = FullConnectedNuralNetwork()
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

    #########################################
    # optional:  save the dataset and load it
    #########################################
    # fiveElementsDataset=np.array(dataSet)
    # np.save('fiveElementsDataset',fiveElementsDataset)
    # enc = np.load('fiveElementsDataset.npy')
    # dataSet = list(enc)

    # for i in range(0,5):
    #     modelTrain(i,dataSet)
    modelTrain(0, dataSet)
