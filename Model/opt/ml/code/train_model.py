# test that model trains properly with example data
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from threading import Thread
from  model import Net
import boto3
import json
from os.path import normpath, join, realpath
import os
# mp.set_start_method('fork', force=True )

# transforms
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))])


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == "__main__":
    label_map = {"['2-ASK', ['ask', 2]]" : 0,
                "['4-ASK', ['ask', 4]]" : 1,
                "['8-ASK', ['ask', 8]]" : 2,
                "['BPSK', ['psk', 2]]" : 3, 
                "['QPSK', ['psk', 4]]" : 4,
                "['8-PSK', ['psk', 8]]" : 5,
                "['16-QAM', ['qam', 16]]" : 6,
                "['64-QAM', ['qam', 64]]" : 7,
                "['Constant Tone', ['constant']]" : 8,
                "['P-FMCW', ['p_fmcw']]" : 9,
                "['N-FMCW', ['n_fmcw']]" : 10}

    # access /opt/ml/directory
    true_path = lambda path : realpath(join("..", normpath(path)))

    # connect to bucket
    bucket = boto3.resource("s3").Bucket(os.environ["SM_TRAINING_DATA_BUCKET"])
    bucket.download_file("label_map.json", true_path("label_map.json"))
    label_map = json.load(open(true_path("label_map.json")))
    bucket.download_file("train/train_metadata.csv", true_path("train_data/train_metadata.csv"))
    metadata_file = open(true_path("train_data/train_metadata.csv"))
    reader = csv.reader(metadata_file)
    reader = list(reader)



    # datasets
    train_num = 100
    training_examples = []
    training_labels = []


    def load_data(num_threads, thread_num, test_data=True, reader=reader):
        reader = reader[int((train_num / num_threads) * thread_num) : int((train_num / num_threads) * thread_num + (train_num / num_threads) )]
        for line in tqdm(reader):
            data_file_name, snr, label = line[0], line[-2], line[-1]
            # download from file and load to array
            bucket.download_file(f"train/iqdata/{data_file_name}", true_path(f"train_data/iqdata/{data_file_name}"))
            iq_data = np.fromfile(true_path(f"train_data/iqdata/{data_file_name}"))
            # stack array
            iq_data = np.vstack(([[iq_data[:1024]]], [[iq_data[1024:]]]))
            # add to dataset
            training_examples.append(iq_data)
            training_labels.append(label_map[label])

    threads = [Thread(target=load_data, args=[6,i, True if i < 5 else False],name="thread{}".format(i)) for i in range(6)]

    for thread_run in threads:
        thread_run.start()
    for thread_join in threads:
        thread_join.join()

    iteration = 0
    while True and iteration < 1:
        batch_size_train = [64]
        batch_size_test = [16]
        learning_rate = [0.001]
        weight_decay = [1e-3]
        epochs = [2]
        

        training_examples = Tensor(np.array(training_examples))
        training_labels = Tensor(np.array(training_labels))
        train_ds = TensorDataset(training_examples, training_labels)
        trainloader = DataLoader(train_ds,batch_size=int(batch_size_train[iteration]),
        shuffle=True, num_workers=0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        net = Net()
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=float(learning_rate[iteration]), weight_decay=float(weight_decay[iteration]))
        accuracy = 0.0
        losses = []

        for epoch in range(int(epochs[iteration])): # loop over the dataset multiple times
            print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
            for batch_idx, (input, target) in tqdm(enumerate(trainloader)):
                if device == 'cuda':
                    input, target = input.cuda(), target.cuda()
                optimizer.zero_grad()
                outputs = net(input, input.shape[0])
                loss = F.nll_loss(outputs, target.long())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            plt.plot(range(len(losses)), losses)
            # plt.savefig("output_plot_{}.png".format(iteration))
            plt.close()

        torch.save(net.state_dict(), true_path("saved_model"))

    exit()

    # iterate over test data
    for inputs, snr, label in tqdm(testloader):
    # for inputs, label in tqdm(testloader):
        if device == 'cuda':
            inputs, label = input.cuda(), label.cuda()
        output = net(inputs, inputs.shape[0]) # Feed Network
        for batch in (output.cpu().detach().numpy() if device == 'cuda' else output.detach().numpy()):
            y_pred.append(int(np.argmax(batch)))

        label = label.data.cpu().numpy()
        for i in label:
            y_true.append(int(i))

    accuracy = np.sum([i == j for i, j in zip(y_pred, y_true)]) / len(y_pred)
    writer.writerow([train_num, batch_size_train[iteration], batch_size_test[iteration], learning_rate[iteration], weight_decay[iteration], epochs[iteration], accuracy])
    print("accuracy: ", accuracy)

    # Build confusion matrix
    cf_matrix = confusion_matrix([list(label_map.keys())[i] for i in y_true], [list(label_map.keys())[i] for i in y_pred], labels=list(label_map.keys()))
    df_cm = pd.DataFrame(cf_matrix / (np.sum(cf_matrix, axis=1)), index = [i for i in list(label_map.keys())], columns = [i for i in list(label_map.keys())])               
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output_{}.png'.format(iteration))
    plt.close()
    iteration+=1

