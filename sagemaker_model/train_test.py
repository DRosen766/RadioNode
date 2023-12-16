from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
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
# mp.set_start_method('fork', force=True )

# transforms
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))])


# batch_size

metadata_file = open("data/metadata.csv")
reader = csv.reader(metadata_file)
reader = list(reader)

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

results_file_name = 'results_info.csv'
fid = open(results_file_name, 'a', encoding='UTF8', newline='')
writer = csv.writer(fid)
    # datasets
visualization_interval = 10
example_num = 0
skip_num = 0
train_num = 1000
test_num = 200
total_num = train_num + test_num
training_examples = []
training_labels = []
testing_examples = []
testing_snr = []
testing_labels = []

def load_data(num_threads, thread_num, test_data=True, reader=reader):
    reader = reader[int((total_num / num_threads) * thread_num) : int((total_num / num_threads) * thread_num + (total_num / num_threads) )]
    for line in tqdm(reader):
        data_file_name, snr, label = line[0], line[-2], line[-1]
        iq_data = np.fromfile(data_file_name)
        iq_data = np.vstack(([[iq_data[:1024]]], [[iq_data[1024:]]]))
        if test_data:
            training_examples.append(iq_data)
            training_labels.append(label_map[label])
        else:
            testing_examples.append(iq_data)
            testing_snr.append(float(snr))
            testing_labels.append(label_map[label])

threads = [Thread(target=load_data, args=[6,i, True if i < 5 else False],name="thread{}".format(i)) for i in range(6)]

for thread_run in threads:
    thread_run.start()
for thread_join in threads:
    thread_join.join()

iteration = 0
while True and iteration < 5:
    batch_size_train = [64,32, 128, 16]
    batch_size_test = [16,16, 16, 16]
    learning_rate = [0.001,0.00075, 0.00075, 0.0005]
    weight_decay = [1e-3, 1e-3, 1e-3, 1e-3]
    epochs = [10] * 5
    

    training_examples = Tensor(np.array(training_examples))
    training_labels = Tensor(np.array(training_labels))
    train_ds = TensorDataset(training_examples, training_labels)
    trainloader = DataLoader(train_ds,batch_size=int(batch_size_train[iteration]),
    shuffle=True, num_workers=0)


    testing_examples = Tensor(np.array(testing_examples))
    testing_snr = Tensor(np.array(testing_snr))
    testing_labels = Tensor(np.array(testing_labels))
    test_ds = TensorDataset(testing_examples, testing_snr, testing_labels)
    testloader = DataLoader(test_ds,batch_size=int(batch_size_test[iteration]),
    shuffle=True, num_workers=0)

    # testloader = trainloader
    # dataloaders


    classes = ["2-ASK", "4-ASK", "8-ASK", "BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "Constant Tone", "P-FMCW", "N-FMCW"]


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(2, 16, (1,4), 1)
            self.conv2 = nn.Conv2d(16, 32, (1,4), 1)
            self.conv3 = nn.Conv2d(32, 48, (1,4), 1)
            self.conv4 = nn.Conv2d(48, 96, (1,4), 1)
            self.conv_drop = nn.Dropout2d()
            self.linear1 = nn.Linear(1344, 560)
            self.linear2 = nn.Linear(560, 240)
            self.linear3 = nn.Linear(240, 11)
            self.norm = nn.LayerNorm(11)
            self.flat = nn.Flatten()


        def forward(self, x, input_batch_size):
            # print("shape 0:", x.shape)
            x = F.relu(F.max_pool2d(self.conv1(x), (1,2)))
            # print("shape 1:", x.shape)
            x = F.relu(F.max_pool2d(self.conv2(x), (1,2)))
            x = F.relu(F.max_pool2d(self.conv3(x), (1,4)))
            # print("shape 2:", x.shape)
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv4(x)),(1,4)))
            # print("shape 3:", x.shape)
            x = x.view(-1 , 1344)
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.dropout(x, training=self.training)
            x = self.linear3(x)
            return F.log_softmax(x)

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
            # if loss.item() <= 1:
            #     print(list(label_map.keys())[np.argmax(outputs.detach().numpy())], list(label_map.keys())[int(target.data.cpu().numpy())], loss.item())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(range(len(losses)), losses)
        plt.savefig("output_plot_{}.png".format(iteration))
        plt.close()


    y_pred = []
    y_true = []


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

