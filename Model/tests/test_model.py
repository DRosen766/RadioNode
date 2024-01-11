import torch.nn as nn
import torch.nn.functional as F

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