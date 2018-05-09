import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
#import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 4x4 square convolution kernel
        # output size = (W-F)/S +1 = (96-4)/1 +1 = 93
        # the output Tensor for one image, will have the dimensions: (32, 93, 93)
        # after one pool layer, this becomes (32, 46, 46)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 46, 3)
        self.conv3 = nn.Conv2d(46, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fcl1 = nn.Linear(6400, 1000)
        self.fcl2 = nn.Linear(1000, 500)
        self.fcl3 = nn.Linear(500, 136)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        ## Implementation of NaimishNet as in https://arxiv.org/pdf/1710.00977.pdf
        
        x = self.conv1(x)           # 2 - Convolution2d1
        x = F.elu(x)                # 3 - Activation1
        x = self.pool(x)            # 4 - Maxpooling2d1     
        x = self.dropout1(x)        # 5 - Dropout1      
        x = self.conv2(x)           # 6 - Convolution2d2
        x = F.elu(x)                # 7 - Activation2
        x = self.pool(x)            # 8 - Maxpooling2d2
        x = self.dropout2(x)        # 9 - Dropout2
        x = self.conv3(x)           # 10 - Convolution2d3
        x = F.elu(x)                # 11 - Activation3
        x = self.pool(x)            # 12 - Maxpooling2d3
        x = self.dropout3(x)        # 13 - Dropout3
        x = self.conv4(x)           # 14 - Convolution2d4
        x = F.elu(x)                # 15 - Activation4
        x = self.pool(x)            # 16 - Maxpooling2d4
        x = self.dropout4(x)        # 17 - Dropout4
        x = x.view(x.size(0), -1)   # 18 - Flatten1
        x = self.fcl1(x)            # 19 - Dense1
        x = F.elu(x)                # 20 - Activation5
        x = self.dropout5(x)        # 21 - Dropout5
        x = self.fcl2(x)            # 22 - Dense2
        x = F.elu(x)                # 23 - Activation6
        x = self.dropout6(x)        # 24 - Dropout6
        x = self.fcl3(x)            # 25 - Dense3 

        # a modified x, having gone through all the layers of your model, should be returned
        return x
