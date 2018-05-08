import torch
import torch.nn as nn
import torch.optim as optim
from models import Net
from model_train import net_sample_output, train_net
from model_evaluate import visualize_output

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

net = Net().to(device)

# define loss and optimisation functions
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# train the network
n_epochs = 100
train_net(net, device, criterion, optimizer, n_epochs)

# get a sample of test data
test_images, test_outputs, gt_pts = net_sample_output(net, device)

visualize_output(test_images, test_outputs, gt_pts)

# after training, save the model parameters in the dir 'saved_models'
torch.save(net.state_dict(), 'saved_models/keypoints_model.pt')
