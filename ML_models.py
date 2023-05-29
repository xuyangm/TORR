import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TwoNN(nn.Module):
    def __init__(self, in_features=784, num_hiddens=200, num_classes=10):
        super(TwoNN, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def init_model(model_name):
    if model_name == "TwoNN":
        model = TwoNN()
    elif model_name == "LeNet":
        model = LeNet()
    elif model_name == "MobileNetV3":
        model = torchvision.models.mobilenet_v3_small()
    elif model_name == "ShuffleNetV2":
        model = torchvision.models.shufflenet_v2_x0_5()

    return model
