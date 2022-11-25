import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding =1)
        # linear layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2) 
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flattening the image
        x = x.view(-1, 7*7*16)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def get_model(path):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    newModel = Net()
    newModel.load_state_dict(torch.load(path), strict=False,map_location=torch.device('cpu'))
    #newModel.to(device)
    return newModel

def classify(model,image_transforms,image_path,classes):

  model = model.eval()
  image = Image.open(image_path)
  image = image_transforms(image).float()
  image = image.unsqueeze(0)

  output = model(image)
  prob = nn.functional.softmax(output,dim=1)
  _,predicted = torch.max(output.data,1)

  return ((prob[0][0].item(),prob[0][1].item()),predicted)

