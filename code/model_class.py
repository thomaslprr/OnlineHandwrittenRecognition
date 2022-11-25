import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn.functional as F
import pickle
import io

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.conv2 = nn.Conv2d(128, 256, 5)
        self.conv3 = nn.Conv2d(256,512,3)
        # linear layers
        self.fc1 = nn.Linear(4608, 2304)
        self.fc2 = nn.Linear(2304, 101)

        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # flattening the image
        x = x.view(-1, 4608)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)
        return x

def get_model(path):
    newModel = Net()

    if torch.cuda.is_available():
        newModel.load_state_dict(torch.load(path), strict=False)
    else:
        with open(path, 'rb') as f:
            contents = f.read()
            content = torch.load(io.BytesIO(contents), map_location='cpu')
            newModel.load_state_dict(content, strict=False)
    return newModel

def classify(model,image_transforms,im,classes):

  model = model.eval()
  image = Image.fromarray(im.astype('uint8'), 'L')
  image = image_transforms(image).float()
  image = image.unsqueeze(0)

  output = model(image)
  prob = nn.functional.softmax(output,dim=1)
  _,predicted = torch.max(output.data,1)

  return ((prob[0][0].item(),prob[0][1].item()),predicted)