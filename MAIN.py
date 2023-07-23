import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import functional
from torch import nn
from torch import optim
from torchvision.utils import make_grid
import numpy
from matplotlib import pyplot
import CNN

#Working in GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#to transform the imape into tensor and normalize the image
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


#importing the training and testing dataset
train_data = CIFAR10(root="cifar",train=True,transform=transform,download=True)
test_data = CIFAR10(root="cifar",train=False,transform=transform,download=True)


'''
classes = train_data.classes
print(classes)
'''


#importing the dataloader
batch_size = 32
train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=0)


'''
for x,y in train_dataloader:
    print("shape of X [N,C,H,W] : {X}".format(X=x.shape))
    print("shape of Y : {y_shape} {y_dtype}".format(y_shape=y.shape,y_dtype=y.dtype))
    break
'''


#Importion the convolution neural network
Brain = CNN.convolution_neural_network().to(device)

#Training the network

#defining the loss
criterian = nn.CrossEntropyLoss()

#defining the optimizer
optimizer = optim.Adam(Brain.parameters(),lr=0.00001)

#assigning the number of epoch
epoch = 500

def Training_model(epochs,criterian,optimizer,dataloader,device,Brain):
    for epoch in range(1,epochs+1):
        total_loss = 0
        for batch,(X,Y) in enumerate(dataloader):
            X,Y = X.to(device),Y.to(device)
            optimizer.zero_grad()
            output = Brain.Forward(X).to(device)
            loss = criterian(output,Y)

            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss

        else:
            value = total_loss/len(dataloader)
            print("EPOCH NO : {EPOCHS} TRAINING LOSS : {VALUE}".format(EPOCHS=epoch,VALUE=value))

print("******TRAINING THE DATA******")
Training_model(epochs=epoch,criterian=criterian,optimizer=optimizer,dataloader=train_dataloader,device=device,Brain=Brain)

#Testing the neural network

def Testing_model(epochs,dataloader,criterian,device,Brain):
    Brain.eval()
    test_loss = 0
    correct = 0
    for epoch in range(1,epochs+1):
        with torch.no_grad():
            for X,Y in dataloader:
                X,Y = X.to(device),Y.to(device)
                prediction = Brain.Forward(X).to(device)
                test_loss = test_loss + criterian(prediction,Y).item()
                val = (prediction.argmax(1)==Y).type(torch.float)
                correct = correct + val.sum().item()
            test_loss = test_loss/len(dataloader)
            correct = correct/len(dataloader.dataset)
    else:
        accuracy = 100*correct
        print("TEST LOSS : ACCURACY: {ACCURACY} AVERAGE LOSS : {AVERAGE_LOSS}".format(EPOCHS=epoch,ACCURACY=accuracy,AVERAGE_LOSS=test_loss))

print("******TESTING THE DATA******")
Testing_model(epochs=epoch,dataloader=test_dataloader,criterian=criterian,device=device,Brain=Brain)


#Sample prediction

def imshow(image):
    image = image/2 + 0.05
    np_image = image.numpy()
    pyplot.imshow(numpy.transpose(np_image,(1,2,0)))
    pyplot.show()

test_pred_data = CIFAR10(root="cifar",train=False,transform=transforms.ToTensor())
classes = test_pred_data.classes

def sample_prediction(n,Brain,device):
    test_pred_dataloader = DataLoader(test_pred_data,batch_size=n,shuffle=True,num_workers=0)
    data_iter = iter(test_pred_dataloader)
    image,label = next(data_iter)
    output = Brain.Forward(image.to(device))
    _,prediction = torch.max(output,1)

    imshow(make_grid(image))
    print('[Ground Truth | Predicted]:\n',' '.join(f'[{classes[label[j]]:5s} | {classes[prediction[j]]:5s}]\n' for j in range(n)))


sample_prediction(25,Brain=Brain,device=device)
