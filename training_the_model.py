import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchinfo import summary
import os


transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Load datasets
'''if you want to use this, need to replace the directory of train and test folder'''
train_dataset=ImageFolder(root=r'D:\code\alpaca_clf\train', transform=transform) 
val_dataset=ImageFolder(root=r'D:\code\alpaca_clf\test', transform=transform)

# Define dataloaders
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load EfficientNet-B3 model
'''i have test with mobilenetv2 model, the accuracy is similar'''
weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT  
model=torchvision.models.efficientnet_b3(weights=weights)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Modify the lass layer of the model
model.classifier=torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=model.classifier[1].in_features, 
                    out_features=1, 
                    bias=True),
    torch.nn.Sigmoid()
)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
epochs=6 #dont need many epochs
for epoch in range(epochs):
    model.train()
    run_loss=0
    for inputs, labels in train_loader:
        labels=labels.float()  
        optimizer.zero_grad()    
        outputs=model(inputs).squeeze() #in here to know what is the output can use summary(model,input_size=)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss+=loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {run_loss/len(train_loader):.4f}')
def predict(image_path):
    image=Image.open(image_path)
    image=transform(image).unsqueeze(0)  
    with torch.no_grad():
        output=model(image)
        prediction=output.item()  
    return prediction  # Return the probability
# torch.save(model, 'efficientnet_b3_alpaca_classifier_final.pth')
# Testing the model with test images (in here i haven't test with not alpaca)
test_path_alpaca=r'D:\code\alpaca_clf\test\alpaca'
images=[img for img in os.listdir(test_path_alpaca)]
true=0
for img in images:
    path_to_img=os.path.join(test_path_alpaca, str(img))
    prediction=predict(path_to_img)
    if prediction>=0.5:
        true+=1   
    print(str(prediction) + ": " + str(prediction>=0.5))
# Calculate accuracy of the test set
# print('Accuracy: ' + str(true/len(images) * 100) + ' %')
print(f'Accuracy: {(true/len(images))*100} %')
# Test with an random image in internet
print(predict(r"D:\code\alpaca_clf\test_with_random_image\test_image.jpg"))
'''Accuracy: 100.0 %
0.5803855061531067(random image)
pretty good
'''
