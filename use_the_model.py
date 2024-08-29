import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchinfo import summary
#must be the same as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''i dont know why if i using saved parameters the accuracy is significantly decrease'''
# Load the saved model
model = torch.load('efficientnet_b3_alpaca_classifier_final.pth')
model.eval()  
'''if we import predict function from training_the_model.py this will train the model again'''
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
        prediction = output.item()  
    return prediction  # Return the probability

# Test the prediction function with an image
image_path = r'D:\code\alpaca_clf\test_with_random_image\test_image.jpg'
probability = predict(image_path)
print(f'Predicted probability: {probability:.4f}')
test_path_alpaca = r'D:\code\alpaca_clf\test\alpaca'
images = [img for img in os.listdir(test_path_alpaca)]
true = 0
for img in images:
    path_to_img = os.path.join(test_path_alpaca, str(img))
    prediction = predict(path_to_img)
    if prediction >= 0.5:
        convert = True 
        true += 1
    else:
        convert = False    
    print(str(prediction) + ": " + str(convert))
# Calculate accuracy of the test set
print('Accuracy: ' + str(true/len(images) * 100) + ' %')
print(summary(model))