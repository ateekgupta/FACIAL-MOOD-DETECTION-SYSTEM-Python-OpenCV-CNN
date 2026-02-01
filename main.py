import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision.transforms import transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# torch.manual_seed(42)

base_path = 'archive (7)/CK+48'
images = []
labels = []

for folder in ['anger','contempt','disgust','fear','happy','sadness','surprise']:
  folder_path = os.path.join(base_path,folder)
  for f in os.listdir(folder_path):
    img_path = os.path.join(folder_path,f)

    if folder == 'anger':
      labels.append(0)
    elif folder == 'contempt':
      labels.append(1)
    elif folder == 'disgust':
      labels.append(2)
    elif folder == 'fear':
      labels.append(3)
    elif folder == 'happy':
      labels.append(4)
    elif folder == 'sadness':
      labels.append(5)
    else:
      labels.append(6)

    images.append(img_path)

# len(images),len(labels)

from numpy.random import test
img_train,img_test,lbl_train,lbl_test = train_test_split(images,labels,test_size=0.2,random_state=42)

custom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class customdataset(Dataset):
  def __init__(self,img_paths,labels,transforms):
    self.img_paths = img_paths
    self.labels = labels
    self.transforms = transforms

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, index):
    img_path = self.img_paths[index]
    label = self.labels[index]

    # load image from disk
    image = Image.open(img_path).convert('RGB')

    image = self.transforms(image)

    return image,torch.tensor(label,dtype=torch.long)


train_dataset = customdataset(img_train,lbl_train,custom_transform)
test_dataset = customdataset(img_test,lbl_test,custom_transform)

train_loader = DataLoader(train_dataset,shuffle=True,batch_size=32)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=32)

import torchvision.models as models
vgg16 = models.vgg16(pretrained = True)
vgg16.classifier

# feature freezing
for param in vgg16.features.parameters():
  param.requires_grad = False


vgg16.classifier = nn.Sequential(
  nn.Linear(25088,4096),
  nn.ReLU(),
  nn.Dropout(p=0.5),
  nn.Linear(4096,4096),
  nn.ReLU(),
  nn.Dropout(p=0.5),
  nn.Linear(4096,7)
)
vgg16 = vgg16.to(device)

x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

out = vgg16(x)
print(out.shape)


lr = 0.001
epochs = 50

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(vgg16.classifier.parameters(),lr = lr)

for epoch in range(epochs):

  total_epoch_loss = 0
  for batch_images, batch_labels in train_loader:
    batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
    optimizer.zero_grad()
    y_pred = vgg16(batch_images)

    # loss
    loss = loss_func(y_pred,batch_labels)

    loss.backward()
    optimizer.step()

    total_epoch_loss += loss.item()

  avg_loss = total_epoch_loss / len(train_loader)
  print(f"epoch : {epoch + 1}, loss : {avg_loss}")





from torchmetrics import Accuracy
vgg16.eval()
accuracy = Accuracy(task = 'multiclass',num_classes=7).to(device)
accuracy.reset()
with torch.no_grad():
  for x,y in test_loader:
    accuracy.update(vgg16(x.to(device)),y.to(device))

avg_acc = accuracy.compute().item()
avg_acc

class_names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

import cv2
import torch
from PIL import Image
import numpy as np



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = custom_transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = vgg16(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            emotion = class_names[pred]
            confidence = probs[0][pred].item()

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )
        
    cv2.imshow("emotiondetector",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quitting")
        break



cap.release()
cv2.destroyAllWindows()
