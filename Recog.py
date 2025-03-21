import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features

class_folder_path = 'D:/LFW Dataset/lfw_funneled'
class_names = os.listdir(class_folder_path)
num_classes = len(class_names)
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('celebrity_recognition_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(face_img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_names[predicted.item()]

        cv2.putText(frame, f'Predicted: {predicted_class}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
