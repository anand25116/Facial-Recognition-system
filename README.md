# 🎭 **Facial Recognition System**  

🚀 This project is a **Facial Recognition System** that uses **ResNet50** to recognize celebrities from the **LFW dataset**. It includes:  

- 📌 **`main.py`** → Trains a deep learning model using **PyTorch**  
- 🎥 **`recog.py`** → Recognizes faces in real time using **OpenCV**  

---

## ✨ **Features**  

✅ **Deep Learning-based Face Recognition** using **ResNet50**  
✅ **Trains on the LFW Dataset** (Labelled Faces in the Wild)  
✅ **Real-time Face Detection & Recognition** using **OpenCV**  
✅ **Transfer Learning** for better efficiency  
✅ **Supports CUDA acceleration** (if available) 🚀  

---

## 🔧 **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/anand25116/facial-recognition-system.git
cd facial-recognition-system
```

### **2️⃣ Install Dependencies**  
Make sure you have Python **3.x** installed. Then, run:  
```sh
pip install torch torchvision opencv-python tqdm pillow
```

### **3️⃣ Download the LFW Dataset**  
- Get the dataset from [here](http://vis-www.cs.umass.edu/lfw/)  
- Extract it and place it inside your project folder:  
  ```
  D:/LFW Dataset/lfw_funneled
  ```

---

## 🏋️‍♂️ **Training the Model**  
Run the following command to start training:  
```sh
python main.py
```
This will train **ResNet50** on the dataset and save the model as:  
```sh
celebrity_recognition_model.pth
```

---

## 🎭 **Real-Time Face Recognition**  
Run the recognition script to detect and classify faces from your webcam:  
```sh
python recog.py
```
Press **`Q`** to exit the webcam feed.  

---

## 👨‍💻 **Technologies Used**  
🔹 **Python** 🐍  
🔹 **PyTorch** 🔥  
🔹 **OpenCV** 📸  
🔹 **ResNet50 (Pretrained Model)** 🏆  
🔹 **TQDM for Progress Tracking** 📊  

---

## 📌 **To-Do / Future Enhancements**  
🚀 Add **real-time FPS counter** for performance analysis  
🚀 Improve **dataset diversity** for better generalization  
🚀 Implement **face embeddings** for advanced recognition  

---

## 🤝 **Contributing**  
Contributions are welcome! Feel free to **fork** this repo and submit a **pull request**.  


