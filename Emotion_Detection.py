import streamlit as st
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import torch
import dlib
import matplotlib.pyplot as plt


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*1*1,256)
        self.fc2 = nn.Linear(256,7)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = torch.relu(self.pool(X))
        X = self.conv3(X)
        X = torch.relu(self.pool(X))
        X = self.conv4(X)
        X = torch.relu(self.pool(X))
        X = self.conv5(X)
        X = torch.relu(self.pool(X))
        X = torch.flatten(X,1)
        X = self.fc1(X)
        X= self.dropout(X)
        X = self.fc2(X)
        return X
#Load pre-trained model
MODEL_PATH = 'D://ranjiny//Guvi_python//Finalproject//model_complete.pth'
loaded_model = torch.load(MODEL_PATH)
loaded_model.eval()

#Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D://ranjiny//Guvi_python//Finalproject//shape_predictor_68_face_landmarks.dat')

#Preprocess the image
def preprocess_image(img):
    transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])
    return transform(img)


# Streamlit page configuration
st.set_page_config(page_title="Emotion Detection from Images", layout="centered")

# Title of the app
st.title("Emotion Detection from Uploaded Images")

# Sidebar configuration
st.sidebar.title("Upload Image")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

#File upload feature
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]
                                         ,key=st.session_state["uploader_key"], accept_multiple_files=False)

if uploaded_file is not None:
    # Display uploaded image
    display_image=st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Extract landmarks from the image
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    image_np=img.resize((128,128))
    image_np = np.array(img)
    faces = detector(image_np)
    continue_processing=False
    
    if len(faces) == 0:
        with st.sidebar.container() as container:
                         
            error_message=st.error("No faces detected. Do you want to continue?")
            col1,col2=st.columns(2)
                         
            if col1.button("Clear"):
                st.session_state["uploader_key"] += 1
                st.rerun()
            elif col2.button("Continue"):
                continue_processing = True
                error_message.empty()  # Clear the error message
                st.warning('Despite detecting no faces trying to process the image file')
          
    if continue_processing or len(faces) > 0:
        if len(faces)>1:
             st.sidebar.write('Multiple faces detected')
        st.sidebar.success('Image file processed successfully')
        for face in faces:
            landmarks = predictor(image_np, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image_np, (x, y), 2, (0, 255, 0), -1)
        image_np=cv2.resize(image_np,(48,48))

        # Preprocess the image
        preprocessed_img = preprocess_image(Image.fromarray(image_np))

        # Add batch dimension
        preprocessed_img = preprocessed_img.unsqueeze(0)

        # Predict emotion
        with torch.no_grad():
            prediction = loaded_model(preprocessed_img)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_emotion = emotion_labels[torch.argmax(probabilities)]

        # Display predicted emotion
        st.markdown(f"<h1 style='font-size: 30px;color: blue;'>Predicted Emotion: {predicted_emotion}</h1>"
                    , unsafe_allow_html=True)

        # Visualize the prediction
        fig, ax = plt.subplots()
        ax.pie(probabilities[0], labels=emotion_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)


