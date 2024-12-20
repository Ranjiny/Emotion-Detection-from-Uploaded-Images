# Emotion-Detection-from-Uploaded-Images
Aim:
Upload an image through a Streamlit application and accurately detect and classify the emotion present in the image using Convolutional Neural Networks (CNNs).

Key components:

Image Upload Interface: The app will allow users to upload images, but only in valid formats (e.g., .jpg, .jpeg, .png) with proper size validation.

Face Detection: Once an image is uploaded, the system will use pre-trained models (such as Dlib or OpenCV) to detect faces within the image.

Facial Landmark Extraction: Key facial features (like eyes, mouth, eyebrows) will be extracted using tools like Dlib or Mediapipe to help in emotion classification.

Emotion Classification: A CNN model will classify the emotion of the person in the image, using datasets like FER-2013 for training.

Performance Optimization: The system will be optimized for real-time performance, ensuring quick image processing and classification.

Technologies:

Streamlit for web development

OpenCV or Dlib for face detection

PyTorch for CNN model development

FER-2013 dataset for emotion classification


Installation packages:
pip install streamlit torch torchvision dlib opencv-python Pillow matplotlib numpy scikit-learn

Download Dlib's "shape_predictor_68_face_landmarks.dat" in an appropriate location if the detector cannot be automatically loaded.
