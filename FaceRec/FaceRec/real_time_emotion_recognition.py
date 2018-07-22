import cv2
import sys
import numpy as np
from keras.models import load_model

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

model = load_model('keras_model/model_5-49-0.62.hdf5')
model.get_config()


video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)

  
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (48, 48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32') / 255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
            result = EMOTIONS[np.argmax(model.predict(face_crop))]
            cv2.putText(frame, result, (x, y), font, 2, (255,69,0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Shivanand Roy', (400,470),font, 1, (255,69,0), 1, cv2.LINE_AA)
            face_image = feelings_faces[np.argmax(model.predict(face_crop))]
            for c in range(0, 3):
                frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


  # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

