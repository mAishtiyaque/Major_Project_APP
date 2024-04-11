import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
face_haar_cascade = cv2.CascadeClassifier('haar.xml')

model = model_from_json(open("detect.json", "r").read())
model.load_weights('detect.h5')
def classify(image):
    #cap=cv2.VideoCapture(0)
    cap=image
    while True:
        frame=image
        #res,frame=cap.read()
        height, width , channel = frame.shape
        # Creating an Overlay window to write prediction and cofidence
        sub_img = frame[0:int(height/8),0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.7, black_rect,0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        lable_color = (10, 210, 10)
        lable = "Facial Emotion Detection"
        lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
        textX = int((res.shape[1] - lable_dimension[0]) / 2)
        textY = int((res.shape[0] + lable_dimension[1]) / 2)
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image )
        result=[]
        try:
            for (x,y, w, h) in faces:
                roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
                roi_gray=cv2.resize(roi_gray,(48,48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis = 0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
                emotion_prediction = emotion_detection[max_index]
                face_detail={
                    "position":(int(x),int(y), int(w), int(h)),
                    "emotion":emotion_prediction,
                     "confidence":float(np.round(np.max(predictions[0])*100,1))
                             }
                result.append(face_detail)
        except :
            pass
        return result
