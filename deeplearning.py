import numpy as np
import cv2
import tensorflow as tf
from scipy.special import softmax


# face detection model
face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                                                './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# face mask recognition model
model = tf.keras.models.load_model('face_cnn_model/')

# label 
labels = ['Mask', 'No Mask', 'Covered Mouth Chin', 'Covered Nose Mouth']


def getColor(label):
    if label == "Mask":
        color = (0,255,0)

    elif label == 'No Mask':
        color = (0,0,255)
    elif label == 'Covered Mouth Chin':
        color = (0,255,255)
    else:
        color = (255,255,0)
        
    return color


def face_mask_prediction(img,verification_label):
    # step - 1 : face detection
    label_text = "Can not detect"
    image = img.copy()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,117,123),swapRB=True)
    # 
    face_detection_model.setInput(blob)
    detection = face_detection_model.forward() # it will give the detection
    for i in range(0,detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.5:
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            box = box.astype(int)
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            # cv2.rectangle(image,pt1,pt2,(0,255,0),1)

            # step -2: Data preprocessing
            face = image[box[1]:box[3],box[0]:box[2]]
            face_blob = cv2.dnn.blobFromImage(face,1,(100,100),(104,117,123),swapRB=True)
            face_blob_squeeze = np.squeeze(face_blob).T
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            face_blob_flip = cv2.flip(face_blob_rotate,1)
            # normalization
            img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()
            # step-3: Deep Learning (CNN)
            img_input = img_norm.reshape(1,100,100,3)
            result = model.predict(img_input)
            result = softmax(result)[0]
            confidence_index = result.argmax()
            confidence_score = result[confidence_index]
            label = labels[confidence_index]
            label_text = '{}: {:,.0f} %'.format(label,confidence_score*100)
            verification_label.text = label_text
            print(label_text)
            # color
            color = getColor(label)
            cv2.rectangle(image,pt1,pt2,color,1)
            cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,2,color,2)
            
    return image,label_text

