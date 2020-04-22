import face_recognition
import cv2
from imutils import paths
import os
import numpy as np
import pickle
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw

def my_face_locations(pil_image):
    face_locations=[]
    ans = engine.detect_with_image(
            pil_image,
            threshold=0.05,
            keep_aspect_ratio=False,
            relative_coord=False ,
            top_k=10)

    for obj in ans:
        box= obj.bounding_box.flatten().astype("int").tolist()
        #box order is top, left, bottom, right, but embedding requires top, right, bottom, left order, so
        (left,top,right,bottom)=box
        face_locations.append(tuple((top,right,bottom,left)))
        #face_locations.append(tuple(int(i) for i in obj.bounding_box.flatten().tolist()))
        print("-----------------------------")
        print(obj.score)
        print(face_locations)
    return face_locations
    
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("./knowns"))

knownEncodings = []
knownNames = []
checkIn=[]
engine = DetectionEngine("./face_detect_model.tflite")


# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb)
    face_locations = my_face_locations(pil_image)
    encodings=face_recognition.face_encodings(rgb,face_locations)

    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        checkIn.append(False)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames,"checkin":checkIn}
f = open("encodings.pkl", "wb")
f.write(pickle.dumps(data))
f.close()




