import face_recognition
import cv2
import os
import pickle

print("[INFO] quantifying faces...")

known_encodings = []
known_names = []
checkin=[]

dirname = 'knowns'
files = os.listdir(dirname)
# loop over the image paths
for filename in files:
    name, ext = os.path.splitext(filename)
    if ext == '.jpg':
        known_names.append(name)
        pathname = os.path.join(dirname, filename)
        img = face_recognition.load_image_file(pathname)
        face_encoding = face_recognition.face_encodings(img)[0]
        
        known_encodings.append(face_encoding)
        checkin.append(False)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": known_encodings, "names": known_names,"checkin":checkin}
f = open("encodings.pkl", "wb")
f.write(pickle.dumps(data))
f.close()




