# Our Working Env
Linux- Debian based x64

# Istallation
*To run the face_recognition applicaion you need these packages 
*python3 opencv-python opencv-contrib-python dlib face_recognition

If you are using Linux, it is recommended to separate the environment with virtualenv as below before installing the package
*$ sudo apt-get install python3 python3-dev python3-venv
*$ python3 -m venv py3
*$ source py3/bin/activate
*(py3) $ pip install --upgrade pip

Next, install the necessary packages as shown below
*(py3) $ pip install opencv-python
*(py3) $ pip install opencv-contrib-python
*(py3) $ pip install dlib
*(py3) $ pip install face_recognition

# Files
## pre-processing.py
You must run pre-processing.py before running face_recog.py. This extracts and encodes face features from images which in the Knowns folder

## face_recog.py
Compare the pre-encoded features with the features obtained through the camera, and if the two values are similar, they are regarded as the same person