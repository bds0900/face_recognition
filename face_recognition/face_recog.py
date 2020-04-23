# face_recog.py

import face_recognition
import cv2
import os
import numpy as np
import requests
import threading
import pickle
import datetime
import argparse

data = pickle.loads(open("./encodings.pkl", "rb").read())
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-r", "--room", type=str, default="2F04",
                    help='the room number')
parser.add_argument("-re", "--reset", type=int, default=30,
                    help='the reset interval(unit in minute)')
args = parser.parse_args()
print(args.room)
print(args.reset)

class myThread (threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
      mutation_query(self.name)


def check_in(query):
    print(query) 
    # headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7InNlcnZpY2UiOiJwaS1wcm9qZWN0QGRldiIsInJvbGVzIjpbImFkbWluIl19LCJpYXQiOjE1ODYyNjgwOTcsImV4cCI6MTU4Njg3Mjg5N30.9-C458JdJzhUBDZ3NxiZVdnT5XpiauRRzBhWBjHbYVw"}
    # request = requests.post('https://murmuring-fortress-24950.herokuapp.com/', json={'query': query}, headers=headers)
    # if request.status_code == 200:
    #     remaining_rate_limit = request.json()#["data"]#["class_id"] # Drill down the dictionary
    #     print("Remaining rate limit - {}".format(remaining_rate_limit))

    #     return remaining_rate_limit
    # else:
    #     raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

def mutation_query(name):
    currnet_time=f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
    query = """
    mutation{
        createAttendance(data:{
            time:"%s"
            room:"%s"
            student:"%s"
        }){
            id
        }
    }
    """ %(currnet_time, args.room,name)
    return check_in(query)


    
    
class FaceRecog():

    def __init__(self):
        # Using OpenCV to capture from device 0.
        self.camera = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.checkin=[]
        
        self.get_face_encoding()

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def reset(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.checkin=[]
        
        self.get_face_encoding()

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True


    def __del__(self):
        del self.camera

    def get_face_encoding(self):
        self.known_face_names=data["names"]
        self.known_face_encodings=data["encodings"]
        self.checkin=data["checkin"]

    #def get_face_encoding(self):


    def get_frame(self):
        # Grab a single frame of video
        #frame = self.camera.get_frame()
        rec,frame=self.camera.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.5:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]
                    if False==self.checkin[index]:
                        self.checkin[index]=True
                        thread1 = myThread(name)
                        thread1.start()


                self.face_names.append(name)
        

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame



if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if key == ord("r"):
            face_recog.reset()
        

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
