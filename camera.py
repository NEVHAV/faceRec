import cv2
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import face_recognition
import dlib

# use haar to train to define human's face
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

video_capture = cv2.VideoCapture(0)
start = 0

face_location = []
face_encoding = []
tolerance = 0.6 #do giong nhau (cang thap thi cang yeu cau giong nhau cao)
face_dict = {}
index = 0

#scan 4 frame per once
process_this_frame = 1

start = 0

while True:
	process_this_frame = process_this_frame%4
	if process_this_frame%4 != 0:
		process_this_frame += 1
		continue

	_, frame = video_capture.read()
	# print(frame.shape)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

	face_location = face_recognition.face_locations(frame)
	face_encoding = face_recognition.face_encodings(frame, face_location)

	if len(face_encoding) == 0:
		if start != 0:
			print ('End time: ', datetime.datetime.now())
		start = 0
	
	#save new face into face_dict, if face_dict had that face then update it
	for x in face_encoding:
		if start == 0:
			print ('Start time: ', datetime.datetime.now())
		start +=1
		matches = face_recognition.compare_faces(list(face_dict.values()), x, tolerance)
		# print (matches)
		if True in matches:
			first_match_index = matches.index(True)
			face_dict[first_match_index] = x
			print ('User ID: ', first_match_index)
			continue
		else:
			print('New face! Your ID is ', index)	
			face_dict[index] = x
			index += 1

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.imshow('Webcam', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()