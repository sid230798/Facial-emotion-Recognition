from data_process import read_labeled_image_list
import cv2
import os

face_cascade = cv2.CascadeClassifier('face.xml')
face_file = open("Input_Faces_Project_2.csv", "w")

def convt_to_faces(file, label) :

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    #for (x, y,w,h) in faces :
    #	print(x, y,w,h)
    x = 160
    y = 75
    w = 128
    h = 128
    offset=30
    #print(faces)
    face_clip = image[y-offset:y+h+offset, x-offset:x+w+offset]  #cropping the face in image
    file_path = "Project_frames" + file[6:]
    #print(file_path)
    face_file.write(file_path+","+str(label[0])+","+str(label[1])+"\n")
    cv2.imwrite(file_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it




filenames, labels = read_labeled_image_list('Input.csv')
image_list = []

## 18240
for index, names in enumerate(filenames[10076: 18240]) :
    #print(names, labels[index])
    if os.path.isdir("Project_frames" + names[6:10]) == False :
        os.mkdir("Project_frames" + names[6:10])
    convt_to_faces(names, labels[index])

#print(image_list[0].shape)
