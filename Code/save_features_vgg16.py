import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

file_path = open('Input_Faces_Project_2.csv', 'r')
VGG16 = tf.keras.applications.VGG16(weights='imagenet', pooling='max', include_top=False)

for index, line in enumerate(file_path):

    line = line.split(",")
    img_path = line[0]
    val = float(line[1])
    arou = float(line[2])

    img = mpimg.imread(img_path)
    img = np.expand_dims(img, axis=0).astype('float')/255
    #img_tensor = tf.convert_to_tensor(img, dtype='float64')/255
    file_path = "Data2/bottleneck_" + str(index)
    features = VGG16.predict(img)
    np.save(file_path, features)
    print("Index : ", str(index), "...Completed")
