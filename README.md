# Facial-emotion-Recognition

Aff-Wild data consists of Videos. SO

framesExtraction.py will extract all frames and save location of it in csv file.

Extract features from this frames using pretrained VGG16 model it's done in save_features_vgg16.py. Features are saved in data/bottleneck_

Then train this features usinf lstm and neural network mode in train.py
