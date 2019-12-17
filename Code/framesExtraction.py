# Program To Read video 
# and Extract Frames 
import cv2 
import os
  
# Function to extract frames 
def FrameCapture(pathVideo, pathDir): 
      
		# Path to video file 
		vidObj = cv2.VideoCapture(pathVideo) 

		# Used as counter variable 
		count = 0

		# checks whether frames were extracted 
		success = 1
		os.mkdir(pathDir)
		print(pathDir)
		while success: 

		    # vidObj object calls read 
		    # function extract frames 
		    #print("Hello")
		    success, image = vidObj.read() 

		    # Saves the frames with frame-count
		    
		    framePath = os.path.join(pathDir, str(count)+".jpg") 
		    cv2.imwrite(framePath, image) 

		    count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    
    # FrameCapture("../aff_wild_videos_annotations_bboxes_landmarks/aff_wild_annotations_bboxes_landmarks_new/videos/train/105.avi") 
    '''
    rootDir = "../aff_wild_videos_annotations_bboxes_landmarks/aff_wild_annotations_bboxes_landmarks_new/videos/train"
    os.mkdir("Frames")
    for subdir, dirs, files in os.walk(rootDir) :
    		for f in files:
    				pathVideo = os.path.join(rootDir, f)    				
    				pathDir = os.path.join("Frames", f.split(".")[0])
    				FrameCapture(pathVideo, pathDir)
    '''			
    
    fileInput = open("Input.csv","w")	
    rootDir1 = "../aff_wild_videos_annotations_bboxes_landmarks/aff_wild_annotations_bboxes_landmarks_new/annotations/train/arousal"
    rootDir2 = "../aff_wild_videos_annotations_bboxes_landmarks/aff_wild_annotations_bboxes_landmarks_new/annotations/train/valence"
    
    path_list = []
    arousal_list = []
    valence_list = []
    for _, _, files in sorted(os.walk(rootDir1)):
    	for index, f in enumerate(sorted(files)) :
    		aroPath = os.path.join(rootDir1, f)
    		valPath = os.path.join(rootDir2, f)
    		print(aroPath, valPath)
    		with open(aroPath) as f1 , open(valPath) as f2 : 
    			for index, (l1, l2) in enumerate(zip(f1, f2)) :
    				imgPath = os.path.join("Frames", f.split(".")[0], str(index)+".jpg")
    				print("-----------"+imgPath)
    				fileInput.write(imgPath+","+l2.strip()+","+l1.strip()+"\n")
    			
    	
