#!/bin/bash
# cd into folder
cd ~/catkin_ws/src/person_detector/

# Remove old files
rm -rf ./database/aligned-faces/*
rm -rf ./models/openface/classifier.pkl
rm -rf ./models/openface/reps.csv
rm -rf ./models/openface/labels.csv
rm -rf ./models/openface/user_mapping.csv

cd ~/programs/openface/util/

# align images and store in ./database/aligned-images
./align-dlib.py $HOME/catkin_ws/src/person_detector/database/faces \
                align outerEyesAndNose \
                $HOME/catkin_ws/src/person_detector/database/aligned-faces \
                --size 96

# batch represent images
cd ../batch-represent
./main.lua -outDir $HOME/catkin_ws/src/person_detector/models/openface \
           -data $HOME/catkin_ws/src/person_detector/database/aligned-faces/

# Train classifier
cd ~/catkin_ws/src/person_detector
./scripts/trainOpenFace.py --cuda train ./models/openface

# Remove old files
#rm -rf ./face_recognizer_files/aligned-images/* &&
#rm -rf ./face_recognizer_files/classifier.pkl &&
#rm -rf ./face_recognizer_files/reps.csv &&
#rm -rf ./face_recognizer_files/labels.csv &&
#rm -rf ./face_recognizer_files/user_indx_mapping.csv &&

# align images and store in ./face_recognizer_files/aligned-images
#../openface/util/align-dlib.py ./face_recognizer_files/training-images align outerEyesAndNose ./face_recognizer_files/aligned-images/ --size 96 &&

# batch represent images
#../openface/batch-represent/main.lua -outDir ./face_recognizer_files/ -data ./face_recognizer_files/aligned-images/ &&

# train classifier
#./scripts/train_open_face.py train ./face_recognizer_files

# Error in batch-represent -- changed this to fix:

  # line 130-134 in ~/blueberry/libraries/openface/batch-represent/dataset.lua
  
    # Commented out:
        # if jit.os == 'OSX' then
        #   wc = 'gwc'
        #   cut = 'gcut'
        #   find = 'gfind'
        # end
        
# Uncomment to change back        
