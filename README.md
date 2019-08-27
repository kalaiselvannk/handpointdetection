#hand point detection


i have referenced open pose hand point detection model as base.(download_pretrained.sh)

i have downloaded the caffe model savefile and loaded the weights into keras model by converting it to np array and then to keras.(caffetokeras.py)

download and extract to dataset/train and test (download_dataset.sh)

the network outputs between -1 to 1.
output resolution is reduced by 8 from original image (if the image is 1200 x 800 x 3 then the output will be 300x100x22)
22-represents different handpoints as channel

training script starts to train from pretrained openpose weight as initial weight.



train_muti_stage.py output is predicted from multiple branches and final output layer to ensure the model focuses on necessary features in the beginning of the network itself.
following layers give (n/8 x m/8 x 22) shape outputs
conv6_2_CPM-
Mconv7_stage2
Mconv7_stage3
Mconv7_stage4
Mconv7_stage5
Mconv7_stage6




