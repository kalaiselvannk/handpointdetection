wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip
wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels_synth.zip
unzip -q hand_labels.zip
unzip -q hand_labels_synth.zip
mkdir dataset
mkdir dataset/train
mkdir datase/test
mv hand_labels dataset/train
mv hand_labels_synth dataset/train

