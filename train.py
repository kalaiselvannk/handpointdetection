from keras.layers import *
from keras.models import *
import numpy as np
import cv2
import glob
import json
##keras model definition-----referenced from open pose

input_ = Input((None, None, 3), name='image')

def main_block(input_, padding='same'):
    x = Conv2D(64, (3,3), strides=(1,1), padding=padding, name='conv1_1', activation='relu')(input_)
    x = Conv2D(64, (3,3), strides=(1,1), padding=padding, name='conv1_2', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool1_stage1')(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv2_1', activation='relu')(x)
    x = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv2_2', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool2_stage1')(x)

    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_1', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_2', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_3', activation='relu')(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding=padding, name='conv3_4', activation='relu')(x)
    x = MaxPool2D((2,2), strides=(2,2), name='pool3_stage1')(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_1', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_2', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_3', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv4_4', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv5_1', activation='relu')(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding=padding, name='conv5_2', activation='relu')(x)
    conv5_3_CPM = Conv2D(128, (3,3), strides=(1,1), padding=padding, name='conv5_3_CPM', activation='relu')(x)
    x = Conv2D(512, (1,1), strides=(1,1), padding='valid', name='conv6_1_CPM', activation='relu')(conv5_3_CPM)
    conv6_2_CPM = Conv2D(22, (1,1), padding='valid', strides=(1,1), name='conv6_2_CPM')(x)
    return conv5_3_CPM, conv6_2_CPM

conv5_3_CPM, conv6_2_CPM = main_block(input_)

def stage_block(conv5_3_CPM, prev_stage, stage, padding='same'):
    x = concatenate([ prev_stage, conv5_3_CPM ], axis=3, name='concat_stage{}'.format(stage))
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv1_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv2_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv3_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv4_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (7,7), strides=(1,1), padding=padding, name='Mconv5_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(128, (1,1), strides=(1,1), padding='valid', name='Mconv6_stage{}'.format(stage), activation='relu')(x)
    x = Conv2D(22, (1,1), strides=(1,1), padding='valid', name='Mconv7_stage{}'.format(stage))(x)
    return x
    
prev_stage = conv6_2_CPM
for stage in range(2, 7):
    prev_stage = stage_block(conv5_3_CPM, prev_stage, stage)
    
x = prev_stage
model = Model(input_, x)

#------ end of model definition

model.load_weights( "./model_weights.h5")
print("weights loaded")


def _eucl_loss(y_true,y_pred):
	return K.sum(K.square(y_true - y_pred)) / batch_size / 2
###
'''
def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses

loss_funcs = get_loss_funcs()
###'''
model.compile(loss='mean_squared_error', optimizer='sgd',,metrics=['mae'])


batchsize=1
labels=glob.glob("dataset/train/*/*/*.json")
labels.sort()
basepath=[]
for i in range(len(labels)):
	images.append(labels[i].split(".json")[0])

def image_generator(files, batch_size = 1):
	while True:
		batch_paths = np.random.choice(a = files,size = batch_size)
		x_image=cv2.imread(batch_paths+".jpg")
		y_label=json.load(batch_paths+".json")
		y_label=np.array(y_label['hand_pts'])
		tmp=np.zeros((1,x_image.shape[1],x_image.shape[2],22))
		for i in range(22):
			tmp[0][int(y_label[i][1]/8)][int(y_label[i][0]/8)][i]=1
		yield (x_image,tmp)

model.fit_generator(image_generator(basepath),steps_per_epoch=100, epochs=10)		

	
		



