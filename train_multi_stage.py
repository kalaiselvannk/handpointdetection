from keras.layers import *
from keras.models import *
import keras
import numpy as np
import cv2
import glob
import json
##keras model definition-----referenced from open pose
filepath="./model.h5"
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
x=[]    
prev_stage = conv6_2_CPM
x.append(prev_stage)
for stage in range(2, 7):
    prev_stage = stage_block(conv5_3_CPM, prev_stage, stage)
    x.append(prev_stage)

model = Model(input_, x)

#------ end of model definition

model.load_weights( "./model_weights.h5")
print("weights loaded")



model.compile(loss = 'mean_square_error', optimizer='sgd',metrics=['mae'])


batchsize=1
labels=glob.glob("./**/*.json",recursive=True)
labels.sort()
print(labels)
basepath=[]
for i in range(len(labels)):
	basepath.append(labels[i].split(".json")[0])

def image_generator(files, batch_size = 1):
	while True:
		x_image=[];y_label=[]
		while True:
			batch_paths = np.random.choice(a = files)
			x_image=cv2.imread(batch_paths+".jpg")
			x_image=np.expand_dims(x_image/256 -0.5,0)
			if x_image.shape[1]%8==0 and x_image.shape[2]%8==0:
				break;
		y_label=json.load(open(batch_paths+".json"))
		y_label=np.array(y_label['hand_pts'])
		print(y_label.shape)
		print(x_image.shape)
		tmp=np.zeros((1,int(x_image.shape[1]/8),int(x_image.shape[2]/8),22))
		print(tmp.shape)
		for i in range(21):
			tmp[0][int(y_label[i][1]/8)][int(y_label[i][0]/8)][i]=1
		yield (x_image,[tmp,tmp,tmp,tmp,tmp,tmp])

model.fit_generator(image_generator(basepath),steps_per_epoch=20, epochs=10,callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='mean_absolute_error', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
])		

	
		



