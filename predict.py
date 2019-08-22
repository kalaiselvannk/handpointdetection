from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt


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
##--------predict on hands.jpg
image=cv2.imread("./handpointdetection/hands.jpg")
image_orig=cv2.resize(image,(960,540))
scale=368/image_orig.shape[1]
scale=scale*3
image=cv2.resize(image_orig,(0,0),fx=scale,fy=scale)


net_out=model.predict(np.expand_dims(image/256 -0.5,0))

out=cv2.resize(net_out[0],(image_orig.shape[1],image_orig.shape[0]))
image_out=image_orig

mask=np.zeros_like(image_out).astype(np.float32)

for chn in range(0,out.shape[-1]-2):
	m=np.repeat(out[:,:,chn:chn+1],3,axis=2)
	m=255*(np.abs(m)>0.09)
	mask=mask+m*(mask==0)
mask=np.clip(mask,0,255)
image_out=image_out*0.8+mask*0.2
print(image_out.shape)
image_out=np.clip(image_out,0,255).astype(np.uint8)
plt.imshow(image_out)
plt.show()


