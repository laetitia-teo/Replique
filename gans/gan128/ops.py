import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, 
                                    Activation, 
                                    AveragePooling2D,
                                    Concatenate,
                                    Add,
                                    BatchNormalization,
                                    UpSampling2D

class ResBlockUp(Layer):
	"""
    Residual block going down from the image.
    Used in the generator architecture.
    Args :
    	- x (Tensor) : input of the block;
        - filters (tuple) :  number of filters in the convolutions;
        - size : size of the kernel in the convolution;
        - strides : stride of the convolution.
    """
	def __init__(self,
			     filters,
			     size=(3, 3),
			     stride=(2, 2),
			     **kwargs):

		super(ResBlockUp, self).__init__(**kwargs)
		# layers - main path
		self.batchnorm =  BatchNormalization()
		self.relu = Activation('relu')
		self.up1 = UpSampling2D(interpolation='blinear')
		self.conv1 = Conv2D(filters, size, stride)
		# layers - skip path
		self.up2 = UpSampling2D(interpolation='blinear')
		self.conv2 = Conv2D(filters, (1, 1), (1, 1))

		self.add = Add()

	def call(self, x):
		skip_x = x

		# main path
		x = self.batchnorm(x)
		x = self.relu(x)
		x = self.up1(x)
		x = self.conv1(x)
		# skip path
		skip_x = self.up2(skip_x)
		skip_x = self.conv2(skip_x)
		# merge
		x = self.add([x, skip_x])

		return x

class ResBlockDown(Layer):
	"""
    Residual block going down from the image.
    Used in the discriminator architecture.
    Args :
        - x (Tensor) : input of the block;
        - filters (tuple) :  number of filters in the convolutions;
        - size : size of the kernel in the convolution;
        - strides : stride of the convolution.
    """
    def __init__(self,
			     filters,
			     size=(3, 3),
			     stride=(2, 2),
			     **kwargs):

    	f1, f2 = filters
    	# layers - main path
    	self.relu = Activation('relu')
    	self.conv1 = Conv2D(f1, size, stride)
    	self.conv2 = Conv2D(f2, size, stride)
    	# layers - skip path
    	self.conv3 = Conv2D(f2, (1, 1), (1, 1))
    	self.avg = AveragePooling2D()

    	self.add = Add()

	def call(self, x):
		skip_x = x

		# main path
		x = self.relu(x)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		# skip path
		skip_x = self.conv3(skip_x)
		skip_x = self.avg(skip_x)
		# merge
		x = self.add([x, skip_x])

		return x
