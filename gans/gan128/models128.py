import tensorflow as tf

from tensorflow.keras.layers import Dense
from ops import res_block_down, res_block_up

class DeepGen(tf.keras.Model):
    """
    Deep Convolutional Generator.
    
    Args:
        - zdim : dimension of the latent space.
    """
    def __init__(self, z_dim):
    	super(DeepGen, self).__init__()
    	self.z_dim = z_dim
    	# layers
    	self.dense1 = Dense()
    	self.res_block1 = res_block_up
    	
    
def deep_discr():
	"""
	Deep Convolutional Discriminator.
	"""
	
