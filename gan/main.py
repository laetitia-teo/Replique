from dataset import ImgDataset
from gan import GAN
import os.path as op

path = op.join('..', 'datasets', 'small')

data = ImgDataset(path)
gan = GAN(data)
