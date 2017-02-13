from tflearn.layers.merge_ops import merge
from tflearn.layers.conv import conv_2d
import os

def create_fire(input,s_output_filters):
    fire2_squeeze = conv_2d(input, s_output_filters, 1, activation='relu')
    fire2_expand1 = conv_2d(fire2_squeeze, 4*s_output_filters, 1, activation='relu')
    fire2_expand2 = conv_2d(fire2_squeeze, 4*s_output_filters, 3, activation='relu')
    network = merge([fire2_expand1, fire2_expand2], mode='concat', axis=3)
    return network

def getXY(files_list,image_shape):
    # Build a HDF5 dataset (only required once)
    from tflearn.data_utils import build_hdf5_image_dataset
    if not os.path.isfile('dataset.h5'):
        build_hdf5_image_dataset(files_list, image_shape=image_shape, mode='file', output_path='dataset.h5',
                                 categorical_labels=True, normalize=True)

    # Load HDF5 dataset
    import h5py
    h5f = h5py.File('dataset.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X,Y

