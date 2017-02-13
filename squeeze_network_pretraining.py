import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
import create_models
import util

model_path = "./squeeze_model/"
files_list = "./train_file/data.txt"

def train(network, X, Y):
    # Training
    model = tflearn.DNN(network, checkpoint_path='./squeeze_pre_training_checkpoint/squeeze_pre_training',
                        max_checkpoints=3, tensorboard_verbose=2,
                        tensorboard_dir="./logs")

    model_file = os.path.join(model_path, "model_squeeze_pre_training.ckpt")
    if os.path.isfile(model_file + ".index"):
        model.load(model_file, weights_only=True)
        print('load modle:' + model_file)

    # Start finetuning
    model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=32, snapshot_epoch=True,
              snapshot_step=200, run_id='squeeze_pre_training')

    model.save(model_file)

if __name__ == '__main__':
    X, Y = util.getXY(files_list,(224, 224))

    num_classes = 78  # num of your dataset
    # preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)

    img_aug = ImageAugmentation()
    img_aug.add_random_rotation(20.0)
    # Network
    x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                           data_preprocessing=img_prep,data_augmentation=img_aug)

    regression = create_models.create_squeezeNet_v1_1(x,num_classes)
    train(regression,X,Y)