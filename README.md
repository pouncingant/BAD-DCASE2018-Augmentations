# BAD-DCASE2018-Augmentations
This is bird-detection ML project using a simple convolutional neural network coded in Pytorch. It is based on the [DCASE 2018 bird audio detection (BAD) competition](https://dcase.community/challenge2018/task-bird-audio-detection) and runs on [Kaggle](https://www.kaggle.com/). The model takes in data as .ogg, converts them into spectrograms, applies augmentations to these spectrograms, then passes the augmented spectrograms on which the model is trained. The data is from the above competition, and is interesting in that it includes three different training sets, so the model must learn to detect the presence of bird signals in the audio regardless of the different soundscapes those signals appear.



# Model overview

The model itself has 4 convolution blocks with 32, 64, 128, and 64 channels, respectively. These each use batch normalization, ReLU, dropout and 2-pixel max pooling before an adaptive max pool and dense layers (with 64, 16, 16, and 1 neuron, respectively). The output provides a value for how certain the model is that the audio contains a bird call. The model was constructed using performance on the birdvox dataset to guide its design, mostly as an educational exercise, and achieved a baseline average ROCAUC of 0.711 using a three-fold experiment with the freefield, warbirb, and birdvox datasets (referenced in dataset overview), where two datasets are used for training and the other for validation.

# Data overview

Three datasets are used for training. They were converted to ogg format by Bilzard (a.k.a tatamiken; we are not affiliated in any way, but if he see's this, I extend my profound thanks!) as follows:
ffmpeg -i "{src_file}" -vn -ac 1 -ar 32000 -ab 72k -acodec libvorbis -f ogg "{dst_file}"

1. [Birdvox](https://www.kaggle.com/datasets/tatamikenn/birdvoxdcase20k)
2. [Freefield](https://www.kaggle.com/datasets/tatamikenn/freefield1010)
3. [Warblrb](https://www.kaggle.com/datasets/tatamikenn/warblrb10k) 



# Project structure

1. Data loading
2. Model definition
3. Augmentation
4. Training loop
5. Hyperparameter tuning
6. Production run
7. Analysis


