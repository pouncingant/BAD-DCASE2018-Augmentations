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

Information on each dataset can be found at each of the links above, as well as the original competition website: [DCASE 2018 bird audio detection (BAD) competition](https://dcase.community/challenge2018/task-bird-audio-detection)

# Project structure

1. Data loading

The datasets are all arranged similarly (initially inspected using Pandas), with an audio folder containing all of the ogg files for that dataset as well as a single csv in the root directory of each dataset with two columns 1. the name of each audio file and 2. whether the audio file contains (1) or does not contain (0) bird vocalizations.

Due to the memory restrictions of Kaggle, I opted to generate spectrograms when grabbing items for each batch the model loads, rather than keep the spectrograms for the whole dataset in memory. This has the disadvantage of being very slow, and one major optimization would be to 1) create a secondary dataset containing the spectrograms for each audio file, and/or 2) use data reduction techniques (such as Mel spectrums) to potentially enable more efficient use of memory. It is worth noting that this inefficiency impacts all downstream processes, so I would recommend anyone repeating this process to implement the above optimizations, thereby affording more time for the hyperparameter search and training in general.

For simplicity, two augmentations (time axis flipping and per-channel energy normalization (PCEN)) are implemented during data loading; however, these are mentioned later.




2. Model definition
3. Augmentation
4. Training loop
5. Hyperparameter tuning
6. Production run
7. Analysis


