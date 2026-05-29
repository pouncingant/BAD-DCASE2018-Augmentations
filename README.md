# BAD-DCASE2018-Augmentations
This is bird-detection ML project using a simple convolutional neural network coded in Pytorch. It is based on the [DCASE 2018 bird audio detection (BAD) competition](https://dcase.community/challenge2018/task-bird-audio-detection) and runs on [Kaggle](https://www.kaggle.com/).

The model takes in data as .ogg, converts them into spectrograms, applies augmentations to these spectrograms, then passes the augmented spectrograms on which the model is trained. The data is from the above competition, and is interesting in that it includes three different training sets, so the model must learn to detect the presence of bird signals in the audio regardless of the different soundscapes those signals appear. Thus, the datasets provide a basis upon which to test the applicability of bird audio detection models in the real-world, where day-to-day, location, and recording equipment related variation as well as class imbalances (varying sparcity of bird calls in different datasets) must be accounted for. 

# Model overview

The model itself has 4 convolution blocks with 32, 64, 128, and 64 channels, respectively. These each use batch normalization, ReLU, dropout and 2-pixel max pooling before an adaptive max pool and dense layers (with 64, 16, 16, and 1 neuron, respectively). The output provides a value for how certain the model is that the audio contains a bird call.

# Data overview

Three datasets are used for training. They were converted to ogg format by Bilzard (a.k.a tatamiken; we are not affiliated in any way, but if he see's this, I extend my profound thanks!) as follows:
ffmpeg -i "{src_file}" -vn -ac 1 -ar 32000 -ab 72k -acodec libvorbis -f ogg "{dst_file}"

1. [Birdvox](https://www.kaggle.com/datasets/tatamikenn/birdvoxdcase20k): 20,000 samples, Ithaca, NY, USA, 10 s each, samples without/with bird 9983:10017
2. [Freefield](https://www.kaggle.com/datasets/tatamikenn/freefield1010): 7,690 samples, worldwide, varying length, samples without/with bird 5755:1935 
3. [Warblrb](https://www.kaggle.com/datasets/tatamikenn/warblrb10k), 8,000 samples, various UK, varying length, samples without/with bird 1955:6045 

Information on each dataset can be found at each of the links above, as well as the original competition website: [DCASE 2018 bird audio detection (BAD) competition](https://dcase.community/challenge2018/task-bird-audio-detection)

# Project structure

1. Data loading

The datasets are all arranged similarly (initially inspected using Pandas), with an audio folder containing all of the ogg files for that dataset as well as a single csv in the root directory of each dataset with two columns 1. the name of each audio file and 2. whether the audio file contains (1) or does not contain (0) bird vocalizations.

The overall procedure for loading data is:
1) For the initial hyperparameter search, the first 333 samples of each dataset are used. For the second hyperparameter search, the first 666 are used. The final test of  parameterization uses all of the samples.   
2) Define 3 training and validation sets each comprising two training datasets with the remaining dataset as the validation set. During the hyperparameter search, the validation set is set to 10% the size of the training set to avoid overfitting the validation set. Each set contains a) the path to the audio file and b) the 0/1 indicator of bird signal presence
3) for each training and validation set a training and validation DataLoader is created.  

Due to the memory restrictions of Kaggle, I opted to generate spectrograms when grabbing items for each batch the model loads, rather than keep the spectrograms for the whole dataset in memory. This has the disadvantage of being very slow, and one major optimization would be to 1) create a secondary dataset containing the spectrograms for each audio file, and/or 2) use data reduction techniques (such as Mel spectrums) to potentially enable more efficient use of memory. It is worth noting that this inefficiency impacts all downstream processes, so I would recommend anyone repeating this process to implement the above optimizations, thereby affording more time for the hyperparameter search and training in general.

For simplicity, two augmentations (time axis flipping and per-channel energy normalization (PCEN)) are implemented during data loading; however, these are mentioned later.

2. Model definition

The model has 4 convolution blocks with 32, 64, 128, and 64 channels, respectively. These each use batch normalization, ReLU, dropout and 2-pixel max pooling before an adaptive max pool and dense layers (with 64, 16, 16, and 1 neuron, respectively). 

The model was constructed using performance on the birdvox dataset to guide its design, mostly as an educational exercise, based loosely on the introduction to convolutional neural networks (CNN) described in the FastAI course.

It was found that average pooling was detrimental to the model compared to max pooling, and since a prediction requires on detecting /any/ bird signal in the audio, max pooling makes more intuitive sense as it allows short signals to remain strong even within long clips (birdvox clips are 10 s each). Other design decisions were primarily arrived at via trial and error, motivated by 1) ensuring the model could overfit on a small dataset, to ensure that it has the ability to detect signals in the data, and 2) maximizing the receiver operating characteristic/area under the curve (ROCAUC) on the validation set.

3. Augmentation

After finalizing the model, its performance on the 3-fold evaluation was a an average validation ROCAUC (vROCAUC) of 0.711. In the hopes of improving this score, several augmentations were made to the spectrograms either during data loading or during the training loop, as listed below:

a. time axis flipping
- A simple flip along the time axis
b. per-channel energy normalization (PCEN)
- Normalization of energy across each frequency
c. frequency masking
- 
d. time masking
e. mixup
f. white noise

Augmented vROCAUC 0.728 

4. Training loop
5. Hyperparameter tuning
6. Production run
7. Analysis


