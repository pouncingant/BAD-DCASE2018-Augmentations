# BAD-DCASE2018-Augmentations
This is a bird detection ML project using a simple convolutional neural network coded in Pytorch. It is inspired by the [DCASE 2018 bird audio detection (BAD) competition](https://dcase.community/challenge2018/task-bird-audio-detection) and runs on [Kaggle](https://www.kaggle.com/).

The model takes in data as .ogg (alongside an indicator of whether the data contains a bird vocalization), converts them into spectrograms, applies augmentations to these spectrograms, then uses the augmented spectrograms to train the model to recognize whether there is a bird in an audio sample or not. The data is from the above competition, and is interesting in that it includes three different training sets, so the model must learn to detect the presence of bird signals in the audio regardless of the different soundscapes those signals appear. Thus, the datasets provide a basis upon which to test the applicability of bird audio detection models in the real-world, where day-to-day, location, and recording equipment related variation as well as class imbalances (varying sparcity of bird calls in different datasets) must be accounted for. 

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
1) For the initial hyperparameter search, the first 333 samples of each dataset are used. For the second hyperparameter search, the first 666 are used. The final test of parameterization uses all of the samples.
2) Three training and validation sets are defined, each comprising two training datasets with the remaining dataset as the validation set. During the hyperparameter search, the validation set is set to 10% the size of the training set to avoid overfitting the validation set as a whole. Each set contains a) the path to the audio file and b) the 0/1 indicator of bird signal presence
3) for each training and validation set a training and validation DataLoader is created.  

Due to the memory and working directory restrictions of Kaggle, I opted to generate spectrograms when grabbing items for each batch the model loads, rather than keep the spectrograms for the whole dataset in memory. This has the disadvantage of being very slow, and one major optimization would be to 1) create a secondary dataset containing the spectrograms for each audio file, and/or 2) use data reduction techniques (such as Mel spectrums) to potentially enable more efficient use of memory. It is worth noting that this inefficiency impacts all downstream processes, so I would recommend anyone repeating this process to implement the above optimizations, thereby affording more time for the hyperparameter search and training in general.

For simplicity, two augmentations (time axis flipping and per-channel energy normalization (PCEN)) are implemented during data loading; however, these are mentioned later in 3. Augmentation.

2. Model definition

The model has 4 convolution blocks with 32, 64, 128, and 64 channels, respectively. These each use batch normalization, ReLU, dropout and 2-pixel max pooling before an adaptive max pool and dense layers (with 64, 16, and 1 neuron, respectively). 

<ins>Model Architecture</ins>

|Stage|Structure|
|---|---|
|input | 1x132x1252|
|Conv(3x3) | 1x132x1252|
|Maxpool(2x2) | 32x66x626|
|Conv(3x3) | 64x66x626|
|Maxpool(2x2) | 64x33x313|
|Conv(3x3) | 128x33x313|
|Maxpool(2x2) | 128x16x156|
|Conv(1x1) | 64x16x156|
|AdaptiveMaxpool((1,1)) | 64x1x1|
|Dense | 64 |
|Dense | 16 |
|Dense | 1 |



The model was constructed using performance on the birdvox dataset to guide its design, mostly as an educational exercise, based loosely on the introduction to convolutional neural networks (CNN) described in the FastAI course, but the primary design goals were: 1) ensure the model can overfit a tiny subset of the birdvox dataset, 2) using a 80/20 split on the birdvox dataset, a ROCAUC of over 70% can be achieved, 3) the receptive field prior to adaptive max pooling is around 1 second (1200 frames), 4) runs on Kaggle hardware. The model has ____104,129___ parameters.

It is worth noting, however, that the Bulbul model by Grill, T. & Schlüter, J. ((2017) Two Convolutional Neural Networks for Bird Detection in Audio Signals. 25th European Signal Processing Conference (EUSIPCO2017). Kos, Greece.) had a very strong score on the datasets introduced above. Their model also used 4 convolution blocks, but uses a considerably narrower model of only 16 channels for each convolutional layer, and more aggressive 3x3 pooling, with a larger dense layer, for 373,169 parameters.

3. Augmentation

After finalizing the model, its performance on the 3-way cross validation was an average validation ROCAUC (vROCAUC) of 0.711. In the hopes of improving this score, several augmentations were made to the spectrograms either during data loading or during the training loop, as listed below:

a. time axis flipping
- A simple flip along the time axis
b. per-channel energy normalization (PCEN)
- Normalization of energy across each frequency
c. frequency masking
- Takes a random band of a random width (0.05 to 0.25% of the entire frequency axis) on the frequency dimension, and suppresses deviations from the mean by a given strength parameter.
d. time masking
- Takes a random band of a random width (0.05 to 0.25% of the entire time axis) on the time dimension, and suppresses deviations from the mean by a given strength parameter.
e. mixup
- Merges two spectrograms from the same batch, using the following rules. 1) if both samples are negative, result applies a random lambda. 2) if one is positive, the lambda for the positive spectrogram is no lower than 0.8 and the label is set to 1 ("hasbird"), 3) if both are positive, lambda is set to 1, and the second spectrogram is ignored. The rationale is 1) if there is no bird, a randomly merged noise profile may result in a useful non-bird call environment-like noise profile that the model can learn to ignore, 2) and 3) bird calls may get obscured by the noise profile of the paired spectrogram, and it may be unreasonable for a model to ascertain the presence of the bird call, so we should either heavily bias the positive spectrogram, or ignore the other entirely if both are positive.
f. white noise
- Applies a given strength of white noise to the spectrogram

___Augmented vROCAUC 0.728___

4. Training loop
5. Hyperparameter tuning
6. Production run
7. Analysis


