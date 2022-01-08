# Dataset

This file contains information about the dataset used for this project and how features are extracted and selected from it.

Table of contents:
- [Dataset](#dataset)
  - [Acquire Data](#acquire-data)
    - [Emo-Music Dataset](#emo-music-dataset)
  - [Feature Extraction](#feature-extraction)
    - [How to Run](#how-to-run)
  - [Feature Selection](#feature-selection)
    - [How to Run](#how-to-run-1)
- [Adding Labels](#adding-labels)

## Acquire Data

### Emo-Music Dataset

All credit is given to the authors of the following research paper:
[1000 songs for emotional analysis of music](https://dl.acm.org/doi/10.1145/2506364.2506365).

The dataset that was used in the project is called *Emotion in Music Database* and can be acquired [here](https://cvml.unige.ch/databases/emoMusic/).
It consists of 1000 songs gathered from [Free Music Archive](https://freemusicarchive.org/).
However, duplicates were found in the data, and after their removal, only 744 songs remain. All the songs have a sampling frequency of 44100Hz and are 45 seconds in length, but due to instability of the annotations for the first 15 seconds, only the last 30 seconds are considered.

The labeling used are arousal and valence. Arousal measures how excited/annoying versus how calm/sleepy a song is, while valence measures how pleasing/relaxing versus how sad/nervous a song is. The annotations were continuous throughout the song, and were done individually for arousal and valence. All annotations were provided on a scale from -1 to 1.

Along with the annotations, the standard deviation of the annotations were also reported. It was shown that the distribution of the annotations for each sample follows an approximate normal distribution. Thus, in an effort to remove outliers, only songs with all samples within the 99% confidence interval was used in this project.

## Feature Extraction

For the feature extraction we used the code provided from [Free Music Archive Github](https://github.com/mdeff/fma/tree/0ea2c9c83c84022fbf369e9dd258c7603baf33c4), which needed modifications to fit our application. All credit for the core implementation is given to the original authors.

The library used to extract interesting features is called [Librosa](https://librosa.org/).

The features extracted are from the following areas:
  - Chroma \
  Related to the 12 pitch classes.
  - Tonnetz \
  Computes the tonal centroid features.
  - RMS - Root Mean Square \
  Which is an indicator of the loudness in the music.
  - MFCC - Mel-Frequency Cepstral Coefficients \
  Which scales the signal to behave as in accordance with human hearing.
  - Spectral \
  Relates to analyses of different frequency spectrums.
  - Zero-Crossing Rate \
  Measures where signals cross between positive and negative.

And for each feature the following attributes are calculated:
  - Mean
  - Std
  - Skew
  - Kurtosis
  - Median
  - Min
  - Max

This will result in a dataframe with 518 features using the Emo-music dataset.

### How to Run

**Specify paths by modifying the following constants in `/src/api/feature_extraction.py`:**

- `SONGS_INFO_PATH` - path to `songs_info.csv` from the Emo-music dataset.

- `AUDIO_CLIP_DIR_PATH` - path to directory containing all songs (mp3-files).

*Note that all paths should be relative to the python working directory - which should be the root directory of the project if you follow the instructions bellow.*


**Execute the following command to run the feature extraction:** \
`python src/api/feature_extraction.py`

**The features will be outputted as:** \
`data/feature_extraction_<datetime>.csv`


## Feature Selection
Two types of feature selection methods are available at the moment; Principal Component Analysis (PCA) and Variance Threshold (VT).

More information can be found here:
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [VT](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)

### How to Run

*Before running feature selection, make sure that features extraction has been successfully performed.*

**Specify parameters by modifying the following constants in `/src/api/feature_selection.py`:**
- `FEATURES_LIBROSA_PATH` - path to `features_librosa_<datetime>.csv` from the feature extraction step.

- `TRVA_SIZE` - trva (training-validation) percentage of data \
Default: 0.8

- `VA_SIZE` - Validation percentage of trva \
Default: 0.25

- `PCA_PERCENT` - Percentage of variance for PCA selection \
Default: 0.99

- `VT_threshold` - Variance threshold for VT selection \
Default: 100

*Note that all paths should be relative to the python working directory - which should be the root directory of the project if you follow the instructions bellow.*

**Execute the following command to run the feature selection:** \
`python src/api/feature_selection.py`

**The features will be outputted as:** \
`data/features_librosa_<datasplit>_<method-name>.npy` (numpy file-format)

# Adding Labels
When both *features extraction* and *features selection* has been performed there should be outputted feature files in the `data/` directory. The directory must also contain labels from the Emo-music dataset - but these have to be added by yourself.

Copy the following files into the `data/` directory:
- `annotations/arousal_cont_average.csv`
- `annotations/arousal_cont_std.csv`
- `annotations/valence_cont_average.csv`
- `annotations/valence_cont_std.csv`
