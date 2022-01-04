# Dataset

This file has information about the dataset used in this project and how features are extracted and selected.

Table of contents:
- [Dataset](#dataset)
  - [Acquire Data](#acquire-data)
    - [Emo-Music Dataset](#emo-music-dataset)
  - [Feature Extraction](#feature-extraction)
    - [How to Run](#how-to-run)
  - [Feature Selection](#feature-selection)
    - [How to Run](#how-to-run-1)

## Acquire Data

### Emo-Music Dataset

All credit are given to the authors of the following research paper:
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

**Specify paths in `/src/api/feature_extraction.py`:**
All paths are specified from root.

- Path to `songs_info.csv` from the Emo-music dataset - located in `main()` \
`filename = Path('data/emo-music-features/annotations/songs_info.csv')`

- Path to directory containing all songs (mp3-files) - located in `save_npy()` \
`filedir = Path('data/emo-music-features/clips_45sec/clips_45seconds')`

**Navigate to the folder `/src/api/` from a terminal.** \
Execute the following command: \
`python .\feature_extraction.py`

**The Output-File will be located in:** \
`data/` with the name `feature_extraction_<datetime>.csv`


## Feature Selection
Two types of feature selection methods are available at the moment; Principal Component Analysis (PCA) and Variance Threshold (VT).

More information can be found here:
- [PCA]([link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))
- [VT](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)

### How to Run

**Specify path in `/src/api/feature_selection.py`:**
- Path (specified from root) to `features_librosa_<datetime>.csv` from the feature extraction step - located in `main()` \
`filepath = Path("./../data/features_librosa.csv")`

**Specify input parameters:**
- trva_size - trva (training-validation) percentage of data \
Default: 0.8

- va_size - Validation percentage of trva \
Default: 0.25

- method - Enum which dictates what feature selection to perform \
Default: Method.PCA

- pca_percent - Percentage of variance if PCA method is selected \
Default: 0.99

- threshold - Variance threshold if VT method is selected \
Default: 100

**Navigate to the folder `/src/api/` from a terminal.** \
Execute the following command: \
`python .\feature_selection.py`

**The output-file will be located in** \
the same directory as the file `features_librosa_<datetime>.csv` is located in \
The files will have to following name: \
`features_librosa_<datasplit>_<method-name>.npy` (numpy file-format)


