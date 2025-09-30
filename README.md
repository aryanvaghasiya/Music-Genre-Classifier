# Music Genre Recognition using GTZAN Dataset

This project develops an automated music genre recognition system using the GTZAN dataset. The primary goal is to classify music clips into one of 10 genres. Two main approaches are explored: a feature-based method using traditional machine learning models and deep neural networks, and a visual method using Convolutional Neural Networks (CNNs) on spectrograms.

##  Data Source

The project utilizes the **GTZAN music collection**, which includes:
- **Audio Recordings**: 1000 WAV files (30 seconds each) spanning 10 genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
- **Feature Data**: Pre-computed audio features (MFCCs, chroma, tempo, etc.) are available in `features_3_sec.csv` and `features_30_sec.csv`.

## Methodologies

Two primary methodologies were implemented and evaluated for genre classification:

### 1. Feature-Based Classification

This approach uses pre-computed audio features from the `features_3_sec.csv` dataset. The features are standardized and then used to train various classification models.

- **Traditional Machine Learning Models**:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier

- **Deep Learning Models**:
  - A series of fully-connected Dense Neural Networks (DNNs) with varying architectures, dropout rates, and optimizers were trained to find the best-performing model.

### 2. Visual-Based Classification (CNN)

This approach transforms audio signals into visual representations (Mel-spectrograms) and uses a Convolutional Neural Network (CNN) to classify them.

- **Spectrogram Generation**: Each 30-second audio clip is converted into a 128x128 Mel-spectrogram.
- **CNN Architecture**: A custom CNN with multiple convolutional, pooling, and dropout layers is trained on these spectrograms.

## Model Performance and Results

The following table summarizes the accuracy scores achieved by each model on the test set.

| Approach              | Model                       | Test Accuracy |
| --------------------- | --------------------------- | :-----------: |
| **Feature-Based**     | **Dense Neural Network (DNN)** | **92.01%**    |
|                       | Random Forest               | 90.00%        |
|                       | XGBoost                     | 89.00%        |
|                       | Support Vector Machine (SVM)| 76.00%        |
|                       | Logistic Regression         | 67.00%        |
| **Visual-Based**      | **Convolutional Neural Network (CNN)** | **71.31%**    |

### Key Observations:

- The **Dense Neural Network (DNN)** achieved the highest accuracy (**92.01%**), demonstrating its effectiveness in capturing complex patterns from the pre-computed audio features.
- Among the traditional models, **Random Forest** and **XGBoost** performed exceptionally well, achieving accuracies of **90.00%** and **89.00%**, respectively.
- The **CNN model** trained on spectrograms achieved a respectable accuracy of **71.31%**. While lower than the best feature-based models, this approach learns features directly from the audio's visual representation without manual feature engineering.
- The **Dense NN** showed strong performance for genres like `reggae`, `pop`, and `metal`, but had some confusion between `rock` and `metal`.
- The **CNN** struggled with genres that have overlapping features, such as `jazz` and `classical`, indicating that a more complex architecture or more extensive data augmentation could be beneficial.

## Technology Stack

- **Python**: Primary development language.
- **Keras/TensorFlow**: For building and training deep learning models.
- **Scikit-learn**: For data preprocessing, traditional model implementation, and performance evaluation.
- **XGBoost**: For implementing the gradient boosting model.
- **Librosa**: For audio signal processing and spectrogram generation.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For data visualization.

## Project Structure

```
├── data/
│   ├── features_3_sec.csv
│   ├── features_30_sec.csv
│   └── genres_original/
│       ├── blues/
│       ├── classical/
│       └── ... (other genres)
├── src/
│   ├── Music_Classfier.ipynb
│   └── Traditional_Models.ipynb
└── README.md
```

## How to Run

1.  **Clone the repository.**
2.  **Ensure the dataset is placed in the `data/` directory** as per the structure above.
3.  **Open the notebooks in the `src/` directory** using a Jupyter environment.
4.  **Install the required libraries** by running the `pip install` commands in the notebooks.
5.  **Execute the cells sequentially** to perform data analysis, model training, and evaluation.


## Contributions
Areen Vaghasiya
Aryan Vaghasiya