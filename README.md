# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: YARLAGADDA SIRI CHANDANA

INTERN ID: CT06DL1154

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:

This implements a complete pipeline for performing sentiment analysis on a dataset of movie reviews. The main goal is to classify each review as either positive or negative based on its textual content using classical NLP preprocessing, feature extraction, and machine learning.

Importing Libraries:

The code imports essential libraries for data manipulation (pandas, numpy), text processing (re, string), visualization (matplotlib, seaborn), machine learning (scikit-learn), and natural language processing (nltk).

Loading the Dataset:

The dataset is loaded from a CSV file located on the user’s machine (Movie_reviews.csv).
The dataset contains two columns:
        review: The raw text of the movie review.
        sentiment: The sentiment label (positive or negative).

Text Preprocessing:

A function clean_text is defined to clean the raw reviews:
   Convert text to lowercase for uniformity.
   Remove HTML tags using regex.
   Remove non-alphabetic characters (punctuation, numbers, symbols).
   Remove stopwords (common words like “the”, “is”) using NLTK’s stopwords list.
This cleaning helps reduce noise and focuses the model on meaningful words.
The cleaned reviews are saved in a new column cleaned_review.

Label Encoding:

The sentiment labels (positive and negative) are mapped to numerical labels (1 and 0 respectively) to prepare for classification.

Splitting Data into Training and Testing Sets:

The cleaned data is split into:
   Training set (80%) to train the model.
   Testing set (20%) to evaluate performance.
This is done with a fixed random seed for reproducibility.

Feature Extraction: TF-IDF Vectorization:

Text data must be converted into numerical features for machine learning.
The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer transforms the cleaned text into vectors reflecting the importance of words.
The vectorizer is fit on training data and then applied to the test data.
The maximum number of features is limited to 5,000 for efficiency.

Model Training: Logistic Regression:

A Logistic Regression classifier is initialized and trained on the TF-IDF vectors of the training data.
Logistic Regression is a commonly used, efficient model for binary classification problems like sentiment analysis.

Prediction and Evaluation:

The trained model predicts sentiment labels for the test set.
Evaluation metrics printed include:
Accuracy: The overall fraction of correct predictions.
Classification report: Precision, recall, F1-score, and support for each class.
A confusion matrix is computed and visualized using Seaborn heatmap to show true positives, true negatives, false positives, and false negatives, providing insight into types of errors.

Visualization:

The confusion matrix plot clearly shows model performance visually with labeled axes and a color gradient.

OUTPUT:
