# Evaluating Machine Learning Models for Multi-Class Hate Speech and Offensive Language Classification Using Natural Language Processing Techniques and Prototype Development

### Project Information

Name: Palla Vishwak Reddy

Project Type: Dissertation Project

Project Area: Natural Language Processing and Multi-Class Text
Classification

Primary Programming Language: Python

Primary Development Environment: Jupyter Notebook

Best Performing Model: Tuned LightGBM with TF-IDF

### Project Overview

This project develops and evaluates machine learning models for
multi-class hate speech and offensive language classification using
natural language processing techniques. The implementation investigates
two text feature extraction techniques, TF-IDF and Bag of Words, and
evaluates several machine learning algorithms under both default and
tuned configurations.

The project also addresses class imbalance using Synthetic Minority
Oversampling Technique (SMOTE). Model performance is assessed using
Accuracy, Precision, Recall, F1-Score and ROC-AUC. Hyperparameter
optimisation is performed using GridSearchCV.

The final evaluation identifies Tuned LightGBM using TF-IDF features as
the best-performing model. A Tkinter prototype is also implemented to
allow users to enter text and receive a predicted class and confidence
score. The prototype includes a separate human evaluation component for
collecting evaluation responses.

### Classification Classes

The target variable is the class column.

The prototype uses the following class mapping:

  Class   Meaning
  ------- --------------------
  0       Hate Speech
  1       Offensive Language
  2       Neutral Content

### Dataset

The implementation loads the dataset from:

labeled_data.csv

The dataset contains 24,783 observations and 7 original columns.

The original columns are:

  Column               Description
  -------------------- -----------------------------
  Unnamed: 0           Dataset index field
  count                Count-related dataset field
  hate_speech          Hate speech count
  offensive_language   Offensive language count
  neither              Neither-class count
  class                Target variable
  tweet                Text input

The target variable used for classification is class.

The notebook reports no missing values and no duplicate records in the
original dataset.

### Dataset Inspection Results

Number of observations: 24,783

Number of original columns: 7

Target variable: class

Unique target values: 0, 1 and 2

Missing values before preprocessing: 0

Duplicate records: 0

### Data Preprocessing

The text data is processed before feature extraction. The preprocessing
workflow includes:

1.  Conversion of text to lowercase.
2.  Removal of URLs.
3.  Removal of user mentions.
4.  Removal of the hashtag symbol.
5.  Removal of punctuation.
6.  Removal of numerical characters.
7.  Removal of extra whitespace.
8.  English stopword removal.
9.  Porter stemming.

The same general preprocessing logic is also implemented in the final
Tkinter prototype so that user-entered text can be transformed before
being passed to the saved TF-IDF vectorizer and LightGBM model.

### Exploratory Data Analysis

The notebook performs exploratory data analysis before model training.

The analysis includes:

1.  Target class distribution.
2.  Word count distribution.
3.  Character count distribution.
4.  Average word count by class.

These analyses are used to inspect the structure of the dataset and
understand the distribution of the text and target classes before model
development.

### Train Test Split

The dataset is divided into training and testing subsets using
stratified train_test_split.

Test size: 20 percent

Random state: 42

Stratification: Applied using the target variable

Training observations: 19,826

Testing observations: 4,957

The stratified split preserves the class distribution between the
training and testing data.

### Feature Extraction

Two feature extraction techniques are implemented.

### TF-IDF

Term Frequency-Inverse Document Frequency is implemented using
TfidfVectorizer.

Maximum number of features: 5,000

TF-IDF training matrix: 19,826 by 5,000

TF-IDF testing matrix: 4,957 by 5,000

The TF-IDF vectorizer is fitted using the training text and then used to
transform the testing text.

### Bag of Words

Bag of Words is implemented using CountVectorizer.

Maximum number of features: 5,000

BoW training matrix: 19,826 by 5,000

BoW testing matrix: 4,957 by 5,000

The Bag of Words representation provides a second feature extraction
approach for comparison with TF-IDF.

### Class Imbalance Handling

The notebook applies Synthetic Minority Oversampling Technique (SMOTE)
to the training feature matrices.

Before SMOTE, the training class distribution was:

Class 1: 15,352

Class 2: 3,330

Class 0: 1,144

After SMOTE, each class contains 15,352 training observations.

Total training observations after SMOTE: 46,056

SMOTE random state: 42

SMOTE is applied only to the training data. The testing data remains
separate for model evaluation.

### Machine Learning Models

The project evaluates the following machine learning algorithms:

1.  Logistic Regression
2.  LightGBM
3.  Decision Tree
4.  Multinomial Naive Bayes

Each model is evaluated using both TF-IDF and Bag of Words
representations.

The project therefore evaluates default and tuned configurations across
both feature extraction techniques.

### Evaluation Metrics

The following metrics are used throughout the project:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Weighted averaging is used for Precision, Recall and F1-Score.
Multi-class ROC-AUC is calculated using a one-vs-rest approach with
weighted averaging.

### Default Model Results

The following results are reported by the notebook.

  ---------------------------------------------------------------------------------
  Model         Technique     Accuracy   Precision     Recall   F1-Score    ROC-AUC
  ------------- ----------- ---------- ----------- ---------- ---------- ----------
  Logistic      TF-IDF          0.8477      0.8918     0.8477     0.8637     0.9301
  Regression                                                             

  Logistic      BoW             0.8521      0.8787     0.8521     0.8635     0.8989
  Regression                                                             

  LightGBM      TF-IDF          0.8838      0.9024     0.8838     0.8907     0.9477

  LightGBM      BoW             0.8759      0.8880     0.8759     0.8810     0.9170

  Decision Tree TF-IDF          0.8683      0.8850     0.8683     0.8755     0.8521

  Decision Tree BoW             0.7527      0.8669     0.7527     0.7974     0.8381

  Multinomial   TF-IDF          0.8104      0.8647     0.8104     0.8315     0.8992
  Naive Bayes                                                            

  Multinomial   BoW             0.8566      0.8778     0.8566     0.8657     0.8993
  Naive Bayes                                                            
  ---------------------------------------------------------------------------------

### Hyperparameter Optimisation

Hyperparameter optimisation is performed using GridSearchCV with
five-fold cross-validation.

### Tuned Logistic Regression

For TF-IDF Logistic Regression, the selected parameter was:

C = 10

Best cross-validation accuracy: 0.9235

For BoW Logistic Regression, the selected parameter was:

C = 1

Best cross-validation accuracy: 0.8382

### Tuned LightGBM

The LightGBM parameter grid contains:

n_estimators: 100 and 200

learning_rate: 0.05 and 0.1

Random state: 42

Five-fold cross-validation is used.

The best TF-IDF LightGBM parameters were:

learning_rate = 0.1

n_estimators = 200

Best cross-validation accuracy: 0.9445

The best BoW LightGBM parameters were:

learning_rate = 0.1

n_estimators = 200

Best cross-validation accuracy: 0.8400

### Tuned Decision Tree

For TF-IDF Decision Tree, the selected parameters were:

criterion = gini

max_depth = None

Best cross-validation accuracy: 0.9380

For BoW Decision Tree, the selected parameters were:

criterion = entropy

max_depth = None

Best cross-validation accuracy: 0.8254

### Tuned Multinomial Naive Bayes

For TF-IDF Multinomial Naive Bayes:

alpha = 0.1

Best cross-validation accuracy: 0.8778

For BoW Multinomial Naive Bayes:

alpha = 0.5

Best cross-validation accuracy: 0.7567

### Tuned Model Results

  ---------------------------------------------------------------------------------
  Model         Technique     Accuracy   Precision     Recall   F1-Score    ROC-AUC
  ------------- ----------- ---------- ----------- ---------- ---------- ----------
  Logistic      TF-IDF          0.8386      0.8784     0.8386     0.8547     0.9131
  Regression    Tuned                                                    

  Logistic      BoW Tuned       0.8521      0.8787     0.8521     0.8635     0.8989
  Regression                                                             

  LightGBM      TF-IDF          0.8897      0.8988     0.8897     0.8931     0.9449
                Tuned                                                    

  LightGBM      BoW Tuned       0.8606      0.8795     0.8606     0.8688     0.9100

  Decision Tree TF-IDF          0.8683      0.8850     0.8683     0.8755     0.8521
                Tuned                                                    

  Decision Tree BoW Tuned       0.7646      0.8725     0.7646     0.8064     0.8514

  Multinomial   TF-IDF          0.7991      0.8431     0.7991     0.8172     0.8662
  Naive Bayes   Tuned                                                    

  Multinomial   BoW Tuned       0.8552      0.8742     0.8552     0.8634     0.8964
  Naive Bayes                                                            
  ---------------------------------------------------------------------------------

### Best Performing Model

The notebook identifies Tuned LightGBM using TF-IDF as the
best-performing model.

Average Score: 0.9032

Accuracy: 0.8897

Precision: 0.8988

Recall: 0.8897

F1-Score: 0.8931

ROC-AUC: 0.9449

Best parameters:

learning_rate = 0.1

n_estimators = 200

Best cross-validation accuracy: 0.9445

The notebook selects this model because it achieved the highest average
score across the reported evaluation metrics and provided the strongest
overall balance among Accuracy, Precision, Recall, F1-Score and ROC-AUC.

### Model Comparison

The results show that LightGBM performs strongly with both feature
extraction techniques. The default TF-IDF LightGBM achieved an Accuracy
of 0.8838 and F1-Score of 0.8907. After tuning, the TF-IDF LightGBM
improved to an Accuracy of 0.8897 and F1-Score of 0.8931.

The tuned TF-IDF LightGBM also produced a ROC-AUC of 0.9449. This was
slightly below the default TF-IDF LightGBM ROC-AUC of 0.9477, but the
tuned model achieved the highest overall average score according to the
notebook's model selection procedure.

The TF-IDF representation generally produced stronger LightGBM results
than the corresponding Bag of Words representation.

### Classification Report of the Selected Model

For Tuned LightGBM using TF-IDF, the notebook reports the following
class-level results:

  Class     Precision   Recall   F1-Score   Support
  ------- ----------- -------- ---------- ---------
  0              0.40     0.48       0.44       286
  1              0.95     0.91       0.93     3,838
  2              0.82     0.93       0.87       833

Overall accuracy: 0.89

Weighted precision: 0.90

Weighted recall: 0.89

Weighted F1-Score: 0.89

### Prototype

A Tkinter graphical prototype is implemented for the selected
classification model.

The prototype is titled:

Prototype for Multi Class Classification and Human Evaluation

The prototype allows a user to enter text into a text input area and
obtain a classification result.

The prototype displays:

Predicted Class

Confidence

Status message

The class mapping used by the prototype is:

0 = Hate Speech

1 = Offensive Language

2 = Neutral Content

### Prototype Processing Workflow

The prototype follows the following workflow:

1.  The user enters text.
2.  The text is converted to lowercase.
3.  URLs are removed.
4.  User mentions are removed.
5.  Hashtag symbols are removed.
6.  Punctuation is removed.
7.  Numbers are removed.
8.  Extra whitespace is removed.
9.  English stopwords are removed.
10. Porter stemming is applied.
11. The processed text is transformed using the saved TF-IDF vectorizer.
12. The saved LightGBM model generates a prediction.
13. The model probability is used to calculate the displayed confidence.
14. The predicted class and confidence are displayed to the user.

### Prototype Interface Features

The Tkinter prototype contains:

Text input area

Prediction Result section

Predicted class display

Confidence display

Predict button

Human Evaluation button

Clear button

Exit button

Status bar

The main application can be resized and maximised.

### Human Evaluation

The prototype includes a separate Human Evaluation component.

The evaluation interface allows an evaluator to review the text and
model prediction and record:

Prediction Correct or Prediction Incorrect

Prediction Accuracy rating from 1 to 5

Ease of Use rating from 1 to 5

Clarity of Result rating from 1 to 5

Confidence in Result rating from 1 to 5

Open Comments

Overall Feedback

### Saved Model and Vectorizer

The prototype loads the saved model and TF-IDF vectorizer using joblib.

Model file:

multi_best_lightgbm_model.pkl

TF-IDF file:

multi_tfidf.pkl

The model and vectorizer must be placed in the same working directory as
the prototype code unless the file paths in the code are changed.

### Software and Libraries

The notebook uses Python and the following main libraries and packages:

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

LightGBM

NLTK

joblib

Tkinter

The project also uses the following scikit-learn components:

train_test_split

CountVectorizer

TfidfVectorizer

GridSearchCV

LogisticRegression

DecisionTreeClassifier

MultinomialNB

evaluation metrics from sklearn.metrics

### Running the Notebook

Place the following files in the working directory:

Palla Vishwak Reddy Final Dissertation Code.ipynb

labeled_data.csv

Run the notebook from the beginning to reproduce the data inspection,
preprocessing, exploratory analysis, feature extraction, SMOTE
processing, model training, model tuning and evaluation.

The notebook uses random_state = 42 for the main train-test split and
SMOTE procedure, supporting repeatable execution under the same
environment and data.

### Running the Prototype

Place these files together:

multi_best_lightgbm_model.pkl

multi_tfidf.pkl

The prototype Python file

Then run the prototype Python program.

The application opens a Tkinter window where text can be entered for
classification.

### Required Model Files

The prototype cannot make predictions if the saved model and vectorizer
files are missing.

The expected filenames are:

multi_best_lightgbm_model.pkl

multi_tfidf.pkl

These filenames are explicitly referenced in the prototype code.

### Reproducibility

The project uses fixed random states in the principal data splitting and
imbalance-handling steps.

Train-test split:

test_size = 0.2

random_state = 42

stratify = y

SMOTE:

random_state = 42

LightGBM:

random_state = 42

GridSearchCV:

cv = 5

scoring = accuracy

n_jobs = -1

### Project Workflow

The complete workflow can be summarised as:

Dataset loading

Data inspection

Missing-value and duplicate checking

Text preprocessing

Exploratory data analysis

Train-test splitting

TF-IDF feature extraction

Bag of Words feature extraction

SMOTE class balancing

Default model training

Default model evaluation

Comparative analysis

Hyperparameter tuning

Tuned model evaluation

Final model comparison

Best model selection

Prototype development

Human evaluation functionality

### Main Findings

The project demonstrates that the selected machine learning algorithms
can classify the three target classes using both TF-IDF and Bag of Words
representations.

The strongest overall model in the notebook is Tuned LightGBM with
TF-IDF.

The model achieved:

Accuracy: 88.97 percent

Precision: 89.88 percent

Recall: 88.97 percent

F1-Score: 89.31 percent

ROC-AUC: 94.49 percent

Average Score: 90.32 percent

The best LightGBM parameters were learning_rate = 0.1 and n_estimators =
200.


### Author

Name: Palla Vishwak Reddy


Techniques and Prototype Development
