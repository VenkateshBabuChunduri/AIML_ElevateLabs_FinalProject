# AI/ML Internship - Project 6: News Article Classification (Fake/Real)

## Objective
* The objective of this project was to build a machine learning model capable of classifying news articles as either "fake" or "real" using Natural Language Processing (NLP) techniques.
* This project covers data preprocessing, text vectorization, model training, and evaluation, providing a solid foundation in text classification.

## Dataset
* The dataset used for this project is the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.
* It consists of two separate CSV files:
    * `Fake.csv`: Contains news articles labeled as fake.
    * `True.csv`: Contains news articles labeled as true.

## Tools and Libraries Used
* **Python**
* **Pandas:** For data loading, manipulation, and combination.
* **NLTK (Natural Language Toolkit):** For text preprocessing tasks like stop word removal.
* **Scikit-learn:** For text vectorization (TF-IDF), data splitting, and machine learning model (Logistic Regression) implementation, and evaluation metrics.
* **Matplotlib:** For creating static visualizations (Confusion Matrix).
* **Seaborn:** For enhanced statistical graphics (Confusion Matrix heatmap).
* **Numpy:** For numerical operations.
* **`pickle`:** For saving and loading trained models and vectorizers.
* **Streamlit:** For building the interactive web application (live demo).

## Project Steps Performed:

### 1. Load and Combine Dataset
* Loaded `Fake.csv` and `True.csv` into separate DataFrames.
* Assigned a numerical label (`0` for fake, `1` for true) to each DataFrame.
* Combined the two DataFrames into a single comprehensive dataset (`df_news`).
* Shuffled the combined dataset to ensure a random mix of fake and real news, crucial for unbiased model training.
* Inspected the dataset for basic information (shape, columns) and confirmed no missing values.
* **Outcome:** A clean, combined, and shuffled dataset was prepared, ready for text preprocessing.

### 2. Clean Text using NLTK
* Combined the `title` and `text` columns into a single `full_text` column to capture all textual information.
* Implemented a text cleaning function to preprocess the `full_text` column, creating a `cleaned_text` column. The cleaning steps included:
    * Converting text to lowercase.
    * Removing URLs, HTML tags, punctuation, and special characters.
    * Removing words containing numbers.
    * Removing common English stop words.
* *(Note: Stemming was intentionally omitted due to NLTK package compatibility, but the remaining cleaning steps are highly effective.)*
* **Outcome:** The raw text data was transformed into a cleaner, more consistent format, removing noise and preparing it for numerical conversion.

### 3. Vectorize with TF-IDF
* Utilized `TfidfVectorizer` from `scikit-learn` to convert the `cleaned_text` into numerical feature vectors.
* Configured the vectorizer to consider the top 10,000 most relevant features (`max_features=10000`) and ignore words appearing in fewer than 5 documents (`min_df=5`).
* **Outcome:** The text data was successfully transformed into a sparse matrix of TF-IDF features (`X_tfidf`) with a shape of `(44898, 10000)`, along with the corresponding labels (`y`).

### 4. Train Logistic Regression Model
* Split the `X_tfidf` features and `y` labels into training and testing sets (80% train, 20% test) using `train_test_split`.
* The `stratify=y` parameter ensured that the class distribution (fake/real) was maintained in both sets.
* Initialized and trained a `LogisticRegression` model on the training data.
* Logistic Regression is a robust linear classifier well-suited for binary classification tasks.
* **Outcome:** A Logistic Regression model was successfully trained and is now ready to make predictions and be evaluated.

### 5. Evaluate Model Metrics
* Used the trained Logistic Regression model to make predictions (`y_pred`) on the unseen test set (`X_test`).
* Calculated and reported several key evaluation metrics:
    * **Accuracy:** Overall correctness of predictions (~98.89%).
    * **Precision:** Proportion of correctly predicted positives among all predicted positives (~0.99 for both classes).
    * **Recall:** Proportion of actual positives correctly identified (~0.99 for both classes).
    * **F1-Score:** Harmonic mean of precision and recall (~0.99 for both classes).
    * **Confusion Matrix:** A table summarizing true positives, true negatives, false positives, and false negatives, visually confirming high performance.
* **Outcome:** The model demonstrated exceptionally high accuracy and robust performance, with very few misclassifications, confirming its effectiveness in distinguishing between fake and real news.

### 6. Save Trained Model and Vectorizer
* Saved the trained `LogisticRegression` model (`logistic_regression_model.pkl`) and the fitted `TfidfVectorizer` (`tfidf_vectorizer.pkl`) using Python's `pickle` library.
* **Outcome:** These serialized objects are now ready to be loaded by the Streamlit application for real-time predictions without needing to retrain the model.

### 7. Create Streamlit Interface for Input and Display Prediction/Explanation
* Developed a `fake_news_app.py` script to create an interactive web application using Streamlit.
* The app loads the saved model and vectorizer.
* It provides a user-friendly interface with a text area for users to input news articles.
* Upon classification, it clearly displays the prediction (FAKE or TRUE), the model's confidence, and a brief explanation of the prediction process.
* **Outcome:** A functional and user-friendly web demo was successfully deployed locally, fulfilling the project's interactive deliverable.

## Visualizations
* The repository includes the following generated plot:
    * `logistic_regression_confusion_matrix.png`: A heatmap visualization of the confusion matrix, illustrating the model's prediction accuracy.

## Conclusion
* This project successfully developed a robust and highly accurate Fake News Classification system.
* By meticulously following a standard machine learning workflow—encompassing data preparation, text preprocessing, feature engineering, model training, evaluation, and deployment—a Logistic Regression model was developed that can reliably distinguish between fake and real news articles.
* The impressive accuracy (approx. 98.89%) and strong performance metrics highlight the model's efficacy.
* The creation of an interactive Streamlit application demonstrates practical deployment skills, making the model accessible and usable.
* This project provides valuable experience in Natural Language Processing and supervised learning, crucial for understanding and combating misinformation in digital media.
