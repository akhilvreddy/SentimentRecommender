import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv("/Users/akhilreddy/Documents/VSCode/SentimentRecommender/Training Data/Sentiment Dataset.csv")

# Drop NA data for easy processing
data = data.dropna()

# Change all numbers to integers for MultinomialNB (multiply everything by 2)
data['Value'] = data['Value'].map({0.0: 0, 0.5: 1, 1.0: 2})

# Split the dataset into training and testing sets
X = data['Sentence']
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Predict class probabilities instead of class labels
y_pred_proba = clf.predict_proba(X_test_tfidf)

# Extract the probabilities for positive (class 2) sentiment
positive_probabilities = y_pred_proba[:, 2]

# Add a measure of the sentence length (number of words) as the second number
# Assuming each sentence is separated by space
sentence_lengths = [len(sentence.split()) for sentence in X_test]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)

# Return two numbers: positive sentiment probability and sentence length
# return positive_probabilities, sentence_lengths
print(positive_probabilities)
print(sentence_lengths)

# Sample sentence for testing
sample_sentence = "I love love love today it is such a good day I love it."

# Preprocess the sample sentence
sample_sentence_tfidf = vectorizer.transform([sample_sentence])

# Predict sentiment probabilities for the sample sentence
sample_positive_prob = clf.predict_proba(sample_sentence_tfidf)[0, 2]
sample_sentence_length = len(sample_sentence.split())

# write a function to send to 
def my_function(fname):
    sample_positive_prob = clf.predict_proba(sample_sentence_tfidf)[0, 2]
    sample_sentence_length = len(sample_sentence.split())

    return (sample_positive_prob, sample_sentence_length)

# print(f"Sample Sentence: {sample_sentence}")
# print(f"Sample Positive Sentiment Probability: {sample_positive_prob}")
# print(f"Sample Sentence Length: {sample_sentence_length} words")