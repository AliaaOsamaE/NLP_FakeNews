import pickle
import pandas as pd
from sklearn import metrics
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from textblob import Word
import gensim
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

Test_Data = pd.read_csv("G:\\NLP\\NLP_FakeNews\\news.csv")

with open('selected features.pkl', 'rb') as f:
    encoder = pickle.load(f)
    tfidf_vect = pickle.load(f)
    count_vectorizer = pickle.load(f)


with open('scaling.pkl', 'rb')as f:
    scaler = pickle.load(f)

with open('selected models.pkl', 'rb') as f:
    PassiveAggressive_TFIDF = pickle.load(f)
    LogisticRegression_TFIDF = pickle.load(f)
    NaiveBayesTFIDF = pickle.load(f)
    PassiveAggressiveWord2Vec = pickle.load(f)
    LogisticRegressionWord2Vec = pickle.load(f)
    NaiveBayesWord2Vec = pickle.load(f)
    PassiveAggressiveCountVectorizer = pickle.load(f)
    LogisticRegressionCountVectorizer = pickle.load(f)
    NaiveBayesCountVectorizer = pickle.load(f)


def tokenize_only(text):
    sentences = sent_tokenize(text)
    return sentences



def train_model(classifier, feature_vector_train, Y_actual, model_name, vector_name):
    predictions = classifier.predict(feature_vector_train)
    test_accuracy = metrics.accuracy_score(predictions, Y_actual)
    print(f"Accuracy of {model_name} using {vector_name} : {test_accuracy}\n")


def sentence_to_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


newsTextList = Test_Data['text'].apply(tokenize_only)

stop = stopwords.words('english')


Test_Data['text'] = Test_Data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
Test_Data['text'] = Test_Data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
Test_Data['text'] = Test_Data['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

Test_Data['title'] = Test_Data['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
Test_Data['title'] = Test_Data['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
Test_Data['title'] = Test_Data['title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


X = Test_Data['text']
Y = Test_Data["label"]

y_encoded = encoder.transform(Y)

x_after_tfidf = tfidf_vect.transform(X)  
Word2Vec_model = Word2Vec(X)
x_vector = [sentence_to_vector(sentence, Word2Vec_model) for sentence in X] 
X_count_vectorizer = count_vectorizer.transform(X)  


accuracy_PassiveAggressiveClassifier_TFIDF = train_model(PassiveAggressive_TFIDF,
                                                         x_after_tfidf, y_encoded, 'PassiveAggressiveClassifier', 'TF-IDF')


accuracy_LogisticRegression_TFIDF = train_model(LogisticRegression_TFIDF, x_after_tfidf,
                                                y_encoded, 'LogisticRegression', 'TF-IDF')


accuracy_MultinomialNB_TFIDF = train_model(NaiveBayesTFIDF, x_after_tfidf, y_encoded, 'MultinomialNB', 'TF-IDF')

accuracy_PassiveAggressiveClassifier_Word2Vec = train_model(PassiveAggressiveWord2Vec,
                                                            x_vector, y_encoded,
                                                            'PassiveAggressiveClassifier', 'Word2Vec')

accuracy_LogisticRegression_Word2Vec = train_model(LogisticRegressionWord2Vec, x_vector,
                                                   y_encoded, 'LogisticRegression', 'Word2Vec')


#scaling
x_woerdtovec_scaled = scaler.transform(x_vector)

accuracy_MultinomialNB_Word2Vec = train_model(NaiveBayesWord2Vec, x_woerdtovec_scaled,
                                              y_encoded, 'MultinomialNB', 'Word2Vec')

accuracy_PassiveAggressiveClassifier_CountVectorizer = train_model(PassiveAggressiveCountVectorizer,
                                                                   X_count_vectorizer, y_encoded,
                                                                   'PassiveAggressiveClassifier', 'CountVectorizer')


accuracy_LogisticRegression_CountVectorizer = train_model(LogisticRegressionCountVectorizer, X_count_vectorizer,
                                                          y_encoded,
                                                          'LogisticRegression', 'CountVectorizer')

accuracy_MultinomialNB_CountVectorizer = train_model(NaiveBayesCountVectorizer, X_count_vectorizer,
                                                     y_encoded, 'MultinomialNB', 'CountVectorizer')