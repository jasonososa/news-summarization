import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from src.utils.utils import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, metrics


def clean_stem_text(df, col_to_stem):
    """
    This function will process the raw text headlines and return headlines in 
    lowercase, without punctuation, with stemmed only of words that are 
    larger than 3 characters
    """
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    stemmed_tokens = []

    for row in df[col_to_stem]:
        tokens = word_tokenize(row)
        punc_stripped = [word.translate(table).lower() for word in tokens]
        alphabet_words = [word for word in punc_stripped if word.isalpha() and not word in stop_words and len(word) > 3]
        stemmed = [porter.stem(word) for word in alphabet_words]
        stemmed_tokens.append(stemmed)
    return [' '.join(stemmed_list) for stemmed_list in stemmed_tokens]

def tfidf_ngram(X_train, X_test, analyzer, ngram_range):
    """
    This function leverages TFidf vectorizer to output ngrams for model training
    """
    tfidf_vect_ngram = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    tfidf_vect_ngram.fit(X_train)
    X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
    X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)
    return X_train_tfidf_ngram, X_test_tfidf_ngram

def tfidf_workflow(pth, col_to_stem, path_to_save_model):
    data = read_data(pth, ['category', 'headline'])
    stemmed_sentences = clean_stem_text(data, col_to_stem)
    X_train, X_test, y_train, y_test = split_train_test(
            X=stemmed_sentences,
            y=data['category'],
            random_state=6059,
            stratify=data['category'],
            test_size=0.15
    )
    X_train_tfidf_ngram_word, X_test_tfidf_ngram_word = tfidf_ngram(X_train, X_test, analyzer='word', ngram_range=(2,2))
    model = train_model(
        naive_bayes.MultinomialNB(),
        X_train_tfidf_ngram_word,
        y_train,
        X_test_tfidf_ngram_word,
        y_test,
        dict(alpha=[0.5, 1]),
        path_to_save_model
    )


if __name__ == "__main__":
    pass