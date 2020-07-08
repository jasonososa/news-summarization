from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import seaborn as sns
from src.utils.utils import *
import string


def headline_corpus(df, filter_condition):
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    filter_condition_df = df[df['category'] == filter_condition]
    filter_condition_list = ' '.join(filter_condition_df['headline'].to_list())

    filter_condition_tokens = word_tokenize(filter_condition_list)
    filter_condition_punc_stripped = [word.translate(table).lower() for word in filter_condition_tokens]
    filter_condition_alphabet_words = [word for word in filter_condition_punc_stripped if word.isalpha() and not word in stop_words and len(word) > 3]

    return filter_condition_alphabet_words

def term_counter(corpus):
    counter_df = pd.DataFrame.from_dict(
        Counter(corpus), orient='index'
    ).reset_index().rename(columns={'index':'terms', 0:'counts'})
    return counter_df

def sns_barplot(x, y, data, title, x_label, y_label, category):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot = sns.barplot(x=x, y=y, data=data, color='gray', ax=ax, edgecolor='k')
    plt.title(title, fontsize=28)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./src/plots/'+category+'.png')
    return 

def plot_workflow(pth, category):
    data = read_data(pth, ['category', 'headline'])
    corpus = headline_corpus(data, category)
    counter_entertainment_terms = term_counter(corpus)
    sns_barplot(
        x='terms',
        y='counts', 
        data = counter_entertainment_terms.nlargest(10, columns='counts'),
        title= 'Top 10 Token Counts for '+category+'\nArticles Headlines',
        x_label='Words',
        y_label='Counts',
        category=category
    )


if __name__ == "__main__":
    pass