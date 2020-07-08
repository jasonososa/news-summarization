# This is an in-progress project
# Title: Get your news summary

# Main project: Create an app that would output news summary from an input date and news category
## Approach:

### Part 1: Trained a model that take as inputs news articles headlines and classifies news articles into 10 different categories. This model will then be used to classify noncategorize news headlines form a different dataset.

### Part 2: Since many newspapers cover the same stories, I will train a model that identifies similar news articles and only surface the different ones.

### Part 3: I will then train a model that summarizes different articles and surface the summary.

## Results:
### Part 1:
Baseline Model:
* Tf-Idf + bigrams followed by multinomial Naive Bayes. I am interested on a model that performs well for both precision and recall. Therefore the harmonic mean of precision and recall or the f1 score is a good metric for measuring model performance.
    * Model performance:
        * For the purpose of fast iteration I used only 10% of the data. With the smaller dataset, model performance was not great with the following metrics:
         * precision = 69.7%
         * recall = 33.0%clear
         * f1 score = 37.0%
    * Metrics with the whole dataset were about the same
Conclusion:
* f1 score for this model was mediocre. I did try other models (Logistic Regression, Random Forest) during exploration using a notebook and Naive Bayes performed about the same or even better.

Second model iteration:
* Use spaCy pretrained model to create an embedding representation of headlines. Decrease multicolinearity of the embeddings with PCA and then train a logistic regression. As before I'd pay attention to f1 score.
Model performance:
   * For fast iteration I used the small Spacy pretrained model and only 10% of the data. The small pretrained model provides an embedding of 96 features which should give a decent performance. Model performance was better than with the simpler ngrams. However, it is still not great:
      * precision = 95.6%
      *  recall = 34.6%
      *  f1 score = 50.6%

Conclusion:
* f1 score for this model was quite low and not acceptable. I did try other models (Random Forest) during exploration using a notebook and logistic regression performed slightly better. It is worth mentioning that precision is quite high, where are recall is very low. This indicates that there is a large number of false negative and point to a potential mistake in re-consolidating the labels.

Next steps:
* Get more compute power and train models using whole dataset
* Re-consolidate labels using a similarity metric like cosine similarity. It is very likely that my guesses are wrong.
* Create embeddings using more complex model embeddings. I'd specifically be interested on using Bert, since it performs quite well for text classification.
* Use a more complex classification models such as neural networks. This will be great to account for not linear interactions of the features.

# How to use:

cd into the project directory

In the command line run: 
    pip3 install -r dependencies.txt

To run the project, type the following in the command line:
    python3 main.py

The project can be run with different arguments by specifying the parameters whie running main.py. For example:
    python3 main.py --category ENTERTAINMENT
