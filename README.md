# This is an in-progress project
# Get your news summary

### Idea:
#### Create an app that would output news summary from and inputed date and news category

### Approach:
#### - Part 1: Trained a model that take as inputs news articles headlines and classifies news articles into 10 different categories.This model will be use to find articles that will be summarize.
#### - Part 2: Since many newspapers cover the same stories, I will train a model that identifies similar news articles and only surface the different ones.
#### - Part 3: I will then train a model that will summarize different articles and surface them.

### Part 1:
Use dataset____ to trained a model that will classify news headlines into 10 different categories.
#### Baseline model:
Tf-Idf + bigrams followed by Naive Bayes, Logistic Regression, or Random Forest
#### Model Performance:

#### Conclusion:

#### Second model iteration:
Use spaCy pre-trained model to create vector representation of headlines. I will use the embeddings to then trained logistic regression, and Random Forest.
#### Model Performance:

#### Conclusion:

#### Third model iteration:
Use spaCy Text Categorizer to classify headlines into categories. The TextCategorizer is as explained on Spacy's website: "Stacked ensemble of a bag-of-words model and a neural network model. The neural network uses a CNN with mean pooling and attention. The “ngram_size” and “attr” arguments can be used to configure the feature extraction for the bag-of-words model."

#### Part 2:
