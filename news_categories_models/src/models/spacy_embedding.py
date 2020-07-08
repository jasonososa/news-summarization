import pandas as pd
from sklearn.decomposition import PCA
from sklearn import model_selection, linear_model, metrics
import spacy.cli
spacy.cli.download("en_core_web_sm")
from src.utils.utils import *


def spacy_embedding_sm(df, col_to_embed):
    nlp_model = spacy.load('en_core_web_sm')
    embedding_list = []

    for row in df[col_to_embed]:
        trained_model = nlp_model(row)
        embedding_list.append(trained_model.vector)

    return embedding_list

def pca_decomposition(X_train, X_test):
    pca = PCA(n_components=96)
    pca.fit(X_train)
    transformed_vectors_X_train = pca.transform(X_train)
    transformed_vectors_X_test = pca.transform(X_test)

    return transformed_vectors_X_train, transformed_vectors_X_test

def spacy_embedding_workflow(pth, col_to_embed, path_to_save_model):
    data = read_data(pth, ['category', 'headline'])
    embedded_text = spacy_embedding_sm(data, col_to_embed)
    X_train, X_test, y_train, y_test = split_train_test(
            X=embedded_text,
            y=data['category'],
            random_state=6059,
            stratify=data['category'],
            test_size=0.15
    )
    pca_trans_X_train, pca_trans_X_test = pca_decomposition(X_train, X_test)
    model =  train_model(
        linear_model.LogisticRegression(),
        pca_trans_X_train,
        y_train,
        pca_trans_X_test,
        y_test,
        dict(C=[0.1, 1], penalty=['elasticnet'], l1_ratio=[0, 0.5], solver=['saga'],  max_iter=[1000]),
        path_to_save_model
    )


if __name__ == "__main__":
    pass