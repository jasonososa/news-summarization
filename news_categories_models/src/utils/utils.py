from joblib import dump
import pandas as pd
from sklearn import model_selection, metrics


def read_data(pth, list_col_to_keep):
    data = pd.read_csv(pth)
    return data[list_col_to_keep]

def split_train_test(X=None, y=None, random_state=6059, stratify=None, test_size=0.15):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        stratify=y,
        random_state=random_state,
        test_size=test_size
    )
    return X_train, X_test, y_train, y_test

def train_model(clf, X_train, y_train, X_test, y_test, parameters_dict, path_to_save_model):
    clf_grid = model_selection.GridSearchCV(estimator=clf, param_grid=parameters_dict, cv=3, scoring='f1_weighted', n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    predictions = clf_grid.predict(X_test)

    prec_weighted = metrics.precision_score(predictions, y_test, average='weighted')
    recall_weighted = metrics.recall_score(predictions, y_test, average='weighted')
    f1_weighted = metrics.f1_score(predictions, y_test, average='weighted')

    metrics_df = pd.DataFrame(
        {
            'weighted_precision': [prec_weighted],
            'weighted_recall': [recall_weighted],
            'weighted_f1': [f1_weighted]
        }
    )

    best_model_params = pd.DataFrame(
        data=clf_grid.best_params_,
        index=range(1)
    )

    metrics_df.to_csv(path_to_save_model+str(clf).replace('()', '')+'_'+'best_model_performance_metrics.csv', index=False)
    best_model_params.to_csv(path_to_save_model+str(clf).replace('()', '')+'_'+'best_model_params.csv', index=False)
    dump(clf_grid, path_to_save_model+str(clf).replace('()', '')+'_'+'best_model.joblib')
    return 