import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
log_name = input("Enter log file name: ")
handler = logging.FileHandler('../' + log_name + '.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


def gridsearch(classifier, param_grid, X_train, y_train,
               X_validation=None, y_validation=None, scorer='spearman'):

    """
    Perform Gridsearch to determine optimal classifier parameters.
    :param classifier: sklearn classifier object
    :param param_grid: parameters to search
    :param X_train: Features to learn beauty scores from
    :param y_train: Beauty scores to learn
    :param X_validation: Features to predict beauty scores off of
    :param y_validation: Beauty scores to compare predictions to
    :param scorer: either 'spearman' or 'r2'
    :return: (np.array) Beauty scores predicted from X_validation features
    """

    if scorer == 'spearman':
        score_func = make_scorer(lambda truth, predictions: spearmanr(truth, predictions)[0],
                                 greater_is_better=True)
    elif scorer == 'r2':
        score_func = 'r2'
    else:
        raise ValueError("Invalid scoring function. Must be either 'r2' or 'spearman'.")

    print("Peforming GridSearch...")
    classifier = GridSearchCV(classifier, param_grid, cv=5, scoring=score_func, verbose=3)
    classifier_fit = classifier.fit(X_train, y_train)
    print("Completed GridSearch.")

    # Log the params of the best fit
    logger.info("Completed GridSearch. Writing best SVR params and score to log.")
    logger.info(classifier_fit.best_params_)

    # Log the score of the best fit
    print("Best Score: " + str(classifier_fit.best_score_))
    logger.info("Best Score: " + str(classifier_fit.best_score_))

    # Use the best fit to predict the beauty scores of the test set
    if X_validation is not None and y_validation is not None:
        y_validation_pred = classifier_fit.predict(X_validation)
        logger.info("Validation R^2: " + str(r2_score(y_true=y_validation, y_pred=y_validation_pred)))
        logger.info("Spearman Rank Coefficient: " + str(spearmanr(y_validation_pred, y_validation)))
        print("Spearman Rank Coefficient: " + str(spearmanr(y_validation_pred, y_validation)))

    return y_validation_pred


def svr_on_lbp(output_path):

    """
    Perform Support Vector Regression on LBP features.

    Writes out the LBP features dataframe with two new columns:
        - avg_beauty_score: The mean of the beauty scores available for each image
        - predicted_score: Predicted beauty scores (only for the samples allocated to the validation set).

    :param output_path: Path to save results dataframe.
    :return: None
    """

    logger.info("\n\n\n\t\t\t*****\t\t\tWelcome to: SVR on LBP.\t\t\t*****\t\t\t\n\n\n")

    # Load the LBP Feautures
    df = pd.read_csv('../img.w.lbp.features.tab', sep="\t", index_col=False,
                     converters={"lbp_feature": lambda x: ([float(y) for y in x.strip("[]").split(", ")])})

    # Take the average of the beauty scores (this is the target variable)
    df['avg_beauty_score'] = df.beauty_scores.apply(lambda x: np.mean(np.asarray([int(i) for i in x.split(',')])))
    df['predicted_score'] = None

    for group, data in df.groupby(['category']):

        print("Training SVR models for category: " + str(group))
        logger.info("Training SVR models for category: " + str(group))

        X = pd.DataFrame(data.lbp_feature.tolist())
        y = pd.DataFrame(data.avg_beauty_score)

        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)

        pred = perform_svr(X_train, y_train, X_test, y_test)

        df.ix[y_test.index.values, 'predicted_score'] = pred
        df.to_csv(output_path)


def perform_svr(X_train, y_train, X_validation, y_validation):

    """
    Perform Support Vector Regression on the input datasets.

    The model will be trained using X_train and y_train (using GridsearchCV with cross-validation).
    The model will be used to predict beauty scores for the images in X_validation.

    :param X_train: Features to learn beauty scores from
    :param y_train: Average beauty scores for images represented in X_train
    :param X_validation: Features to predict beauty scores on
    :param y_validation: Average beauty scores for images represented in X_train
    :return: Predicted beauty scores for data in X_validation
    """

    # Do SVR
    svr = SVR()

    Cs = np.arange(5., 7.)
    Cs = np.array([2. ** x for x in Cs])

    gammas = np.arange(3., 5.)
    gammas = np.array([2. ** x for x in gammas])

    param_grid = {
        'C' : Cs,
        'gamma': gammas,
        'epsilon': np.linspace(0.01, 0.5, 2),
        'kernel': ['rbf'],
    }

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation).ravel()

    pred = gridsearch(svr, param_grid, X_train, y_train,
                      X_validation, y_validation)

    return pred


if __name__ == '__main__':

    output_path = '../'
    feature_file_name = input("Enter file name to save features to: ")
    svr_on_lbp(output_path + feature_file_name + '.csv')