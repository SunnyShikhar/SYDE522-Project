import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
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
    classifier = GridSearchCV(classifier, param_grid, cv=2, scoring=score_func, verbose=3)
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

    # Once everything is done
    df.to_csv(output_path)


def svr_on_deep_features(output_path, reduce=True):

    logger.info("\n\n\n\t\t\t*****\t\t\tWelcome to: SVR on Deep Features.\t\t\t*****\t\t\t\n\n\n")

    # Load the LBP Feautures
    df = pd.read_pickle('../resnet.grouped.deep.features.p')

    # Take the average of the beauty scores (this is the target variable)
    df['avg_beauty_score'] = df.beauty_scores.apply(lambda x: np.mean(np.asarray([int(i) for i in x.split(',')])))
    df['predicted_score'] = None

    df = df[df.deep_feature.notnull()]

    df['valid_deep_feature'] = df.deep_feature.apply(
        lambda x: not np.any(np.isnan(x)) and np.all(np.isfinite(x)))

    # Remove any invalid deep features
    df = df[df.valid_deep_feature]

    grouped = True
    if grouped:
        for group, data in df.groupby(['category']):

            print("Training SVR models for category: " + str(group))
            logger.info("Training SVR models for category: " + str(group))

            X_train = pd.DataFrame(data[data.group == 'Train'].deep_feature.tolist())
            X_test = pd.DataFrame(data[data.group == 'Test'].deep_feature.tolist())

            y_train = pd.DataFrame(data[data.group == 'Train'].avg_beauty_score)
            y_test = pd.DataFrame(data[data.group == 'Test'].avg_beauty_score)

            if reduce:
                print("Reducing dimensionality with PCA.")
                logger.info("Reducing dimensionality with PCA.")
                pca = PCA(0.95, random_state=42)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                logger.info("PCA Reduce Dimensionality to: " + str(X_train.shape[1]) + " components.")
                X_test = pca.transform(X_test)
                print(str(X_train.shape))

            pred = perform_svr(X_train, y_train, X_test, y_test)

            df.ix[y_test.index.values, 'predicted_score'] = pred

    else:
        print("Training SVR models for all data.")
        logger.info("Training SVR models for all data.")

        X_train = pd.DataFrame(df[df.group == 'Train'].deep_feature.tolist())
        X_test = pd.DataFrame(df[df.group == 'Test'].deep_feature.tolist())

        y_train = pd.DataFrame(df[df.group == 'Train'].avg_beauty_score)
        y_test = pd.DataFrame(df[df.group == 'Test'].avg_beauty_score)

        if reduce:
            print("Reducing dimensionality with PCA.")
            logger.info("Reducing dimensionality with PCA.")
            pca = PCA(0.95, random_state=42)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            logger.info("PCA Reduce Dimensionality to: " + str(X_train.shape[1]) + " components.")
            X_test = pca.transform(X_test)
            print(str(X_train.shape))

        pred = perform_svr(X_train, y_train, X_test, y_test)

        df.ix[y_test.index.values, 'predicted_score'] = pred

    # Write results to CSV
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

    param_grid = {
        #'C' : [1, 5, 10, 25],
        'C': [48, 64, 128],
        #'gamma': ['auto', 0.0001, 0.01, 0.1],
        'gamma': ['auto', 0.001],
        'epsilon': [0.01, 0.05, 0.1],
        'kernel': ['linear', 'rbf'],
    }

    logger.info("Param Grid: " + str(param_grid))

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation).ravel()

    pred = gridsearch(svr, param_grid, X_train, y_train,
                      X_validation, y_validation)

    return pred


def perform_rfr(X_train, y_train, X_validation, y_validation):

    rfr = RandomForestRegressor()

    param_grid = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': np.arange(10, 110, 10),
        'min_weight_fraction_leaf': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    logger.info("Param Grid: " + str(param_grid))

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation).ravel()

    pred = gridsearch(rfr, param_grid, X_train, y_train,
                      X_validation, y_validation)

    return pred


def rfr_on_lbp(output_path):

    """
    Perform Random Forest Regression on LBP features.

    Writes out the LBP features dataframe with two new columns:
        - avg_beauty_score: The mean of the beauty scores available for each image
        - predicted_score: Predicted beauty scores (only for the samples allocated to the validation set).

    :param output_path: Path to save results dataframe.
    :return: None
    """

    logger.info("\n\n\n\t\t\t*****\t\t\tWelcome to: RFR on LBP.\t\t\t*****\t\t\t\n\n\n")

    # Load the LBP Feautures
    df = pd.read_csv('../img.w.lbp.features.tab', sep="\t", index_col=False,
                     converters={"lbp_feature": lambda x: ([float(y) for y in x.strip("[]").split(", ")])})

    # Take the average of the beauty scores (this is the target variable)
    df['avg_beauty_score'] = df.beauty_scores.apply(lambda x: np.mean(np.asarray([int(i) for i in x.split(',')])))
    df['predicted_score'] = None

    for group, data in df.groupby(['category']):

        print("Training RFR models for category: " + str(group))
        logger.info("Training RFR models for category: " + str(group))

        X = pd.DataFrame(data.lbp_feature.tolist())
        y = pd.DataFrame(data.avg_beauty_score)

        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)

        pred = perform_rfr(X_train, y_train, X_test, y_test)

        df.ix[y_test.index.values, 'predicted_score'] = pred

    # Print to CSV at the end
    df.to_csv(output_path)

if __name__ == '__main__':

    output_path = '../'
    feature_file_name = input("Enter file name to save features to: ")
    # rfr_on_lbp(output_path + feature_file_name + '.csv')
    # svr_on_deep_features(output_path + feature_file_name + '.csv')
    svr_on_lbp(output_path + feature_file_name + '.csv')
