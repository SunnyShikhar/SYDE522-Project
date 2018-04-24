import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV


def gridsearch(classifier, param_grid, X_train, y_train, X_validation, y_validation):

    """Helper function to perform GridSearchCV on the input classifier and param grid."""

    print("Param Grid: " + str(param_grid))

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation).ravel()

    # Define the Spearman correlation scoring function
    score_func = make_scorer(lambda truth, predictions: spearmanr(truth, predictions)[0],
                             greater_is_better=True)

    print("Starting Gridsearch...")
    # Train the classifier on the training data using GridSearch
    classifier = GridSearchCV(classifier, param_grid, cv=5, scoring=score_func, verbose=0)
    classifier_fit = classifier.fit(X_train, y_train)
    print("Completed Gridsearch.")

    # Print the best parameters and results on the training data
    print('Best Params:' + str(classifier_fit.best_params_))
    print("Best Score: " + str(classifier_fit.best_score_))

    # Use the best fit to predict the beauty scores of the test set
    y_validation_pred = classifier_fit.predict(X_validation)
    print("Validation R^2: " + str(r2_score(y_true=y_validation, y_pred=y_validation_pred)))
    print("Spearman Rank Coefficient: " + str(spearmanr(y_validation_pred, y_validation)))

    return y_validation_pred


def regress_lbp(input_path, classifier, param_grid, output_path):

    """Perform GridSearch using the input classifier and param grid to predict beauty scores using LBP features."""

    # Load the LBP features
    df = pd.read_csv(input_path, sep="\t", index_col=False,
                     converters={"lbp_feature":
                                     lambda x: ([float(i) for i in x.strip("[]").split(", ")])})

    # Take the average of the beauty scores (this is the target variable)
    df['avg_beauty_score'] = df.beauty_scores.apply(
        lambda x: np.mean(np.asarray([int(i) for i in x.split(',')]))
    )
    df['predicted_score'] = None

    for group, data in df.groupby(['category']):
        print("\n\t\tTraining SVR models for category: " + str(group) + "\n")

        X = pd.DataFrame(data.lbp_feature.tolist())
        y = pd.DataFrame(data.avg_beauty_score)

        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)

        pred = gridsearch(classifier, param_grid, X_train, y_train, X_test, y_test)

        df.ix[y_test.index.values, 'predicted_score'] = pred

    # Write results to CSV
    df.to_csv(output_path)



def regress_deep_features(input_path, classifier, param_grid, output_path):

    """Perform GridSearch using the input classifier and param grid to predict beauty scores using deep features."""

    # Load the deep features
    # These were pickled because the vectors were too big to save to CSV.
    df = pd.read_pickle(input_path)

    # Take the average of the beauty scores (this is the target variable)
    df['avg_beauty_score'] = df.beauty_scores.apply(
        lambda x: np.mean(np.asarray([int(i) for i in x.split(',')]))
    )
    df['predicted_score'] = None

    # Remove any invalid deep features
    df = df[df.deep_feature.notnull()]
    df['valid_deep_feature'] = df.deep_feature.apply(
        lambda x: not np.any(np.isnan(x)) and np.all(np.isfinite(x)))
    df = df[df.valid_deep_feature]

    for group, data in df.groupby(['category']):
        print("\n\t\tTraining SVR models for category: " + str(group) + "\n")

        # For deep features, training and test datasets were manually labelled
        # to make sure that they matched those generated in LBP. This was necessary
        # because deep feature extraction failed for 1 image in Inception and for
        # 35 images in ResNet.
        X_train = pd.DataFrame(data[data.group == 'Train'].deep_feature.tolist())
        X_test = pd.DataFrame(data[data.group == 'Test'].deep_feature.tolist())
        y_train = pd.DataFrame(data[data.group == 'Train'].avg_beauty_score)
        y_test = pd.DataFrame(data[data.group == 'Test'].avg_beauty_score)

        print("Reducing dimensionality with PCA.")
        pca = PCA(0.95, random_state=42)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        print("PCA Reduce Dimensionality to: " + str(X_train.shape[1]) + " components.")
        X_test = pca.transform(X_test)

        pred = gridsearch(classifier, param_grid, X_train, y_train, X_test, y_test)

        df.ix[y_test.index.values, 'predicted_score'] = pred

    # Write results to CSV
    df.to_csv(output_path)