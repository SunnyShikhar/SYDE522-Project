from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import regression


def perform_svr_on_lbp():

    """This example generates the best performing SVR LBP results in Appendix A."""

    input_path = '../img.w.lbp.features.tab'
    output_path = '../test.new.code.csv'
    classifier = SVR()
    param_grid = {
        'C': [64],
        'gamma': [8, 16],
        'epsilon': [0.01],
        'kernel': ['rbf'],
    }
    regression.regress_lbp(input_path, classifier, param_grid, output_path)


def perform_svr_on_deep_features():

    """This example generates the best performing SVR ResNet results in Appendix A."""

    input_path = '../resnet.grouped.deep.features.p'
    output_path = '../test.new.code.csv'

    classifier = SVR()
    param_grid = {
        'C': [1, 5],
        'gamma': ['auto', 0.0001],
        'epsilon': [0.1, 0.15],
        'kernel': ['rbf'],
    }
    regression.regress_deep_features(input_path, classifier, param_grid, output_path)


def perform_rfr_on_lbp():

    """This example generates the best performing RFR LBP results in Appendix A."""

    input_path = '../img.w.lbp.features.tab'
    output_path = '../test.new.code.csv'

    classifier = RandomForestRegressor()
    param_grid = {
        'max_features': ['auto', 'log2'],
        'max_depth': [40, 60, 70, 90],
        'min_weight_fraction_leaf': [0.01]
    }
    regression.regress_lbp(input_path, classifier, param_grid, output_path)


def perform_rfr_on_deep_features():

    """This example generates the best performing RFR ResNet results in Appendix A."""

    input_path = '../resnet.grouped.deep.features.p'
    output_path = '../test.new.code.csv'

    classifier = RandomForestRegressor()
    param_grid = {
        'max_features': ['auto'],
        'max_depth': [30, 40, 70, 80],
        'min_weight_fraction_leaf': [0.01, 0.05]
    }
    regression.regress_deep_features(input_path, classifier, param_grid, output_path)


if __name__ == '__main__':

    perform_rfr_on_lbp()
    perform_rfr_on_deep_features()