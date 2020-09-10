#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jul 21

@author: Sebastian Gonzalez
"""

####################################
### Neccessary Import Statements ###
####################################
# Data Manipulation
import numpy as np

# Model Classes
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

# Model Evaluation Tools
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import scipy.stats as stats

# Project Tools
from src.data_scripts.feature_engineering import bow_data_loader
from src.model_scripts import model_evaluation as me


####################################
### Define our Modular Functions ###
####################################
def label_transformer(labels_arr: np.array):
    """
    Purpose
    -------
    The purpose of this function is to take an unmodified labels array
    (where all of the entries are class labels as strings) and map
    each unique element to a numerical value to end up with a numerical
    labels array that is more sutible for Sklearn classification models.

    Parameters
    ----------
    labels_arr : Numpy Array
        This array is a collection of the class labels (as strings) for
        all of the training instances that we are working with.

    Returns
    -------
    to_return : (dict, Numpy Array)
        This function returns a tuple which contains a dictionary and a
        Numpy array of integer values. The dictionary is a mapper whose
        keys are the old string labels and the values that they point to
        are the numerical values that each instance of them in the original
        labels array were replaced with. The Numpy array is the resulting
        labels array after that numerical replacement.

    Raises
    ------
    AssertionError
        An AssertionError can be raised by this function in a number of
        different ways:
            1. The first way is that if the elements of the `labels_arr`
               are not strings.
            2. Another way is that the array that results from substituting
               in the numerical values is of a different size than the
               original array of string values.

    References
    ----------
    1. https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    """
    # First let's collect all of the unique labels and order them in
    # alphabetical order.
    assert any([isinstance(labels_arr[0], np.str_),
                isinstance(labels_arr[0], str)])
    unique_labels_arr = np.unique(labels_arr)

    # Now that we have this collection, pair up those values with a
    # numerical value (sequentially).
    labels_mapper = dict(
        zip(unique_labels_arr.tolist(),
            list(range(0, unique_labels_arr.size)))
    )

    # Now replace the values
    numerical_labels_arr = labels_arr.copy()
    for old_label, new_label in labels_mapper.items():
        numerical_labels_arr[numerical_labels_arr == old_label] = new_label

    assert numerical_labels_arr.size == labels_arr.size
    to_return = (labels_mapper, numerical_labels_arr.astype(int))

    return to_return


def cv_hyperparamter_tuning(
        model,
        mode: str,
        run_brute=False,
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to take in an instantiated model
    object and perform 1 or 2 Cross-Validation searches to find the
    optimal settings of pre-determined (or specified; see 7. in the
    description of `**kwargs` in the Parameters section) hyperparameters.

    Parameters
    ----------
    model : Sklearn model object
        This object is what is returned after a Sklearn model is
        instantiated. This represents the model whose hyper-parameters
        will be tuned by this function. Note that this function only
        accepts values for this parameter that correspond to the accepted
        values of the `mode` parameter (see below).
    mode : str
        This string specifies the name of the Sklearn model that is being
        worked with by this function. Note that this function only supports
        the following values for this parameter: "svm", "nb", "rf", "lr",
        "adab", and "knn". Otherwise a value error will be raised.
    run_brute : Bool
        This Boolean controls whether or not this function also performs
        a Brute Force grid CV search. It will not if it is set to False
        (which is its default value) and it will if it is set to True.
    **kwargs : dict
        This function allows for the use of keyword argumnents to further
        control its behavior. Its accepted keyword arguments are:
            1. "x_train" - This REQUIRED keyword argument allows the user
                           to specify what feature matrix will be used to
                           fit the CV search object(s). If this argument
                           is not specifed, then a `ValueError` will be
                           raised.
            2. "y_train" - This REQUIRED keyword argument allows the user
                           to specify what labels array will be used to
                           fit the CV search object(s). If this argument
                           is not specifed, then a `ValueError`will be
                           raised.
            3. "k_value" - This keyword argument allows the user to specify
                           how many folds to use when performing the CV
                           search(es). If two such searches are performed
                           (meaning that the value of `run_brute` is set
                           to True), then specifying a value for this
                           argument and none for 4. and 5., will mean that
                           both of those searches will use the same number
                           of folds. Note that this parameter defaults to
                           a value of 5 when it is not specified.
            4. "k_value_random" - This keyword argument allows the user
                                  to specify how many folds to utilize
                                  when performing the Randomized search.
                                  If not specified, this parameter defaults
                                  to the value of `k_value`.
            5. "k_value_brute" - This keyword argument allows the user to
                                 specify how many folds to utilize when
                                 performing the Brute search. If not
                                 specified, this parameter defaults to the
                                 value of `k_value`. NOTE that the value
                                 of this parameter is ignored when
                                 `run_brute` is set to False.
            6. "scoring_method" - This keyword argument allows the user
                                  to specify what scoring method to base
                                  the judgements of the CV search(es) on.
            7. "custom_search_grid" - This keyword argument allows the user
                                      to specify any additions and/or
                                      updates they would like to make to
                                      the parmater grid that is used to
                                      perform the Randomized Search. See
                                      the source code for the default
                                      value of this dictionary grid. NOTE
                                      that this parameter must be a
                                      dictionary, otherwise a `ValueError`
                                      will be raised.

    Returns
    -------
    to_return : Sklearn model object
        This function returns a Sklearn model object that represents the
        estimator that was determined to be the best when performing the
        Cross-Validated search(es).

    Raises
    ------
    ValueError
        A ValueError is raised by this function when the user passes in
        a non-dictionary object to the "custom_search_grid" keyword
        argument.

    References
    ----------
    1. https://scikit-learn.org/stable/modules/grid_search.html#grid-search
    2. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    3. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
    4. https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    """
    # First, collect all neccessary variables.
    parameters_to_tune_dict = {
        "svm": {
            "C": stats.uniform(loc=1, scale=999),
            "decision_function_shape": ["ovo", "ovr"],
            "gamma": ["scale", "auto"]
        },
        "nb": {
            "var_smoothing": stats.uniform(loc=0, scale=1e-5)},
        "rf": {
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_leaf": stats.randint(low=1, high=4),
            "max_depth": stats.randint(low=1, high=6)
        },
        "lr": {
            "C": stats.uniform(loc=1, scale=999),
            "l1_ratio": stats.uniform(loc=0, scale=1)
        },
        "adab": {
            "n_estimators": stats.randint(low=30, high=70),
            "learning_rate": stats.uniform(loc=0.01, scale=0.99)
        },
        "knn": {
            "n_neighbors": stats.randint(low=2, high=7),
            "weights": ["uniform", "distance"]
        }
    }
    search_grid_dist_dict = parameters_to_tune_dict.get(mode, None)
    if isinstance(search_grid_dist_dict, type(None)):
        # If the user did specify a correct value for the `mode`
        # parameter value.
        error_message = "The passed-in value for the mode parameter ({}) \
        is not an accepted value. \nSee function doc-string for accepted \
        parameter values.".format(mode)
        raise ValueError(error_message)

    x_train = kwargs.get("x_train", None)
    y_train = kwargs.get("y_train", None)
    if any([
        isinstance(x_train, type(None)), isinstance(y_train, type(None))
    ]):
        # If the user did not specify a complete training data set.
        error_message = "The function expected a complete training data \
    	set to be passed in to the `x_train` and `y_train` keyword arguments. \
    	However, neither required keyword arguments were used. \nSee function \
    	docstring for more information."
        raise ValueError(error_message)

    k_value = kwargs.get("k_value", 5)
    k_value_random = kwargs.get("k_value_random", k_value)
    k_value_brute = kwargs.get("k_value_brute", k_value)

    scoring_method = kwargs.get("scoring_method", None)

    custom_search_grid = kwargs.get("custom_search_grid", {})
    if not isinstance(custom_search_grid, dict):
        error_message = "The passed-in value for the `custom_search_grid` \
    	keyword argument was of type `{}`. \nIt must be of type \
    	`dict`.".format(type(custom_search_grid))
        raise ValueError(error_message)
    search_grid_dist_dict.update(custom_search_grid)

    # Start the search by first using a RandomizedGrid to narrow down
    # the range of the optimal hyperparameter values.
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_grid_dist_dict,
        scoring=scoring_method,
        random_state=169,
        cv=k_value_random,
        refit=True)
    random_search_result = random_search.fit(x_train, y_train)

    if run_brute:
        # With this narrowed range of values, now let's perform a more
        # brute foroce standard grid search IF the user has given the OK
        # to do so.
        random_best_params_dict = random_search_result.best_params_

        search_grid_narrow_dict = random_best_params_dict.copy()
        for key, value in random_best_params_dict.items():
            # Let's go through the parameter dictionary we got from the
            # Randomized Search to create the grid dictionary we will
            # need for the brute force search.
            if any([isinstance(value, int), isinstance(value, float)]):
                # If we come accross any numerical values in the result
                # of the randomized search, create a grid of values with
                # it.
                params_with_large_cont_vals = ["C"]
                params_with_int_vals = ["min_samples_leaf",
                                        "max_depth",
                                        "n_estimators",
                                        "n_neighbors",
                                        "degree"]
                params_in_unit_interval = ["var_smoothing",
                                           "l1_ratio",
                                           "learning_rate",
                                           "tol",
                                           "min_samples_split",
                                           "min_samples_leaf",
                                           "max_samples"]
                if key in params_with_large_cont_vals:
                    # If we are working with a parameter that can take on a
                    # continuous value beyond the interval [0, 1].
                    new_grid_values = np.arange(start=value - 2,
                                                stop=value + 2.5,
                                                step=0.5)
                    final_grid_values = new_grid_values[new_grid_values >= 0]
                    search_grid_narrow_dict[key] = final_grid_values.tolist()
                elif key in params_with_int_vals:
                    # If we are working with a parameter that can only
                    # take on integer values that are greater than 0.
                    new_grid_values = np.arange(start=value - 1,
                                                stop=value + 2)
                    final_grid_values = new_grid_values[new_grid_values > 0]
                    search_grid_narrow_dict[key] = final_grid_values.tolist()
                elif key in params_in_unit_interval:
                    # If we are working with a parameter that can take
                    # on a continuous value that is restricted to the
                    # interval [0, 1].
                    new_grid_values = np.arange(start=value - 0.1,
                                                stop=value + 0.2,
                                                step=0.1)
                    unit_interval_checker = np.logical_and(
                        new_grid_values > 0, new_grid_values <= 1)
                    final_grid_values = new_grid_values[unit_interval_checker]
                    search_grid_narrow_dict[key] = final_grid_values.tolist()
            elif isinstance(value, str):
                # If we come across a parameter whose value is a string,
                # we do NOT want to create a grid, but instead would
                # like to simply enclose that string value in a list to
                # satisfy the schema of the parameter grid that the
                # `GridSearchCV()` function takes.
                search_grid_narrow_dict[key] = [value]

        brute_search = GridSearchCV(estimator=model,
                                    param_grid=search_grid_narrow_dict,
                                    scoring=scoring_method,
                                    cv=k_value_brute,
                                    refit=True)
        brute_search_result = brute_search.fit(x_train, y_train)

        to_return = brute_search_result.best_estimator_
    else:
        # If the user does NOT want to also perform a brute force grid
        # search. In that case, simply return the best estimator that
        # we got form the Randomized Search.
        to_return = random_search_result.best_estimator_

    return to_return


def model_fitting(
        parent_class_label,
        mode: str,
        calibrate_probs=True,
        run_cv_tuning=True,
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to provide an easy to use tool for
    the user that, given a specified dataset and model, fits a Sklearn
    Machine Learning Model (that may have its class probability
    predicitions and/or hyperparameters tuned if specified) for future
    use.

    Parameters
    ----------
    parent_class_label : NoneType or str
        This stirng specifies for which parent class the user would like
        to receive a list of child strings. NOTE that if you would like
        for the function to return all of the class labels that live in
        the tier 1 (what has oftentimes been referred to as the parent
        class labels throughout this project), then simply pass in the
        string "parents" to this argument.

        NOTE that when this value is set to `None`, the function will
        look for training/testing data in the `child_class_data` keyword
        argument. If data is not passed into this argument, a `KeyError`
        will be raised. The format that this argument is expecting is a
        tuple of  Numpy arrays, one representing the feature matrix and
        the  other representing the labels matrix.
    mode : str
        This string specifies the name of the Sklearn model that is being
        worked with by this function. Note that this function only supports
        the following values for this parameter: "svm", "nb", "rf", "lr",
        "adab", and "knn". Otherwise a value error will be raised.
    calibrate_probs : Bool
        This Boolean controls whether or not this function will take the
        neccessary steps to calibrate the predicted class membership
        probability distribution. See 5., 6., and 7. in the References
        sections for more information about the way in which this function
        does this task. NOTE that this parameter has a default value of
        True.
    run_cv_tuning : Bool
        This Boolean controls whether or not this function will using the
        CV parameter tuning function defined above to determine the best
        setting of the specified model for the specified data. NOTE that
        this parameter has a default value of True.
    **kwargs : dict
        This function is set up to use keyword arguments to further specify
        the behavior of this function. The accepted keywords arguments
        of this function and what they do are as follows:
            1. "test_data_frac" - This keyword argument allows the user
                                  to specify what float value in [0, 1]
                                  to use when splitting the specified
                                  dataset into training and testing data.
                                  This deafults to 0.25 when not specified.
            2. "k_value" - This keyword argument specifies how many folds
                           to use when calibrating the probability
                           predictions of the model (specified by the
                           passed-in  value of `mode`). NOTE that the
                           value of this keyword argument will be ignored
                           when the value of `calibrate_probs` is set to
                           False.
            3. "child_class_data" - This keyword argument specifies what
                                    dataset to use with this function. It
                                    must be a tuple that contains the
                                    feature matrix as its first argument
                                    and the labels array as its second.
                                    Both of these objects should be Numpy
                                    arrays. NOTE that the value of this
                                    argument will be ignored if the
                                    `parent_class_label` argument is not
                                    set to `None`.

    Returns
    -------
    to_return : Sklearn model object
        This function returns a fitted Sklearn model. The steps that this
        function takes to arrive at that model is controlled by the values
        of parameters such as `calibrate_probs` and `run_cv_tuning`.

    Raises
    ------
    ValueError
        This function raises a ValueError is a non-accepted argument is
        passed into the `mode` (required) argument.

    References
    ----------
    1. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    2. https://scikit-learn.org/stable/modules/svm.html
    3. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    4. https://scikit-learn.org/stable/modules/naive_bayes.html
    5. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
    6. https://en.wikipedia.org/wiki/Probabilistic_classification
    7.. https://scikit-learn.org/stable/modules/calibration.html
    8. https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV
    9. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    10. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    11. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss
    12. https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    """
    # First, collect all neccessary variables.
    normalized_mode = "".join(mode.lower().split())
    accepted_modes = ["svm", "nb", "rf", "lr", "adab", "knn"]
    if normalized_mode not in accepted_modes:
        error_message = "The passed-in value for the mode parameter ({}) \
        is not an accepted value. \nSee function doc-string for accepted \
        parameter values.".format(mode)
        raise ValueError(error_message)

    test_data_frac = kwargs.get("test_data_frac", 0.25)
    k_value = kwargs.get("k_value", 5)

    # Next, obtain the data that we will be needing.
    if isinstance(parent_class_label, type(None)):
        # If the user does NOT want this function to obtain the article
        # class data by running the `bow_data_loader` function. Instead,
        # they would like to specify the data to use through the keyword
        # argument `child_class_data`.
        feature_matrix, raw_labels_arr = kwargs.get("child_class_data",
                                                    (None, None))
        if any([isinstance(feature_matrix, type(None)),
                isinstance(raw_labels_arr, type(None))]):
            # If the user has not correctly specified the values of the
            # data that they would like for the function to use.
            error_message = "By setting `parent_class_label` to `None`, \
            you have specified that you would like for this function to \
            use your own data. \nHowever, you have not correctly specified \
            that data. See the function docstring for how to do that \
            correctly."
            raise ValueError(error_message)
        if raw_labels_arr.dtype == "int":
            # If the user has specified a labels array that has already
            # been mapped to a numerical form.
            numerical_labels_arr = raw_labels_arr
        elif isinstance(raw_labels_arr[0], str):
            # If the user has instead specified a labels array that has
            # YET to be mapped to a numerical form.
            _, numerical_labels_arr = label_transformer(raw_labels_arr)
    else:
        # If the user DOES want this function to obtain the neccessary
        # data by running the `bow_data_loader` function.
        feature_matrix, raw_labels_arr = bow_data_loader(parent_class_label)

        # After loading, transform the labels column and then split into
        # training and testing data.
        _, numerical_labels_arr = label_transformer(raw_labels_arr)

    # Now, split the obtained data (no matter how it was obtained) into
    # training and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        numerical_labels_arr,
        test_size=test_data_frac,
        random_state=369
    )

    # Now, instantiate the neccessary model for the specified mode.
    if normalized_mode == "svm":
        # If the user would like for a Suport Vector Machine
        # classification model to be fitted.
        base_model = SVC(probability=True)
    elif normalized_mode == "nb":
        # If the user would like for a Naive Bayes classification model
        # to be fitted.
        base_model = GaussianNB()
    elif normalized_mode == "rf":
        # If the user would like for a Random Forect classification
        # model to be fitted.
        base_model = RandomForestClassifier(random_state=169,
                                            class_weight="balanced")
    elif normalized_mode == "lr":
        # If the user would like for a Logistic Regression
        # classification model to be fitted.
        base_model = LogisticRegression(penalty="elasticnet",
                                        fit_intercept=True,
                                        class_weight="balanced",
                                        solver="saga",
                                        max_iter=200)
    elif normalized_mode == "adab":
        # If the user would like for an AdaBoost classification model to
        # be fitted.
        base_model = AdaBoostClassifier(algorithm="SAMME.R",
                                        random_state=669)
    elif normalized_mode == "knn":
        # If the user would like for a K-Nearest-Neighbor Classifier to
        # be fitted.
        base_model = KNeighborsClassifier()

    # Determine if we need to also perform a CV search for the optimal
    # settings of the model's parameters to best fit this data.
    if run_cv_tuning:
        # If the user would like to use Cross-Validation to perform a
        # search for the best hyper-parameter settings of the models.
        tuned_model = cv_hyperparamter_tuning(model=base_model,
                                              mode=normalized_mode,
                                              x_train=x_train,
                                              y_train=y_train)
    else:
        tuned_model = base_model

    # Determine if we need to also instantiate a calibration object.
    if calibrate_probs:
        # If the user would like for this function to also calibrate
        # the predicted class membership probability distribution.

        # Instantiate the calibration model objects.
        calib_sigmoid_model = CalibratedClassifierCV(
            base_estimator=tuned_model, cv=k_value, method="sigmoid")
        calib_isotonic_model = CalibratedClassifierCV(
            base_estimator=tuned_model, cv=k_value, method="isotonic")

        # Fit the calibration models with the training data.
        calib_sigmoid_model.fit(x_train, y_train)
        calib_isotonic_model.fit(x_train, y_train)

        calibrated_models_list = [calib_sigmoid_model,
                                  calib_isotonic_model]

        # Determine which calibration works best.
        prob_dist_sigmoid = calib_sigmoid_model.predict_proba(x_test)
        prob_dist_isotonic = calib_isotonic_model.predict_proba(x_test)

        sigmoid_brier = me.multiple_brier_score_loss(
            y_test,
            prob_dist_sigmoid
        )
        isotonic_brier = me.multiple_brier_score_loss(
            y_test, prob_dist_isotonic
        )

        best_calib_index = np.argmin([sigmoid_brier, isotonic_brier])

        # Asign best calibrated model to the `final_model` variable
        # name.
        final_model = calibrated_models_list[best_calib_index]

    else:
        # If the user would NOT like for this function to also calibrate
        # the predicted class membership probability distribution.
        final_model = tuned_model

    to_return = final_model

    return to_return


def models_comparison(parent_class_label: str, models_to_fit="all"):
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    ----------
    parent_class_label : str
        This stirng specifies for which parent class the user would like
        to receive a list of child strings. NOTE that if you would like
        for the function to return all of the class labels that live in
        the tier 1 (what has oftentimes been referred to as the parent
        class labels throughout this project), then simply pass in the
        string "parents" to this argument.
    models_to_fit : str or list; default "all"
        This argument allows

    Returns
    -------
    to_return :
        This function returns a

    References
    ----------
    1.
    """
    # First, collect all neccessary variables that will be used for the
    # rest of the function.
    if models_to_fit == "all":
        actual_models_to_fit = ["svm", "nb", "rf", "lr", "adab", "knn"]
    else:
        actual_models_to_fit = models_to_fit

    # Next load in the data that will be used to train and evaluate the
    # resulting models.
    feature_matrix, raw_labels_arr = bow_data_loader(parent_class_label)
    _, numerical_labels_arr = label_transformer(raw_labels_arr)

    x_train, x_test, y_train, y_test = train_test_split(feature_matrix,
                                                        numerical_labels_arr,
                                                        test_size=0.25,
                                                        random_state=569)

    # Now, use the `model_fitting()` function defined above to obtain
    # a collection of fitted models.
    best_models_list = [
        model_fitting(
            parent_class_label=None,
            mode=model_name,
            child_class_data=(x_train, y_train),
            test_data_frac=0.15
        ) for model_name in actual_models_to_fit
    ]

    # With these fitted models, use the tools in the `model_evaluation`
    # module to determine which one is best. Return the one that is
    # best.
    unique_x_test, one_hot_y_test = me.true_classes_compilation(x_test,
                                                                y_test)
    num_class_labels = one_hot_y_test.shape[1]
    compiled_predictions_list = [
        me.predicted_classes_compilation(
            ml_model=model,
            test_feature_matrix=unique_x_test,
            available_labels_arr=np.arange(0, num_class_labels),
            closeness_threshold=0.05
        ) for model in best_models_list
    ]
    metrics_list = [
        me.metric_reporter("hamming", one_hot_y_test, prediction)
        for prediction in compiled_predictions_list
    ]

    index_of_best_model = np.argmin(metrics_list)
    best_model = best_models_list[index_of_best_model]
    to_return = best_model

    return to_return


def save_model(parent_class_label: str, run_comparison=True, **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    ----------
    parent_class_label : str
        This stirng specifies for which parent class the user would like
        to receive a list of child strings. NOTE that if you would like
        for the function to return all of the class labels that live in
        the tier 1 (what has oftentimes been referred to as the parent
        class labels throughout this project), then simply pass in the
        string "parents" to this argument.
    run_comparison : Bool
    **kwargs : dict

    Returns
    -------
    to_return :
        This function returns a string that indicates whether or not the
        process undertaken to save the specified model was successful.

    References
    ----------
    1.
    """
    to_return = None
    ###

    return to_return


def load_model(parent_class_label: str, **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    ----------
    parent_class_label : str
        This stirng specifies for which parent class the user would like
        to receive a list of child strings. NOTE that if you would like
        for the function to return all of the class labels that live in
        the tier 1 (what has oftentimes been referred to as the parent
        class labels throughout this project), then simply pass in the
        string "parents" to this argument.
    **kwargs : dict

    Returns
    -------
    to_return :
        This function returns a

    References
    ----------
    1.
    """
    to_return = None
    ###

    return to_return
