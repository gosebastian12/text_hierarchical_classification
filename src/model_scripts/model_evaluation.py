#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 8

@author: Sebastian Gonzalez
"""

####################################
### Neccessary Import Statements ###
####################################
# Data Manipulation
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import hamming_loss

# Data Validation
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


####################################
### Define our Modular Functions ###
####################################
def multiple_brier_score_loss(
        labels_arr: np.array,
        labels_prob_dist_arr: np.array):
    """
    Purpose
    -------
    The purpose of this function is to compute the multiple brier score
    loss associated with a given predicted probability distribution
    matrix for a set of true labels outputted by a trained model when it
    receives a new training instance. Recall that this is an indication
    of how well calibrated the probability predictions of a particular
    classifier are. The **lower** the loss, the better the classifier
    calibration. See References section for more information.

    The equation that is implemented to arrive at this score is the
    following:
        \\begin{align}
        B = \\frac{1}{N}\\sum_{t=1}^{N}\\sum_{i=1}^{R} (f_{ti} - o_{ti})^2
        \\end{align}
    where $B$ is the (Multiple) Brier Score Loss, $N$ is the number of
    training instances, $R$ is the number of unique class labels, $f_{ti}$
    is the probability that was forecasted for the i-th class of the t-th
    instance, and $o_{ti}$ is an indicator variable for the i-th class
    of the t-th training instance that is 1 if the t-th instance does in
    fact belong to class i and 0 otherwise. NOTE that this score is
    guranteed to lie with the closed-interval [0, 2] (compare with the
    modern, binary formulation of this metric which is confied to the
    closed interval [0, 1]).

    Parameters
    ----------
    labels_arr : Numpy Array
        This Numpy Array contains all of the correct labels of the data
        instances. The particular shape of this array in un-important
        **so long as** the number of elements in it are equal to the
        number of rows in `labels_prob_dist_arr` (see Raises section).
    labels_prob_dist_arr : Numpy Array
        This multi-dimensional Numpy Array contains all of the class
        probability predictions for the feature matrix that was used to
        create it. As such, its number of rows is set by the number of
        data instances we have and its number of columns is set by the
        number of unique classes a given instance can fall into.

    Returns
    -------
    to_return : float
        This function returns a float value that is the Multiple Brier
        Loss. That is, a measure of how well the probabilities in
        probability matrix prediction are calibrated.

    Raises
    ------
    AssertionError
        An AssertionError can be raised by this function in a number of
        different ways:
            1. The first way is that if the value passed into
               `labels_prob_dist_arr` is not properly normalized and
               after following the steps to try and normalize it, it is
               still not normalized for some reason.
            2. One is that the number of elements in the `labels_arr` are
               NOT equal to the number of rows in `labels_prob_dist_arr`.
            3. Another way is that the number of columns for
               `labels_prob_dist_arr` is not equal to the number of unique
               classes in `labels_arr`.
            4. Lastly, such a error will be raised if the resultant Brier
               Score Loss does not fall in the closed-interval [0, 2].
               This would indicate that an error was made in the
               calcaulation.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Brier_score#3-component_decomposition
    2. https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    3. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    # First, collect neccessary variables and ensure that the vectorized
    # operations of this function will work.
    normalized_probs_check = labels_prob_dist_arr.sum(axis=1)
    if not np.all(normalized_probs_check == 1):
        # If the passed in probability array is NOT properly normalized.
        normalized_labels_prob_dist_arr = \
            labels_prob_dist_arr / labels_prob_dist_arr.sum(
                axis=1).reshape(-1, 1)
        assert np.all(normalized_labels_prob_dist_arr.sum(axis=1) == 1)
    else:
        normalized_labels_prob_dist_arr = labels_prob_dist_arr

    normalized_labels_arr = labels_arr.flatten().reshape(-1, 1)
    num_instances = normalized_labels_prob_dist_arr.shape[0]
    num_classes = normalized_labels_prob_dist_arr.shape[1]

    assert normalized_labels_arr.shape[0] == num_instances
    assert np.unique(normalized_labels_arr).size == num_classes
    assert np.all(np.logical_and(normalized_labels_prob_dist_arr >= 0,
                                 normalized_labels_prob_dist_arr <= 1))

    enc = OneHotEncoder(sparse=False)
    one_hot_labels_arr = enc.fit_transform(normalized_labels_arr)

    # Now, perform the calculation
    square_diffs_arr = (
        normalized_labels_prob_dist_arr - one_hot_labels_arr)**2
    brier_score = square_diffs_arr.sum(axis=1).mean()

    # Lastly, verify the result and if it's good, return it.
    assert 0 <= brier_score <= 2
    to_return = brier_score

    return to_return


def true_classes_compilation(
        feature_matrix: np.array,
        labels_arr: np.array):
    """
    Purpose
    -------
    The purpose of this function is to take an un-changed Bag-of-Words
    (BOW) feature matrix and (numerical) labels array and find the
    instances (rows) that are in fact duplicates. After finding these
    cases, a new feature matrix is created where the duplicate rows are
    reduced to a single row. As this new feature matrix is created, a new
    labels array is created as well; first, the function ensures that
    (1D) index of a label in the new labels array corresponds to the
    instance in the new feature matrix that has that same index value for
    its row index. Second, the labels for the duplicate instances are all
    compiled into a list and stored in the label index whose value is
    equal to the row index in which the single instance for the original
    duplicate rows appears in the new feature matrix.

    Parameters
    ----------
    feature_matrix : Numpy Array
        This is the 2D Numpy Array (of shape MxN) that contains the M
        feature values of the N training instances we have in our dataset.
    labels_arr : Numpy Array)
        This is the 1D Numpy Array (of length N) that contains all of the
        numerical label values for the training instances in the array
        that is passed into `feature_matrix`.

    Returns
    -------
    to_return : (Numpy Array, Numpy Array)
        This function returns a tuple that contains a Numpy Array and
        nested list of lists. The first element represents the new feature
        matrix that no longer has duplicate rows. The second element is
        another Numpy Array; this one corresponds to the new labels array
        which is 2D andone-hot-encoded where the rows correspond (in a
        1-to-1 fashion) to the rows in the new feature matrix and the
        columns correspond to a particular class label. That is, $a_{ij}$
        is 1 if the ith (unique) instance contains class j in its collection
        of class labels and 0 otherwise (evidently, the elements of this
        array can only be 1 or 0).

    Raises
    ------
    `AssertionError`
        An assertion error may be raised for a number of different reasons
        which include:
            1. If the array passed into `labels_arr` is NOT comprised of
               numerical values
            2. If the number of rows in the array passed into
               `feature_matrix` is NOT equal to the number of elements
               in the array that was passed into `labels_arr`.
            3. If the number of rows in the new feature matrix is NOT
               equal to the number of  elements in the new labels array.
            4. If all of the labels stored in new labels list are not
               themselves put into lists.
            5. If the labels that the function identified for the
               duplicate instances are not the correct labels.
            6. If for some reason, the new feature matrix and/or the new
               labels array has more rows/elements that their original
               counter-parts.
            7. If the difference in the number of rows between the new
               and old feature matricies is not the same as the difference
               between the number of elements between the new and old
               labels array.

    References
    ----------
    1. https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    2. https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    """
    # First, validate the data that has been passed-in to this function.
    normalized_labels_arr = labels_arr.flatten()
    assert any([isinstance(normalized_labels_arr[0], float),
                isinstance(normalized_labels_arr[0], int),
                isinstance(normalized_labels_arr[0], np.int),
                isinstance(normalized_labels_arr[0], np.int8),
                isinstance(normalized_labels_arr[0], np.int16),
                isinstance(normalized_labels_arr[0], np.int32),
                isinstance(normalized_labels_arr[0], np.int64)])
    assert feature_matrix.shape[0] == normalized_labels_arr.size

    # Next, pass the feature matrix into `np.unique` to get all of the
    # tools that we will be needing.
    unique_feature_matrix, unique_indices, reconstruct_indices, row_counts = \
        np.unique(ar=feature_matrix,
                  return_index=True,
                  return_inverse=True,
                  return_counts=True,
                  axis=0)
    num_unique_labels = np.unique(ar=normalized_labels_arr).size

    # Now compile the labels for these unique rows.
    unique_labels_arr = normalized_labels_arr[unique_indices]
    compiled_labels_arr = np.zeros([
        unique_feature_matrix.shape[0], num_unique_labels
    ])

    single_label_indicies = np.argwhere(row_counts == 1).flatten().tolist()
    single_labels = unique_labels_arr[single_label_indicies].tolist()
    for feat_index, feat_label in zip(single_label_indicies, single_labels):
        # Iterate over all of the indicies of the rows that correspond
        # to instances that only have one class label and the class label
        # that they correspond to.
        compiled_labels_arr[feat_index][feat_label] = 1

    # Now compile the labels for the training instances that have multiple
    # labels and save them to `compiled_labels_arr`.
    mult_label_indicies = np.argwhere(row_counts > 1).flatten().tolist()
    for feat_index in mult_label_indicies:
        # Iterate over the indicies for where the instance in the feature
        # matrix corresponds to multiple labels.

        # Take advantage of the fact that for the instances that have
        # multiple labels, their indicies in the reconstruct array (the
        # array of indicies that can be used to reconstruct the original
        # array from the resulting unique array) are all the same, the
        # index of the first occurance of the row in the original array
        # which is the index that we have access to from
        # `mult_label_indicies`. Thus, all we have to do is determine
        # where this row index occurs in the reconstruct array and those
        # indicies are the indicies that we will find our labels at.
        og_row_indicies = np.argwhere(
            reconstruct_indices == feat_index
        ).flatten()
        mult_labels = normalized_labels_arr[og_row_indicies].tolist()

        # Save result as list
        compiled_labels_arr[feat_index][mult_labels] = 1

    # Validate data and if it passes the tests, return it.
    assert np.all(compiled_labels_arr.sum(axis=1) >= 1)
    assert np.all(np.logical_or(
        compiled_labels_arr == 0, compiled_labels_arr == 1
    ))

    assert unique_feature_matrix.shape[0] == compiled_labels_arr.shape[0]
    assert compiled_labels_arr.shape[1] == num_unique_labels

    assert feature_matrix.size >= unique_feature_matrix.size
    assert normalized_labels_arr.size >= compiled_labels_arr.shape[0]
    assert feature_matrix.shape[0] - unique_feature_matrix.shape[0] == \
        normalized_labels_arr.size - compiled_labels_arr.shape[0]

    to_return = (unique_feature_matrix, compiled_labels_arr)
    return to_return


def predicted_classes_compilation(
        ml_model,
        test_feature_matrix: np.array,
        available_labels_arr: np.array,
        closeness_threshold: float,
        give_passed_probs=False,
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to compile all of the classes whose
    predicted probabilities by the used model are all the highest and
    very similar since this represents a situation in which the model
    cannot assertively decide between a small set of possible classes.

    Parameters
    ----------
    ml_model : Sklearn model object
        This model object represents the ML algorithm Sklearn
        implementation to use in this function. NOTE that if this model
        and yet to be trained, the function will look for training data
        in the `training_data` keyword argument (see more details below).
        NOTE that since this function relies entirely on the predicted
        probability matrix from this model, it would be wise to train the
        model before-hand so that can you also tune its probability
        predictions which will make them more accurate.
    test_feature_matrix : Numpy Array)
        This 2D Numpy Array represents the testing feature matrix that
        this function will use. NOTE that if the rows of this matrix are
        NOT strictly unique, then the keys of the
    available_labels_arr (Numpy Array) - This (1D) Numpy Array is a
                                         collection of all of the labels
                                         that an instance can be
                                         categorized into.
    closeness_threshold (float) - This float represents the upper bound
                                  value that is used to find any
                                  maximum-probability clusters.
    give_passed_probs (Bool) - This Boolean controls whether or not the
                               function also returns the probability
                               estimate for all of the returned predicted
                               labels. This may be useful for debugging.
                               NOTE that the default value of this
                               parameter is `False.`
    **kwargs (dict) - This function is set up to make use of keyword
                      arguments for one particular case: when the user
                      passes in an unfitted model to the `ml_model`
                      argument. In this situation, the function will then
                      look for an `X` and `y` training array in the
                      `training_data` keyword argument. If no value is
                      passed into this keyword argument, then the function
                      will raise a `ValueError` (see Raises section for
                      more information).

    Returns
    -------
    to_return : tuple or dictionary
        This function returns either a dictionary or a tuple. The former
        is the case when the `give_passed_probs` is set to True and the
        latter is the case when that argument is set to False or the rows
        of the  array passed into `test_feature_matrix` are not strictly
        unique. If just a dictionary, that dictionary has keys ranging
        from 0 to the  number of unique rows in the array that was passed
        into  `test_feature_matrix` and values that are the labels that
        correspond to the instance which the row index of the key that
        points to this value. If a tuple, there are few possibilities of
        what its values represent. One is that the first element will be
        a dictionary with index keys that point to a collection of
        probabilites for the compiled class and the second element will
        be the same dictionary described for the non-tuple case. The
        other possibility is that the first will be the result of only
        keeping the unique rows of `test_feature_matrix` and the other
        elemements will either be both the probability and labels
        dictionary or simply just the labels dictionary.

    Raises
    ------
    ValueError
        This function raises such an error when a fitted model is NOT
        passed into the `ml_model` argument and nothing is passed to the
        `training_data` keyword argument.
    AssertionError
        This function raises such an error when the labels and proability
        dictionaries do not have the same length.

    References
    ----------
    1. https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
    2. https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    3. https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    4. https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    5. https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html
    """
    # First get the class probability distribution.
    try:
        # Check to see if the user passed in a fitted model.
        _ = check_is_fitted(ml_model)
    except NotFittedError:
        # If the user did NOT pass in a fitted model.
        x_train, y_train = kwargs.get("training_data", (None, None))
        if any([isinstance(x_train, type(None)),
                isinstance(y_train, type(None))]):
            # If the user did not pass in any training data.
            error_message = "The Sklearn model object that was passed in to\
            the `ml_model` argument has yet to be trained. \nBecause of this\
            the function looked for training data in the `training_data` \
            keyword argument, BUT found no values for it. \nSee function \
            docstring for more information."
            raise ValueError(error_message)
        ml_model.fit(x_train, y_train)

    unique_feat_mat_checker = np.unique(ar=test_feature_matrix,
                                        return_counts=True,
                                        axis=0)
    if np.all(unique_feat_mat_checker[1] == 1):
        # If each row of the passed in test feature matrix is unique.
        feat_mat = test_feature_matrix
        was_unique = True
    else:
        # Otherwise.
        feat_mat = unique_feat_mat_checker[0]
        was_unique = False

    num_unique_labels = available_labels_arr.size
    labels_prob_dist_arr = ml_model.predict_proba(feat_mat)
    assert labels_prob_dist_arr.shape[1] <= num_unique_labels

    # Second, find out if there is a "max-probability cluster"; that is,
    # if there are multiple probabilities that are near the tail of the
    # distribution that are very close in value.
    max_prob_indicies = np.argmax(a=labels_prob_dist_arr, axis=1)
    max_probs = labels_prob_dist_arr.max(axis=1)

    diff_max_arr = max_probs.reshape(-1, 1) - labels_prob_dist_arr
    threshold_checker = np.logical_and(diff_max_arr >= 0,
                                       diff_max_arr <= closeness_threshold
                                       )
    threshold_passed_indices = np.argwhere(threshold_checker)

    current_passed_row_index = 0
    full_label_predictions_arr = np.zeros([
        labels_prob_dist_arr.shape[0], num_unique_labels
    ])
    full_label_probs_dict = {}
    for row_index in range(labels_prob_dist_arr.shape[0]):
        # Iterate through each row of the feature matrix that we are
        # using.

        # Check to see if we have already determined that we are done
        # with this row.
        include_current_passed = threshold_passed_indices[
            current_passed_row_index, 0] == row_index
        assert include_current_passed

        try:
            include_next_passed = threshold_passed_indices[
                current_passed_row_index + 1, 0] == row_index
        except IndexError:
            # If an IndexError (which will occur when there are no more
            # elements in the `threshold_passed_indices` list to check).
            include_next_passed = False
        if not include_next_passed:
            # If the feature row that we are currently checking does NOT
            # have any max-prob clustering (which we know is the case
            # since the only element in the instance row that passed the
            # threshold check was the max prob itself), then there's
            # nothing to check and we can move on to the next feature
            # instance.

            # Obtain the single probability that passed the threshold
            # test.
            two_d_max_prob_index = threshold_passed_indices[
                current_passed_row_index
            ]
            passed_prob = labels_prob_dist_arr[two_d_max_prob_index[0],
                                               two_d_max_prob_index[1]]

            # Now run checks to make sure that it is the same max-prob
            # that we found earlier.
            assert np.isclose(passed_prob, max_probs[row_index])
            assert two_d_max_prob_index[1] == max_prob_indicies[row_index]

            # If these tests were passed, then save the class label that
            # corresponds to this label.
            full_label_predictions_arr[row_index][
                two_d_max_prob_index[1]] = 1
            full_label_probs_dict[row_index] = [passed_prob]

            # Update neccessary values.
            current_passed_row_index += 1
        else:
            # If there IS indeed max-prob clustering.
            full_label_probs_dict[row_index] = []
            while include_current_passed:
                # Iterate while we still are looking at probabilities
                # that passed the threshold check for the current row
                # we have in our iteration.
                two_d_index = threshold_passed_indices[
                    current_passed_row_index
                ]

                passed_label = available_labels_arr[two_d_index[1]]
                passed_prob = labels_prob_dist_arr[two_d_index[0],
                                                   two_d_index[1]]

                full_label_predictions_arr[row_index][passed_label] = 1
                full_label_probs_dict[row_index].append(passed_prob)

                # Update neccessary values.
                current_passed_row_index += 1
                include_current_passed = threshold_passed_indices[
                    current_passed_row_index, 0] == row_index

    assert np.all(full_label_predictions_arr.sum(axis=1) >= 1)
    assert np.all(np.logical_or(
        full_label_predictions_arr == 0, full_label_predictions_arr == 1
    ))
    assert full_label_predictions_arr.shape[0] == labels_prob_dist_arr.shape[0]
    if give_passed_probs:
        # If the user wishes to have the function also return the predicted
        # class probabilities of all of the labels that were found to fall
        # into the max-prob cluster. The user may want to do this to
        # check that what the function has found is in fact correct.
        assert full_label_predictions_arr.shape[0] == len(
            full_label_probs_dict)
        to_return_list = [full_label_probs_dict,
                          full_label_predictions_arr]

    else:
        # If the user doesn't care to also have the function returned
        # the predicted probabilities of all the compiled classes.
        to_return_list = [full_label_predictions_arr]

    if was_unique:
        # If the user passed in a feature matrix that DID have all
        # unique rows, then simply return the probability and
        # prediction dictionaries.
        to_return = to_return_list[0] if len(to_return_list) == 1 \
            else tuple(to_return_list)
    else:
        # If the user passed in a feature matrix that does NOT have
        # all unique rows, then also return the new feature matrix
        # that was created to ensure that the feature matrix that was
        # used in the function did in fact have unique rows.
        to_return_list.insert(0, feat_mat)
        to_return = tuple(to_return_list)

    return to_return


def metric_reporter(
        mode: str,
        compiled_truth: np.array,
        compiled_predictions: np.array,
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    -------
    mode : str
        The string that is passed into this parameter controls the
        calculations that this function makes to arrive at an accuracy
        score. Namely, if:
            1. "exact", then the function will only deem a prediction for
               an instance as a success if it predicts all of the actual
               classes that a given instance falls into;
            2. "partial" or "hamming", then the function will award
               partial credit to a prediction that only predicts a subset
               of the actual classes where the credit is equal to the
               number of sucessful class predictions;
            3. "average", then the function will execute the  computations
               of both the "exact" and "partial" ("hamming") modes and
               will average their resulting accuarcy scores (if a weighted
               average is desired, then it can be specified through the
               keyword arguments of this function, see  below.)
    compiled_truth : Numpy Array
        This array
    compiled_predictions : Numpy Array
        This array
    **kwargs : dict
        This function is set up to accept keyword arguments

    Returns
    -------
    to_return :
        This function returns a

    Raises
    ------
    ValueError
        This error is raised when

    References
    ----------
    1. https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    2. https://medium.com/towards-artificial-intelligence/understanding-multi-label-classification-model-and-accuracy-metrics-1b2a8e2648ca
    3. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html
    4. https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
    5. https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit/234354
    6. https://scikit-learn.org/stable/modules/multiclass.html
    """
    to_return = None
    # First, collect all neccessary and define all of the neccessary
    # paramaters and functions.
    normalized_mode = "".join(mode.lower().split())
    average_weights = kwargs.get("average_weights", (None, None))

    assert compiled_truth.shape == compiled_predictions.shape
    num_instances = compiled_predictions.shape[0]

    def comparison_func(
            actual_arr: np.array,
            predicted_arr: np.array,
            comp_mode=normalized_mode):
        """
        The purpose of this function is to . It returns an integer that is either zero or one.
        """
        sub_to_return = None
        if comp_mode == "exact":
            # If the user would like for this function to report the accuracy
            # score obtained from only accepting strict matches for the class
            # predictions.
            row_by_row_comparison = np.all(actual_arr == predicted_arr,
                                           axis=1)
            sub_to_return = row_by_row_comparison.sum() / num_instances
        elif comp_mode in ("partial", "hamming"):
            # If the user would like for this function to report the accuracy
            # score obtained from rewarding partial credit for prediction
            # instances that identified some of the correct class labels.
            computed_hamming = hamming_loss(actual_arr, predicted_arr)
            sub_to_return = computed_hamming
        elif comp_mode == "average":
            # If the user would like for this function to report the accuracy
            # score obtained from averaging the scores obtained from the exact
            # and partial methods.
            row_by_row_comparison = np.all(actual_arr == predicted_arr,
                                           axis=1)
            exact = row_by_row_comparison.sum() / num_instances

            computed_hamming = hamming_loss(actual_arr, predicted_arr)

            sub_to_return = average_weights[0] * exact + \
                average_weights[0] * computed_hamming

        return sub_to_return

    # Next do the neccessary computations.
    to_return = comparison_func(actual_arr=compiled_truth,
                                predicted_arr=compiled_predictions)

    return to_return


def prob_threshold_optimizer():
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    ----------
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
    # First,

    return to_return


def hi_lint():
    """
    Purpose
    -------
    The purpose of this function is to

    Parameters
    ----------
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
    # First,

    return to_return
