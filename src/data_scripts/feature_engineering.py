    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jul 21

@author: Sebastian Gonzalez
"""

####################################
### Neccessary Import Statements ###
####################################
# Data Access
import json
import os

# Data Manipulation
import pandas as pd
import numpy as np

# Model objects
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

# Articles DataFrame
from src.data_scripts import database_compiler
FULL_DF = database_compiler.load_in_full_data()

# Taxonomy
PATH_OF_SCRIPT = os.path.dirname(__file__)
REL_PATH_TO_TAXONOMY = "../../data/raw/iab_taxonomy-v2.json"
FULL_PATH_TO_TAXONOMY = os.path.join(PATH_OF_SCRIPT,
                                     REL_PATH_TO_TAXONOMY)

TAXONOMY_FILE = open(FULL_PATH_TO_TAXONOMY)
TAXONOMY_DICT = json.load(TAXONOMY_FILE)
TAXONOMY_FILE.close()


####################################
### Define our Modular Functions ###
####################################
def children_retrival(
        parent_class_label: str,
        full_dict=TAXONOMY_DICT):
    """
    Purpose
    -------
    The purpose of this function is to use the taxonomy JSON file (this
    file can be found in ./data/raw/) to compile a list of strings that
    describe all of the immediate children of a specified parent class
    label.

    Parameters
    ----------
    parent_class_label : str
        This stirng specifies for which parent class the user would like
        to receive a list of child strings. NOTE that if you would like
        for the function to return all of the class labels that live in
        the tier 1 (what has oftentimes been referred to as the parent
        class labels throughout this project), then simply pass in the
        string "parents" to this argument.
    full_dict : dict 
        This dictionary is the loaded in taxonomy dictionary that is
        obtained from taxonomy JSON file (see purpose section for more
        about this file).

    Returns
    -------
    to_return : (int, list)
        The integer specifies at which tier level (1, 2, 3, or 4) the
        children live in and the list of strings is the collection of all
        of the child class labels for the specified parent class.
    """
    parent_class_label = parent_class_label.lower()
    # First, we need to look at the parent class label to determine
    # what kind of search we would like to perform.
    if parent_class_label == "parents":
        # If we are interested in compiling the list of class labels
        # that live at the top of the hierarchy.
        raw_children_list = [list(sequence_dict.values())[0]
                             for sequence_dict in full_dict]
        children_list = [
            child for child in raw_children_list if isinstance(
                child, str)]

        child_tier_lvl = 1
        child_tier_lvls_list = [child_tier_lvl] * len(children_list)
    else:
        # If we are looking at children that live in tiers 2, 3, or 4.

        # First, go through the taxonomy dictionary and get the cases
        # that contain this parent label for further inspection.
        matched_dicts_list = [
            potential_dict for potential_dict in full_dict \
            if parent_class_label in str(potential_dict).lower()]

        # Now use each of these matched dictionaries to see if we can
        # find instances where the parent class label that we are
        # interested in is a key because that will then point us to its
        # children.
        children_list = []
        child_tier_lvls_list = []
        for current_dict in matched_dicts_list:
            # Note that we are only iterating through the matched
            # dictionaries. That is, the dictionaries that have the
            # parent class label somewhere in them.
            value_is_dict = isinstance(list(current_dict.values())[0],
                                       dict)
            # we are almost guaranteed that this will return True.
            is_key = False
            child_tier_lvl = 1
            while value_is_dict and not is_key:
                # Perform the check as many times as possible (as many
                # times as we have dictionaries to work with) and/or
                # until we figure out that the parent class label does
                # in fact act as a key in one of the nested
                # dictionaries.
                child_dict = list(current_dict.values())[0]
                is_key = list(child_dict.keys())[
                    0].lower() == parent_class_label
                # in order for these lines of code to be ran,
                # `current_dict` has to be a dictionary, so these lines
                # of code should never fail since that is a condition
                # for this while loop to continues.

                # update the values we have
                current_dict = child_dict
                value_is_dict = isinstance(
                    list(current_dict.values())[0], dict)
                child_tier_lvl += 1

            if is_key:
                # If the while loop from above ended because we determined
                # that the specified parent class label does act as a key in
                # one of the nested dictionaries.
                child_label = list(current_dict.values())[0]
                if isinstance(child_label, dict):
                    # If the parent class label points to yet another
                    # dictionary since its tier sequence continues on.
                    child_label = list(child_label.keys())[0]
                assert isinstance(child_label, str)
                # Just to be sure ;)
                children_list.append(child_label)
                child_tier_lvls_list.append(child_tier_lvl)

    # Remove duplicates that may have occured and perform further data
    # validation tests.
    final_children_list = list(set(children_list))

    assert isinstance(child_tier_lvl, int)
    assert 0 < child_tier_lvl <= 4
    assert np.all(np.array(child_tier_lvls_list) == child_tier_lvls_list[0])

    to_return = (child_tier_lvls_list[0], final_children_list)

    return to_return


def class_data_retrival(
        parent_class_label: str,
        give_child_tier_lvl=False,
        full_data_frame=FULL_DF):
    """
    Purpose
    -------
    The purpose of this function is to provide a tool to the user that
    allows them to be able to obtain all of the articles that fall into
    the children classes of a specified parent class so that this data
    can be used as training and testing instances for the classifier
    that distinguishes between those children classes.

    Parameters
    ----------
    parent_class_label : str
        This string represents the class label that is the Parent class
        of all of the sub-classes that will be distignuished and predicted
        by a classifier that you wish to build. I.e., if you want to build
        a classifier for the children of the "Auto Type" class (which
        includes "Budget Cars", "Concept Cars", and "Luxury Cars" to name
        a few), then you simply have to pass in the "Auto Type" string to
        this parameter.
    give_child_tier_lvl : Bool
        This Boolean allows the user to control whether or not the function
        returns the tier level that the children classes of 
        `parent_class_label` live in the classification taxonomy or not.
        If set to False (which is its default value), then the function
        will NOT return this value and if it is set to True, it WILL.
    full_data_frame : Pandas DataFrame
        This Pandas DataFrame contains every single article that makes up
        the entire corpus that you have accessed/collected.

    Returns
    -------
    to_return : Pandas DataFrame
        This contains all of the articles that belong to one of the
        children classes of the specified parent class.

    References
    ----------
    1. See the taxonomy JSON and/or CSV files if you wish to put
       together a list of all of the parent classes. It may also be
       instructuive to go through the Taxonomy Exploration and/or
       Database Access notebooks to do this as well.
    """
    to_return = None
    parent_class_label = parent_class_label.lower()
    # Normalize this parameter value to handle cases where the user
    # may have used a different capitalization scheme.
    # First, compile a list of all of the child classes that
    # correspond to this parent class.
    child_tier_lvl, children_list = children_retrival(parent_class_label)

    # Second, let's obtain the data that belong to the children sub-
    # classes of the specified parent class.
    tier_label = "Tier{}".format(child_tier_lvl)
    child_articles_dfs_list = [
        full_data_frame[full_data_frame[tier_label] == child_label] \
        for child_label in children_list
    ]
    articles_class_df = pd.concat(objs=child_articles_dfs_list,
                                  ignore_index=True)

    if give_child_tier_lvl:
        to_return = (child_tier_lvl, articles_class_df)
    else:
        to_return = articles_class_df

    return to_return


def bag_of_words_converter(
        mode: str,
        parent_class_label,
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to take a collection of article
    texts and use Bag-of-Word (BOW) algorithms to convert this into a
    numerical feature set that is much more suitable for training
    Machine-Learning models.

    Parameters
    ----------
    mode : str
        This string allows the user to specify how they would like the
        article preprocessed content to be converted to a numberical
        vector; that is, it allows them to specify what bag of words
        method to use. The available options include:
            1. "TF" - In this mode, the Term-Frequency approach will be
                      used to convert the textual data to numeric data.
            2. "TF-IDF" - In this mode, the Term-Frequency-Inverse-
                          Document-Frequency approach will be used to 
                          convert the textual data to numeric data.
    parent_class_label : str or NoneType
        If this argument is a string, then it represents the class label
        that is the parent class of all of the sub-classes that will be
        distignuished and predicted by a classifier that you wish to build. 
        I.e., if you want to build a classifier for the children of the
        "Auto Type" class (which includes "Budget Cars", "Concept Cars",
        and "Luxury Cars" to name a few), then you simply have to pass in
        the "Auto Type" string to this parameter.

        If this argument is instead a NoneType object, then the 
        `class_data_retrival` function will NOT be called. Instead, the
        value that the `articles_df` variable will take on is whatever
        the user has specified with the keyword argument `articles_df`
        (see below for more information).
    **kwargs : dict
        This function is set up so that the user can specify what settings
        they would like for the various paramters for the functions being
        used. Accepted keyword arguments include:
            1. "articles_df" - This keyword argument provides the user a
                               bypass in the event that they would like
                               to be able to specify the exact DataFrame
                               of child class articles that whose article
                               contents will be converted to a numerical
                               form using a specified bag-of-words model.
                               NOTE that the value of this keyword argument
                               will be ignored if the `parent_class_label` 
                               argument is NOT a NoneType object.
            2. "lower_n_gram" - This keyword argument  allows the user to 
                                specify what they would like lower bound
                                for the setting of the n-gram parameter
                                in the used BOW model. Its default value
                                will be 1 when it is notspecified.
            3. "upper_n_gram" - This keyword argument allows the user to 
                                specify what they would like upper bound
                                for the setting of the n-gram parameter
                                in the used BOW model. Its default value
                                will be whatever "lower_n_gram" is.
            4. "upper_features" - This keyword argument allows the user
                                  to  specify the maximum number of BOW
                                  features the used BOW model can make.
                                  This is a useful parameter that helps
                                  prevent your classifier that gets trained
                                  on the returned BOW data from over-
                                  fitting to this data. Its default value
                                  is `None` meaning that the implemented 
                                  model will not have such an upper bound
                                  on the number of features it can create.
            5. "apply_pca" - This keyword argument allows the user to
                             specify whether they want the dataset that
                             results from the transformation done by the
                             BOW-model to then be transformed by a PCA
                             model and possibly have its number of features
                             reduced in size. Its default value is True.
            6. "pca_ncomps" - This keyword argument allows the user to
                              specify how many features to keep after a
                              PCA tranformation is applied. Its allowed
                              setting is identical to `n_components` 
                              argument that is specifed in the sklearn PCA
                              model instantiation. Its default value is
                              0.95 Note that this parameters is ignored
                              if the `apply_pca` keyword argument is set
                              to False.

    Returns
    -------
    to_return : (BOW Sklearn model, Numpy Array)
        This function returns a tuple; the first is the BOW imblearn model
        that was used to transform the data and the second is the resultant
        numerical feature matrix (which, possibly, also underwent another
        transformation by a PCA model).

    References
    ----------
    1. https://towardsdatascience.com/text-classification-in-python-dd95d264c802
    2. https://www.mygreatlearning.com/blog/bag-of-words/#sh3
    3. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    4. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    5. https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XynJxy2z0oI
    6. https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
    7. https://towardsdatascience.com/tf-idf-explained-and-python-sklearn-implementation-b020c5e83275
    8. https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb
    """
    # First, get values for the parameters that we will need to use for
    # the rest of the function.
    normalized_mode = "".join(mode.lower().split("-"))
    if not parent_class_label:
        articles_df = kwargs.get("articles_df", None)
        if isinstance(articles_df, type(None)):
            raise KeyError(
                "You have indicated that you would like to pass in your own Article DataFrame, but did NOT do so.")
    else:
        articles_df = class_data_retrival(parent_class_label)

    # Next, collect all the keywords that contain all of the ways in
    # which the user has specified their desired behavior for the
    # implemented bag of words model.
    lower_n_gram = kwargs.get("lower_n_gram", 1)
    upper_n_gram = kwargs.get("upper_n_gram", lower_n_gram)

    upper_features = kwargs.get("upper_features", None)

    apply_pca = kwargs.get("apply_pca", True)

    # Next, instantiate the correct model.
    if normalized_mode == "tf":
        # If the user would like to implement a Term-Frequency model.
        bow_model = CountVectorizer(ngram_range=(lower_n_gram, upper_n_gram),
                                    max_features=upper_features)
    elif normalized_mode == "tfidf":
        # If the user would like to implement a Term-Frequency Inverse-
        # Document-Frequency model.
        bow_model = TfidfVectorizer(ngram_range=(lower_n_gram, upper_n_gram),
                                    max_features=upper_features)

    # Now, fit the instantiated model with the list of pre-processed
    # article contents.
    article_pp_contents_list = articles_df.Preprocessed_Content.tolist()
    tf_idf_matrix = bow_model.fit_transform(article_pp_contents_list)

    # Perform any post-processing steps that the user may have specified
    if apply_pca:
        # If the user wishes to apply PCA with a pre-determined number
        # of components to keep.
        pca_model = PCA(n_components=kwargs.get("pca_ncomps", 0.95))
        new_tf_idf_matrix = pca_model.fit_transform(tf_idf_matrix.toarray())
        to_return = (bow_model, new_tf_idf_matrix)

    else:
        to_return = (bow_model, tf_idf_matrix)

    return to_return


def imbalance_handler(
        mode: str,
        parent_class_label: str):
    """
    Purpose
    -------
    The purpose of this function is to provide the user a tool that
    allows them to easily manipulate their training and/or test dataset
    so that it is  significantly more balanced between its classes. One
    would want to do this in order to improve the realiability of their
    classifier that will get trainined on this dataset (see 1. in the
    References section for more information about this).

    **Note, however, that if a class has only 5 or fewer article
    instances that belong to it, it will be dropped completely due to
    the fact that the SMOTE and ENN algorithms used in this function
    rely on at least 6 nearest-neighbors of a class to exist. If this
    class label is particularly important and you would like to keep it
    around, then obtain more data for it.**

    Parameters
    ----------
    mode : str
        This string allows the user to specify how they would like the
        imbalancing of the dataset to be handled. The available options
        include:
            1. "smote" - In this mode, the only algorithm that will be
                         implemented to make  the dataset more balanced
                         is the over-sampling algorithm SMOTE. See 1., 3.,
                         4., and 5. in the References section for more
                         information about this algorithm.
            2. "enn" - In this mode, the only algorithm that will be
                       implemented to make the dataset more balanced is
                       the under-sampling algorithm Edited-Nearest Neighbors
                       (ENN). See 1. and 6. for more information about
                       this algorithm.
            3. "smote-enn" - In this mode, this function will implement
                             both the SMOTE and ENN algorithms; SMOTE
                             will oversample to make the classes balanced
                             and ENN will under-sample to remove any newly
                             generated samples in the minority class(es)
                             that are not helpful. See 1. and 7. for more
                             information about the benefits of doing using
                             this 2-step process and for how this is
                             implemented in the imbalanced-learn module.
    parent_class_label : str
        This string represents the class label that is the Parent class
        of all of the sub-classes that will be distignuished and predicted
        by a classifier that you wish to build. I.e., if you want to build
        a classifier for the children of the "Auto Type" class (which
        includes "Budget Cars", "Concept Cars", and "Luxury Cars" to name
        a few), then you simply have to pass in the "Auto Type" string to
        this parameter.

    Returns
    -------
    to_return : (Sparse Numpy Array, Numpy Array)
        The former element represents the new feature matrix (where some
        rows correspond to the article instances that were synthetically
        generated if the user specifed for over-sampling to occur) and the
        latter element represents the new class labels. Note that the
        number of rows in both these array objects are the same since each
        row of the two correspond to the same (real or synthetic) article
        instance.

    References
    ----------
    1. https://towardsdatascience.com/guide-to-classification-on-imbalanced-datasets-d6653aa5fa23
    2. https://imbalanced-learn.readthedocs.io/en/stable/index.html
    3. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    4. https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
    5. https://www.kite.com/blog/python/smote-python-imbalanced-learn-for-oversampling/
    6. https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.EditedNearestNeighbours.html#imblearn.under_sampling.EditedNearestNeighbours
    7. https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html
    """
    # First, get values for the parameters that we will need to use for
    # the rest of the function.
    normalized_mode = "".join(mode.lower().split("-"))
    child_tier_lvl, raw_articles_df = class_data_retrival(
        parent_class_label, give_child_tier_lvl=True)

    # Before performing any transformations on our data, we need to
    # double check that it is suitable for the BOWs and balance
    # correcting model. If it is not, then perform any corrections
    # neccessary.
    child_tier_label = "Tier{}".format(child_tier_lvl)
    counts_of_classes = raw_articles_df[child_tier_label].value_counts()
    counts_checker = counts_of_classes.values <= 6
    num_with_less = counts_checker.sum()
    if num_with_less > 0:
        # If any of the classes that we are working with have 5 or fewer
        # articles in them. If this is the case, then we cannot use any
        # of the over/under-sampling algorithms that investigate the
        # characteristics of its 6 nearest-neighbors. Our current
        # solution is to simply drop this class from consideration.
        indicies_with_less = np.argwhere(counts_checker).flatten()
        labels_with_less = counts_of_classes.index.values[indicies_with_less]
        if num_with_less == 1:
            # If there is exactly 1 class labels that we are going to
            # have to remove from the DataFrame of articles.
            assert labels_with_less.size == 1
            label_to_remove = labels_with_less[0]
            articles_df = raw_articles_df[raw_articles_df[child_tier_label]
                                          != label_to_remove]
        elif num_with_less == 2:
            # If there are exactly 2 class labels that we are going to
            # have to remove from the DataFrame of articles.
            assert labels_with_less.size == 2
            conditions_to_remove = np.logical_and(
                raw_articles_df[child_tier_label] != labels_with_less[0],
                raw_articles_df[child_tier_label] != labels_with_less[1])
            articles_df = raw_articles_df[conditions_to_remove]
        else:
            # If there are 3 or more class labels that we are going to
            # have to remove from the DataFrame of articles.
            assert labels_with_less.size >= 3
            for i in range(len(labels_with_less)):
                #
                if i == 0:
                    # If we are on our first iteration. In this case, we
                    # need to instantiate the `conditions_to_remove`
                    # object with the first two labels that we want to
                    # remove.
                    conditions_to_remove = np.logical_and(
                    	raw_articles_df[child_tier_label] != labels_with_less[i],
                    	raw_articles_df[child_tier_label] != labels_with_less[i + 1])
                elif i > 1:
                    # If we are on either our third or further down
                    # iteration. If this is the case, then we know that
                    # the `conditions_to_remove` object has been
                    # instantiated. We just need to add on to it with
                    # the remaining labels that we would like to remove.
                    conditions_to_remove = np.logical_and(
                        conditions_to_remove, raw_articles_df[child_tier_label] \
                        != labels_with_less[i])
            articles_df = raw_articles_df[conditions_to_remove]

    else:
        # All the article counts for each class pass the test :).
        articles_df = raw_articles_df
    # Next, obtain your X (features) matrix and your y (labels) vector.
    _, featue_matrix = bag_of_words_converter(mode="tfidf",
                                              parent_class_label=None,
                                              articles_df=articles_df,
                                              upper_n_gram=2,
                                              upper_features=300,
                                              apply_PCA=True)
    labels_arr = np.array(
        articles_df[child_tier_label].tolist())

    # Next, implement the algorithm the user has specified.
    if normalized_mode == "smote":
        # If the user would first like to oversample with the SMOTE
        # algorithm.
        sm_model = SMOTE(random_state=169,
                         n_jobs=3)
        final_feature_matrix, final_labels_arr = sm_model.fit_resample(
            featue_matrix, labels_arr)
    elif normalized_mode == "enn":
        # If the user would like to undersample with the Tomek links
        # algorithm
        enn_model = EditedNearestNeighbours(sampling_strategy="auto",
                                            n_jobs=3)
        final_feature_matrix, final_labels_arr = enn_model.fit_resample(
            featue_matrix, labels_arr)
    elif normalized_mode == "smoteenn":
        # If the user would first like to oversample with SMOTE and then
        # improve on that new set of samples by undersampling with the
        # ENN algorithm

        # Instantiate the smoteenn object from imblearn that first
        # performs SMOTE and then ENN.
        sm_enn_model = SMOTEENN(random_state=169,
                                n_jobs=3)

        # Fit and resample with this pipeline object.
        final_feature_matrix, final_labels_arr = sm_enn_model.fit_resample(
            featue_matrix, labels_arr)

    to_return = (final_feature_matrix, final_labels_arr)

    return to_return


def bow_data_saver(
        parent_class_label: str,
        obtain_data=True,
        rel_path_to_bow_directory="../../data/final/BOW_data",
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to provide the user with a tool that
    will take all of the articles that belong to the children classes of
    `parent_class_label`, run them through a BOW model, handle any
    existing class imbalance, and then save the results to a `.npz` file
    for easy future access since following all of those steps is very
    expensive computationally. NOTE that the name of the file will
    simply be a cleaned up form of the string passed into the
    `parent_class_label` argument.

    Parameters
    ----------
    parent_class_label : str
        This string represents the class label that is the Parent class
        of all of the sub-classes that will be distignuished and predicted
        by a classifier that you wish to build. I.e., if you want to build
        a classifier for the children of the "Auto Type" class (which
        includes "Budget Cars", "Concept Cars", and "Luxury Cars" to name
        a few), then you simply have to pass in the "Auto Type" string to
        this parameter.
	obtain_data : Bool
        This Boolean allows the user to indicate how they would like to
        obtain the data that will ultimately be saved. If set to True (which
        is its default value), then this function will call the
        `imbalance_handler` function to obtain the balanced BOW data. If
        set to False, then the user is then expected to pass in what feature
        matrix and labels array to save to disk.
    rel_path_to_bow_directory : str
        This string represents the path to the directory where we wish to
        store the balanced BOW numerical data RELATIVE TO THIS SCRIPT; not
        relative to whatever directory you are in when using this function.
    **kwargs : dict
        The use of Keyword Arguments with this function is to specify the
        data that will be saved in the event that the user has specified
        that they would NOT like for the data to be obtained through other
        means. As such, the values of ANY Keyword Arguments will be ignored
        if the value of `obtain_data` is set to True. The accepted keyword
        arguments are:
            1. "feature_matrix" - This keyword argument allows the user 
                                  to specify what feature matrix to save.
            2. "labels_arr" - This keyword argument allows the user to
                              specify what labels array to save.

    Returns
    -------
    to_return : str
        This function returns one of two strings. One indicates that the
        saving of the desired data was successful while the other one
        indicates that that process in fact failed.

    Raises
    ------
    KeyError
        Keyword Arguments are required when the `obtain_data` argument is
        set to False. If these Keyword Arguments are not given, then a
        KeyError will be raised.

    References
    ----------
    1. https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    # First, get everything about the paths straightened out.
    script_path = PATH_OF_SCRIPT
    full_path_to_directory = os.path.join(script_path,
                                          rel_path_to_bow_directory)
    os.chdir(full_path_to_directory)

    # Next, determine how we are going to be obtaining the data that we
    # will soon be storing.
    if obtain_data:
        # If the user would like this function to call the imbalance
        # handler function (defined above) to obtain the data that we
        # will be saving.
        featue_matrix, labels_arr = imbalance_handler(
            mode="SmoteENN", parent_class_label=parent_class_label)
    else:
        # If the user doess NOT want the imbalance handler function to
        # be called. If this is the case, then they must then pass in
        # the data that they would like to saved in the keyword
        # arguments that this function accepts (see function
        # documentation for more information).
        featue_matrix = kwargs.get("featue_matrix", None)
        labels_arr = kwargs.get("labels_arr", None)
        if not all([featue_matrix, labels_arr]):
            # If NOT all of the required keyword arguments were
            # specified.
            error_message = "Required keyword arguments were not \
            specified. \nPlease see function documentation for more \
            information."
            raise KeyError(error_message)

    # Finally, save the data.
    save_file_name = parent_class_label.replace(" ", "_")
    np.savez("{}.npz".format(save_file_name),
             featue_matrix,
             labels_arr)

    # As a last check, let's just make sure the file actually got saved
    files_in_saving_dir = os.listdir()
    if "{}.npz".format(save_file_name) in files_in_saving_dir:
        to_return = "BOW data for child classes of {} saved successfully!"\
        .format(parent_class_label)
    else:
        to_return = "BOW data saving failed."

    return to_return


def bow_data_loader(
        parent_class_label: str, 
        rel_path_to_bow_directory="../../data/final/BOW_data"):
    """
    Purpose
    -------
    The purpose of this function is to provide an easy tool for the user
    to load in a particular bag of words (bow) dataset by simply
    specifying the class label of the parent class.
    
    Parameters
    ----------
    parent_class_label : str
        This string represents the class label that is the Parent class
        of all of the sub-classes that will be distignuished and predicted
        by a classifier that you wish to build. I.e., if you want to build
        a classifier for the children of the "Auto Type" class (which
        includes "Budget Cars", "Concept Cars", and "Luxury Cars" to name
        a few), then you simply have to pass in the "Auto Type" string to
        this parameter; you could also pass in "Auto_Type" to this example
        if desired.
    rel_path_to_bow_directory : str
        This string represents the path to the directory where we wish to
        store the balanced BOW numerical data RELATIVE TO THIS SCRIPT; not
        relative to whatever directory you are in when using this function.

    Returns
    -------
    to_return : (Numpy Array, Numpy Array)
        This function returns a tuple which contains the loaded-in feature
        matrix and the loaded-in labels array.
    """
    normalized_pcl = parent_class_label.replace(" ", "_")
    # First, navigate to neccessary directory
    script_path = PATH_OF_SCRIPT
    full_path_to_directory = os.path.join(script_path,
                                          rel_path_to_bow_directory)
    os.chdir(full_path_to_directory)
    
    # Obtain the correct file name to load in
    files_in_dir = os.listdir()
    file_name_list = [
        file for file in files_in_dir \
        if file[file.find("_")+1:-4:] == normalized_pcl
    ]
    assert len(file_name_list) == 1
    file_name = file_name_list[0]
    
    # Now, load in the data and save the results to return.
    numpy_load_obj = np.load(file_name)
    feat_mat, labs_ar = numpy_load_obj["arr_0"], numpy_load_obj["arr_1"]
    assert feat_mat.shape[0] == labs_ar.size
    to_return = (feat_mat, labs_ar)
    
    return to_return

