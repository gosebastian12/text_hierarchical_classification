#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jul 21

@author: Sebastian Gonzalez
"""

####################################
### Neccessary Import Statements ###
####################################
# data manipulation
import pandas as pd
import numpy as np
import swifter

# visualization tools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


####################################
### Define our Modular Functions ###
####################################
def node_counter(last_tier: str, data_frame: pd.DataFrame):
    """
    Purpose
    -------
    The purpose of this function is to determine how many articles
    belong to the class labels that make up the tier level that the user
    gets to specify.

    Parameters
    ----------
    last_tier : str
        This string represents the tier level that the user wishes to work
        with (i.e., `Tier3`. These labels that belong to this tier will
        be the labels that we will then investigate to determine how many
        articles belong to each.
    data_frame : Pandas DataFrame
        This is the DataFrame that will be used to perform the calculations
        and operations to determine the number of articles that belong to
        the specified labels.

    Returns
    -------
    to_return : dict
        The returned object is a Python dictionary whose keys are
        `Node_labels` and `Node_counts`. The first key points to all of
        the class labels that live in the specified tier level and the
        second key points to the number of articles that are in each of
        those labels; both of these are Python lists.

    Raises
    ------
    AssertionError
        This error gets raised when, for some reason, the number of
        articles labels that were found in the specified tier level does
        NOT match the number of article counts. We raise an error instead
        of just creating the dictionary that will be returned because such
        a case would result in there not be a  1-to-1 mapping between the
        two lists that make up the values of the dictionary that would be
        returned.
    """
    # Create the grouping tiers list
    grouping_list = {"Tier1": "Tier1",
                     "Tier2": ["Tier1", "Tier2"],
                     "Tier3": ["Tier1", "Tier2", "Tier3"],
                     "Tier4": ["Tier1", "Tier2", "Tier3", "Tier4"]}
    try:
        grouping_to_use = grouping_list[last_tier.title()]
    except KeyError:
        return "Invalid `last_tier` argument. \
        See function docstring for valid argument-values."

    # Do the group-by
    grouped_by_tiers_df = data_frame.groupby(by=grouping_to_use)
    counts_df = grouped_by_tiers_df.count()

    # Now get the value counts and their corresponding labels.
    if grouping_to_use == "Tier1":
        # If we only have to deal with a 1D-index.
        node_names_list = counts_df.index.tolist()
    else:
        # If we only have to deal with a multi-dimensional index.
        node_names_list = [multi_index[-1] for multi_index in counts_df.index]

    node_counts_list = counts_df.Article_URL.values.tolist()

    assert len(node_names_list) == len(node_counts_list)
    nodes_dict = {"Node_labels": node_names_list,
                  "Node_counts": node_counts_list}

    return nodes_dict


def histogram(data_frame: pd.DataFrame, x_arg: str, **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to create a histogram using Ploty
    and user-specified data in a way that defaults to additional plot
    formatting that helps make it look more visually apealing while also
    being flexibile enough to allow for the user to easily specify
    formatting that they want.

    Parameters
    ----------
    data_frame : Pandas DataFrame
        This DataFrame contains the data that will be plotted in the
        resulting figure.
    x_args : str
        Whenever we plot a histogram, we need to specify what the quantity
        is that will be binned up. This required argument allows the user
        to do just that.
    **kwargs : dict
        The use of keyword arguments in this function is to specify
        parameters of the histogram that control its functionality and
        apparence. If these arguments are not specified, then they will
        either default to their default values of `px.histogram()` or will
        default to pre-determined values. The list of accepted keyword
        arguments (and the values that they default to when not specified)
        are listed directly below. If you are trying to customize a certain
        part of the plot and not seeing any changes, it may be due to the
        fact that you are not passing in the correct keyword.
            1. `x_label` (str) - Plotly (and thus, to an extent, this
                                 function) will default to having the
                                 x-axis label be whatever `x_arg` is. The
                                 user can specify what they want this label
                                 to be instead by passing in a value to
                                 this acepted keyword argument.
            2. `y_label` (str) - Plotly (and thus, to an extent, this
                                 function) will default to having the
                                 y-axis label be "count" for histograms.
                                 The user can specify what they want this
                                 label to be instead by passing in a value
                                 to this acepted keyword argument.
            3. `plot_title` (str) - Ploty will default to having no plot
                                    title. This function will default to
                                    the title of the plot being "Histogram".
                                    If the user wishes to specify their
                                    own plot title, they can pass in a
                                    string to this accepted keyword
                                    argument.
            4. `num_bins` (int) - This accepted keyword argument allows
                                  for the user to specify how many bins
                                  they would like to be used to create
                                  the histogram.
            5. `lower_x` (int) - Plotly (and thus this function) default
                                 to the lower bound of the x-axis being
                                 the minimumof the passed in data. If the
                                 user wishes to specify their own lower
                                 bound for x, they can do so with this
                                 accepted keyword argument.
            6. `upper_x` (int) - Plotly (and thus this function) default
                                 to the upper bound of the x-axis being
                                 the maximum of the passed in data. If the
                                 user wishes to specify their own upper
                                 bound for x, they can do so with this
                                 accepted keyword argument.
            7. `lower_y` (int) - Plotly (and thus this function) default
                                 to the lower bound of the y-axis being
                                 0. If the user wishes to specify their
                                 own lower bound for y, they can do so
                                 with this accepted keyword argument.
            8. `upper_y` (int) - Plotly (and thus this function) default
                                 to the upper bound of the y-axis being
                                 the maximum of the determined bin counts.
                                 If the user wishes to  specify their own
                                 upper bound for y, they can do so with
                                 this accepted keyword argument.
            9. `to_normalize` (str) - This is exactly theargument `histnorm`
                                      of px.histogram. See reference number
                                      1 below for its accepted values.
            10. `to_cumaliate` (Bool) - Plotly (and thus this function)
                                        will default to the histogram NOT
                                        representing a cumulative
                                        distribution. If the user wishes
                                        to receive a histogram that does
                                        do this, then they can set this
                                        keyword argument to True.
            11. `opacity_lvl` (float) - Plotly will, by default set the
                                        opacity level of the figure to 1.
                                        However, this function will default
                                        this value to 0.65. If the user
                                        wishes to specify their own value,
                                        they can do so by passing in a
                                        float between 0 and 1 to this
                                        accepted keywordargument.
            12. `bar_color` (str) - Plotly will, by default, set the color
                                    of the bars to blue. This function will
                                    default to "rgb(95, 62, 227)" which
                                    is a lighter shade of purple. If the
                                    user wish to specify their own color,
                                    they can do so by using this accepted
                                    keyword argument.
            13. `plot_width` (int) - This accepted keyword argument allows
                                     the user to specify the width they
                                     would like for the resulting figure
                                     in pixels. This defaults to 1300 when
                                     not specified.
            14. `plot_height` (int) - This accepted keyword argument allows
                                      the user to specify the height they
                                      would like for the resulting figure
                                      in pixels. This defaults to 800 when
                                      not specified.

    Returns
    -------
    This function returns a `plotly.graph_objs._figure.Figure` object
    that is made up of the plotted histogram. NOTE that the bars of
    the histogram will be enclosed by dark-gray lines for visual
    appeal.

    References
    ----------
    1. https://plotly.github.io/plotly.py-docs/generated/plotly.express.histogram.html
    2. https://plotly.com/python/histograms/
    """
    # Collect any keywords that may have been passed into the
    # function for plot customization purposes.
    # Axes labels.
    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", None)

    axes_label_dict = {"count": "Count"}
    # if NONE of these conditional statements pass, this object
    # will NOT be modified.
    if np.all([x_label, y_label]):
        # If the user passed in keyword arguments for BOTH the x and y
        # axes.
        axes_label_dict["count"] = y_label
        axes_label_dict[x_arg] = x_label
    elif x_label:
        # If the user passed in keyword arguments for JUST the x axis.
        axes_label_dict[x_arg] = x_label
    elif y_label:
        # If the user passed in keyword arguments for JUST the y axis.
        axes_label_dict["count"] = y_label
    else:
        # If the user DID NOT pass any keyword arguments to control
        # any of the axes labels.
        cleaned_x_label = " ".join(x_arg.split("_")).title()
        axes_label_dict[x_arg] = cleaned_x_label

    plot_title = kwargs.get("plot_title", "Histogram")

    # Bin control.
    num_bins = kwargs.get("num_bins", None)

    lower_x = kwargs.get("lower_x", None)
    upper_x = kwargs.get("upper_x", None)
    lower_y = kwargs.get("lower_y", None)
    upper_y = kwargs.get("upper_y", None)

    # Probability control.
    to_normalize = kwargs.get("to_normalize", None)
    to_cumaliate = kwargs.get("to_cumaliate", None)

    # Apperance control.
    opacity_lvl = kwargs.get("opacity_lvl", 0.65)
    bar_color = kwargs.get("bar_color", None)

    plot_width = kwargs.get("plot_width", 1300)
    plot_height = kwargs.get("plot_height", 800)

    # Create and update the histogram.
    hist_obj = px.histogram(data_frame,
                            x=x_arg,
                            color=bar_color,
                            hover_name=None,
                            hover_data=None,
                            labels=axes_label_dict,
                            opacity=opacity_lvl,
                            histnorm=to_normalize,
                            range_x=[lower_x, upper_x],
                            range_y=[lower_y, upper_y],
                            cumulative=to_cumaliate,
                            nbins=num_bins,
                            width=plot_width,
                            height=plot_height)
    hist_obj.update_traces(marker_color='rgb(195, 62, 227)',
                           marker_line_color='rgb(0, 0, 0)',
                           marker_line_width=.75)
    hist_obj.update_layout(font={"family": "Times New Roman",
                                 "size": 18,
                                 "color": "black"},
                           title={"text": plot_title,
                                  "y": 0.965,
                                  "x": 0.5,
                                  "xanchor": "center",
                                  "yanchor": "top"})

    to_return = hist_obj

    return to_return


def bar_plot(data_frame: pd.DataFrame, x_arg: str, y_arg: str, **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to create a bar chart using Ploty
    and user-specified data in a way that defaults to additional plot
    formatting that helps make it look more visually apealing while also
    being flexibile enough to allow for the user to easily specify
    formatting that they want.

    Parameters
    ----------
    data_frame : Pandas DataFrame
        This DataFrame contains the data that will be plotted in the
        resulting figure.
    x_arg : str
        This required argument allows the user to specify what column in
        the passed in DataFrame to use to determine what will go on the
        x-axis.
    y_arg : str)
        This required argument allows the user to specify what column in
        the passed in DataFrame to use to determine what will go on the
        y-axis.
    **kwargs : dict
        The use of keyword arguments in this function is to specify
        parameters of the bar chart that control its functionality and
        apparence. If these arguments are not specified, then they will
        either default to their default values of `px.bar()` or will
        default to pre-determined values. The list of accepted keyword
        arguments (and the values that they default to when not specified)
        are listed directly below. If you are trying to customize a certain
        part of the plot and not seeing any changes, it may be due to the
        fact that you are not passing in the correct keyword.
            1. `x_label` (str) - Plotly (and thus, to an extent, this
                                 function) will default to having the
                                 x-axis label be whatever `x_arg` is. The
                                 user can specify what they want this
                                 label to be instead by passing in a value
                                 to this acepted keyword argument.
            2. `y_label` (str) - Plotly (and thus, to an extent, this
                                 function) will default to having the
                                 y-axis label be whatever `y_arg` is. The
                                 user can specify what they want this label
                                 to be instead by passing in a value to
                                 this acepted keyword argument.
            3. `plot_title` (str) - Ploty will default to having no plot
                                    title. This function will default to
                                    the title of the plot being "Histogram".
                                    If the user wishes to specify their
                                    own plot title, they can pass in a
                                    string to this accepted keyword
                                    argument.
            4. `lower_x` (int) - Plotly (and thus this function) default
                                 to the lower bound of the x-axis being
                                 the minimum of the passed in data. If the
                                 user wishes to specify their own lower
                                 bound for x, they can do so with this
                                 accepted keyword argument.
            5. `upper_x` (int) - Plotly (and thus this function) default
                                 to the upper bound of the x-axis being
                                 the maximum of the passed in data. If
                                 the user wishes to specify their own
                                 upper bound for x, they can do so with
                                 this accepted keyword argument.
            6. `lower_y` (int) - Plotly (and thus this function) default
                                 to the lower bound of the y-axis being
                                 0. If the user wishes to specify their
                                 own lower bound for y, they can do so
                                 with this accepted keyword argument.
            7. `upper_y` (int) - Plotly (and thus this function) default
                                 to the upper bound of the y-axis being
                                 the maximum of the determined bin counts.
                                 If the user wishes to  specify their own
                                 upper bound for y, they can do so with
                                 this accepted keyword argument.
            8. `opacity_lvl` (float) - Plotly will, by default set the
                                       opacity level of the figure to 1.
                                       However, this function will default
                                       this value to 0.65. If the user
                                       wishes to specify their own value,
                                       they can do so by passing in a
                                       float between 0 and 1 to this
                                       accepted keyword argument.
            9. `bar_color` (str) - Plotly will, by default, set the color
                                   of the bars to blue. This function will
                                   default to "rgb(95, 62, 227)" which is
                                   a lighter shade of purple. If the user
                                   wish to specify their own color, they
                                   can do so by using this accepted keyword
                                   argument.
            10. `plot_width` (int) - This accepted keyword argument allows
                                     the user to specify the width they
                                     would like for the resulting figure
                                     in pixels. This defaults to 1300 when
                                     not specified.
            11. `plot_height` (int) - This accepted keyword argument allows
                                      the user to specify the height they
                                      would like for the resulting figure
                                      in pixels. This defaults to 800 when
                                      not specified.
            12. `x_label_tilt` (int) - This accepted keyword argument allows
                                       the user to specify the angle in
                                       which they would like for the x-axis
                                       labels to be rotated. This is
                                       particularly useful for when these
                                       labels are long strings of text.
            13. `desired_orientation` (str) - This accepted keyword
                                              argument allows for the user
                                              to specify whether they want
                                              the bars to be vertical
                                              (from the  x-axis;  "v") or
                                              horizontal (from the y-axis;
                                              "h"). It will default to
                                              "v" when not spcified.

    Returns
    -------
    to_return : `plotly.graph_objs._figure.Figure
        This object is made up of the plotted bar chart. NOTE that the
        bars of the histogram will be enclosed by dark-gray lines for
        visual appeal.

    References
    ----------
    1. https://plotly.com/python/bar-charts/
    2. https://plotly.com/python-api-reference/generated/plotly.express.bar.html
    """
    to_return = None
    # First, collect all of the keyword arguments
    # Axes labels.
    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", None)

    axes_label_dict = {}
    # if NONE of these conditional statements pass, this object
    # will NOT be modified.
    if np.all([x_label, y_label]):
        # If the user passed in keyword arguments for BOTH the x and y
        # axes.
        axes_label_dict[x_arg] = x_label
        axes_label_dict[y_arg] = y_label
    elif x_label:
        # If the user passed in keyword arguments for JUST the x axis.
        axes_label_dict[x_arg] = x_label
    elif y_label:
        # If the user passed in keyword arguments for JUST the y axis.
        axes_label_dict[y_arg] = y_label
    else:
        # If the user DID NOT pass any keyword arguments to control
        # any of the axes labels.
        cleaned_x_label = " ".join(x_arg.split("_")).title()
        axes_label_dict[x_arg] = cleaned_x_label

        cleaned_y_label = " ".join(y_arg.split("_")).title()
        axes_label_dict[y_arg] = cleaned_y_label

    plot_title = kwargs.get("plot_title", "Bar Chart")

    # Bar control.
    lower_x = kwargs.get("lower_x", None)
    upper_x = kwargs.get("upper_x", None)
    lower_y = kwargs.get("lower_y", None)
    upper_y = kwargs.get("upper_y", None)

    # Apperance control.
    opacity_lvl = kwargs.get("opacity_lvl", 0.65)
    bar_color = kwargs.get("bar_color", "rgb(195, 62, 227)")

    plot_width = kwargs.get("plot_width", 1300)
    plot_height = kwargs.get("plot_height", 800)

    x_label_tilt = kwargs.get("x_label_tilt", 0)

    desired_orientation = kwargs.get("desired_orientation", "v")
    if desired_orientation == "h":
        x_arg, y_arg = y_arg, x_arg
        text_arg = x_arg
    elif desired_orientation == "v":
        text_arg = y_arg

    # Now create the plot.
    bar_obj = px.bar(data_frame=data_frame,
                     x=x_arg,
                     y=y_arg,
                     text=text_arg,
                     labels=axes_label_dict,
                     opacity=opacity_lvl,
                     orientation=desired_orientation,
                     range_x=[lower_x, upper_x],
                     range_y=[lower_y, upper_y],
                     width=plot_width,
                     height=plot_height)
    bar_obj.update_traces(marker_color=bar_color,
                          marker_line_color='rgb(0, 0, 0)',
                          marker_line_width=.75)
    bar_obj.update_layout(font={"family": "Times New Roman",
                                "size": 18,
                                "color": "black"},
                          title={"text": plot_title,
                                 "y": 0.965,
                                 "x": 0.5,
                                 "xanchor": "center",
                                 "yanchor": "top"},
                          xaxis_tickangle=x_label_tilt)
    to_return = bar_obj

    return to_return


def violin_plot(
        label_name,
        data_frame: pd.DataFrame,
        x_arg: str,
        mode="word",
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to create a violin chart using Ploty
    and user-specified data in a way that defaults to additional plot
    formatting that helps make it look more visually apealing while also
    being flexibile enough to allow for the user to easily specify
    formatting that they want.

    Parameters
    ----------
    label_name : str or list or Bool
        This argument allows the user to specify how the function will
        determine what will be plotted. If it is simlpy a string, then
        that represents the label within the DataFrame that will be used
        to take a subset of the passed-in DataFrame to plot. If it is a
        list of strings, then that represents multiple labels that will
        be used to create a (larger) subset of the passed-in DataFrame to
        plot. If it is a Boolean (either True or False), then there will
        be no subsetting of the passed-in DataFrame meaning that it will
        be used to plot as it is.
    data_frame : Pandas DataFrame
        This DataFrame contains the data that will be used to determine
        what will be plotted in the resulting figure.
    x_arg : str
        This required argument allows the user to specify what column in
        the passed in DataFrame to use to determine what will go on the
        x-axis.
    y_arg : str
        This required argument allows the user to specify what column in
        the passed in DataFrame to use to determine what will go on the
        y-axis.
    mode : str; default "word"
    **kwargs : dict
        The use of keyword arguments in this function is to specify
        parameters of the violin plot that control  its functionality and
        apparence. If these arguments are not specified, then they will
        either default to their default values of `go.violin()` or will
        default to pre-determined values. The list of accepted keyword
        arguments (and the values that they default to when not specified)
        are listed directly below. If you are trying to customize a certain
        part of the plot and not seeing any changes, it may be due to the
        fact that you are not passing in the correct keyword.
            1. `x_label` (str) - Plotly (and thus, to an extent, this
                                 function) will default to having the
                                 x-axis label be whatever `x_arg` is. The
                                 user can specify what they want this
                                 label to be instead by passing in a value
                                 to this accepted keyword argument.
            2. `y_label` (str) - Plotly (and thus, to an extent, this
                                function) will default to having the y-axis
                                label be whatever `y_arg` is. The user can
                                specify what they want this label to be
                                instead by passing in a value to this
                                acepted keyword argument.
            3. `plot_title` (str) - Ploty will default to having no plot
                                   title. This function will default to
                                   the title of the plot being "Histogram".
                                   If the user wishes to specify their own
                                   plot title, they can pass in a string
                                   to this accepted keyword argument.
            4. `box_on` (Bool) -
            5. `mean_on` (Bool) -
            6. `points_specification` (str) -
            7. `upper_y` (int) - Plotly (and thus this function) default
                                 to the upper bound of the y-axis being
                                 the maximum of the determined bin counts.
                                 If the user wishes to  specify their own
                                 upper bound for y, they can do so with
                                 this accepted keyword  argument.
            8. `plot_width` (int) - This accepted keyword argument allows
                                    the user to specify the width they
                                    would like for the resulting figure
                                    in pixels. This defaults to 1300 when
                                    not specified.
            9. `plot_height` (int) - This accepted keyword argument allows
                                     the user to specify the height they
                                     would like for the resulting figure
                                     in pixels. This defaults to 800 when
                                     not specified.
            10. `opacity_lvl` (float) - Plotly will, by default set the
                                        opacity level of the figure to 1.
                                        However, this function will default
                                        this value to 0.65. If the user
                                        wishes to specify their own value,
                                        they can do so by passing in a
                                        float between 0 and 1 to this
                                        accepted keyword argument.
            11. `plot_color` (str) -
            12. `outside_line_color` (str) -
            13. `desired_orientation` (str) - This accepted keyword
                                              argument allows for the user
                                              to specify whether they want
                                              the bars tobe vertical (from
                                              the x-axis; "v") or horizontal
                                              (from the y-axis;  "h"). It
                                              will  default to "v" when not
                                              spcified.

    Returns
    -------
    to_return : plotly.graph_objs._figure.Figure
        This object is made up of the plotted violin chart. NOTE that the
        "violin" will be enclosed by a dark-gray line for visual appeal.

    References
    ----------
    1. https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Violin.html
    2. https://plotly.com/python/violin/
    """
    # Work with the chunk of the DataFrame that corresponds to this
    # label.
    # First, we have to get this chunk.
    if isinstance(label_name, str):
        # If the user is ONLY interested in seeing the distribution
        # of articles in one nodes.
        chunk_df_0 = data_frame[data_frame.Label == label_name]
        chunk_df_1 = chunk_df_0.copy()
    elif isinstance(label_name, list):
        # If the user in interested in seeing the distribution of
        # article in MULTIPLE nodes.
        chunk_dfs_list = [data_frame[data_frame.Label == label]
                          for label in label_name]
        chunk_df_0 = pd.concat(objs=chunk_dfs_list,
                               ignore_index=True)
        chunk_df_1 = chunk_df_0.copy()
    else:
        # If the use instead wishes to specify which article rows they
        # would like to work with by passing in their own DataFrame.
        # If this is done, then the function will simply use whatever
        # was passed in the `data_frame` argument.
        chunk_df_0 = data_frame.copy()
        chunk_df_1 = chunk_df_0.copy()

    # Second, we have to figure out the character or word counts (which
    # depends on the mode that the user has specified in the `mode`
    # argument).
    if mode == "word":
        # if the user would like to parse the preprocessed content by
        # WORDS.
        def word_counter_func(row): return len(
            row.Preprocessed_Content.split())

        def unique_word_counter_func(row): return len(
            set(row.Preprocessed_Content.split())
        )
        chunk_df_1["pp_word_count"] = chunk_df_0.swifter.apply(
            word_counter_func, axis=1)
        chunk_df_1["pp_unique_word_count"] = chunk_df_0.swifter.apply(
            unique_word_counter_func, axis=1)
        y_arg = "pp_word_count"
    else:
        # if the user would like to parse the preprocessed content by
        # CHARACTERS.
        chunk_df_1["pp_char_count"] = None
        chunk_df_1["pp_unique_char_count"] = None
        y_arg = "pp_char_count"

    # Collect keyword arguments
    # Labels.
    axes_label_dict = {}
    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", None)
    # if NONE of these conditional statements pass, this object
    # will NOT be modified.
    if np.all([x_label, y_label]):
        # If the user passed in keyword arguments for BOTH the x and y
        # axes.
        axes_label_dict[x_arg] = x_label
        axes_label_dict[y_arg] = y_label
    elif x_label:
        # If the user passed in keyword arguments for JUST the x axis.
        axes_label_dict[x_arg] = x_label
    elif y_label:
        # If the user passed in keyword arguments for JUST the y axis.
        axes_label_dict[y_arg] = y_label
    else:
        # If the user DID NOT pass any keyword arguments to control
        # any of the axes labels.
        cleaned_x_label = " ".join(x_arg.split("_")).title()
        axes_label_dict[x_arg] = cleaned_x_label

        cleaned_y_label = " ".join(y_arg.split("_")).title()
        axes_label_dict[y_arg] = cleaned_y_label

    plot_title = kwargs.get("plot_title", "Violin Plot")

    # Plot functionality.
    box_on = kwargs.get("box_on", True)
    mean_on = kwargs.get("mean_on", True)
    points_specification = kwargs.get("points_on", "outliers")

    # Plot apperance
    plot_width = kwargs.get("plot_width", 1300)
    plot_height = kwargs.get("plot_height", 800)

    opacity_lvl = kwargs.get("opacity_lvl", 0.5)
    plot_color = kwargs.get("plot_color", "lightseagreen")
    outside_line_color = kwargs.get("outside_line_color", "black")

    desired_orientation = kwargs.get("desired_orientation", "v")
    if desired_orientation == "h":
        x_arg, y_arg = y_arg, x_arg

    # Create violin plot.
    violin_obj = go.Violin(x=chunk_df_1[x_arg],
                           y=chunk_df_1[y_arg],
                           box_visible=box_on,
                           meanline_visible=mean_on,
                           points=points_specification,
                           opacity=opacity_lvl,
                           fillcolor=plot_color,
                           line_color=outside_line_color,
                           orientation=desired_orientation)
    fig_obj = go.Figure(data=violin_obj)
    fig_obj.update_layout(width=plot_width,
                          height=plot_height,
                          font={"family": "Times New Roman",
                                "size": 18,
                                "color": "black"},
                          title={"text": plot_title,
                                 "y": 0.965,
                                 "x": 0.5,
                                 "xanchor": "center",
                                 "yanchor": "top"},
                          xaxis_title=axes_label_dict.get(x_arg, x_arg),
                          yaxis_title=axes_label_dict.get(y_arg, y_arg))

    to_return = (violin_obj, fig_obj)
    return to_return


def subplotter(
        data_frames_list: list,
        subplot_shape: list,
        mode="violin",
        **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to take a list of DataFrames that
    the user has compiled and use them to create a Plotly subplot that
    contains several plots in the specified shape.

    Parameters
    ----------
    data_frames_list : list of Pandas DataFrames
        This list contains the DataFrames that specify the data to be
        plotted on each of these subplots. NOTE that the length of this
        list does not have to be equal to product of the two numbers
        specified in the `subplot_shape` parameter; such a case will mean
        that each subplot will contain multiple graphs.
    subplot_shape : list or tuple of ints
        This list specifies what shape the subplot figure will use. The
        first element specifies how many rows while the second element
        specifies how many columns will be used.
    mode : str
        This string allows the user to specify what kinds of graphs will
        make up the subplot figure. The current options are "violin"
        (which is currently its default value), "bar", and "histogram".
    **kwargs: dict
        The use of keyword arguments in this function is to specify
        parameters of the subplot that control its functionality and
        apparence. If these arguments are not specified, then they will
        either default to their default values of a subplot (see reference
        1.) or will default to pre-determined values. The list of accepted
        keyword arguments (and the values that they default to when not
        specified) are listed directly below. If you are trying to customize
        a certain part of the plot and not seeing any changes, it may be
        due to the fact that you are not passing in the correct keyword.
            1. `plot_title` (str) - Ploty will default to having no plot
                                    title. This function will default to
                                    the title of the plot being "Histogram".
                                    If the user wishes to specify their
                                    own plot title, they can pass in a
                                    string to this accepted keyword
                                    argument.
            2. `plot_width` (int) - This accepted keyword argument allows
                                    the user to specify the width they
                                    would like for the resulting figure
                                    in pixels. This defaults to 1300 when
                                    not specified.
            3. `plot_height` (int) - This accepted keyword argument allows
                                     the user to specify the height they
                                     would like for the resulting figure
                                     in pixels. This defaults to 800 when
                                     not specified.

    Returns
    -------
    to_return : plotly.graph_objs._figure.Figure
        This object is made up of the plotted subplot figure.

    Raises
    ------
    AssertionError
        Such an error will be raised if the attempt to group the specified
        DataFrames to be put on subplots together fails.

    References
    ----------
    1. https://plotly.com/python/subplots/
    2. https://plotly.com/python/legend/
    """
    to_return = None
    # Instantiate the figure object
    num_rows, num_cols = subplot_shape
    fig = make_subplots(rows=num_rows, cols=num_cols)

    # Now compile the plots.
    new_dfs_list = []
    three_left = True
    start_index, end_index = 0, 3
    while three_left:
        #
        dfs_to_concat = data_frames_list[start_index:end_index:]
        assert len(dfs_to_concat) == 3
        new_df = pd.concat(dfs_to_concat, ignore_index=True)
        new_dfs_list.append(new_df)

        start_index = end_index
        end_index += 3

        remaining_length = len(data_frames_list[start_index::])
        three_left = False if remaining_length < 3 else True
    assert 0 <= remaining_length < 3
    if remaining_length == 1:
        # If we only have one article left then there is NO NEED to do
        # any concatenation of multiple DataFrames.
        new_dfs_list.append(data_frames_list[-1])
    elif remaining_length == 2:
        # If we only have two articles left, we still have to concat
        # them together before appending them to the new list.
        pair_of_dfs = data_frames_list[start_index::]
        new_df = pd.concat(pair_of_dfs, ignore_index=True)
        new_dfs_list.append(new_df)

    violin_plots_list = [
        violin_plot(
            False,
            df,
            "Tier1",
            x_label="Class Label",
            desired_orientation="h")[0] for df in new_dfs_list]

    # Add the plots to the figure.
    row_index, col_index = 1, 1
    # NOTE that these row and column indicies start at 1.
    for violin_obj in violin_plots_list:
        fig.add_trace(violin_obj,
                      row=row_index,
                      col=col_index)
        col_index += 1
        if col_index > num_cols:
            # If we have filled out the row that we are on.
            row_index += 1
            col_index = 1

    # Make the plot prettier. Compile keyword arguments.
    # Labels.
    plot_title = kwargs.get("plot_title", "Violin Plot")

    # Plot apperance
    plot_width = kwargs.get("plot_width", 1300)
    plot_height = kwargs.get("plot_height", 800)

    # Make it pretty now.
    fig.update_layout(width=plot_width,
                      height=plot_height,
                      font={"family": "Times New Roman",
                            "size": 18,
                            "color": "black"},
                      title={"text": plot_title,
                             "y": 0.965,
                             "x": 0.5,
                             "xanchor": "center",
                             "yanchor": "top"},
                      showlegend=False)
    to_return = fig

    return to_return


def group_plotter(group_name: str, data_frame: pd.DataFrame):
    """
    Purpose
    -------
    The purpose of this function is to provide a simple tool that allows
    the user to generate bar and violin plots that describe important
    statistics for whatever group of classes/subclasses they may be
    interested in. The use-case in mind was to allow for a quick
    investigation on the distribution of article counts and article
    characteristics acoross the different class labels you are
    attempting to build a text classifier to predict.

    Parameters
    ----------
    group_name : str
        This string specifies where along the class hierachy the group of
        class labels they are investigating live.
    data_frame : Pandas DataFrame
        This DataFrame contains the data that will be used to determine
        what will be plotted in the resulting figures.
    Returns
    -------
    to_return (plotly.graph_objs._figure.Figure, plotly.graph_objs._figure.Figure)
        This function returns a tuple containing two different
        `plotly.graph_objs._figure.Figure` objects. The former corresponds
        to the created bar chart while the latter corresponds to the created
        subplot of violin plots.

    References
    ----------
    1. https://numpy.org/doc/stable/reference/generated/numpy.ceil.html#numpy.ceil
    """
    # First, define some important variables that we will need later
    # on.
    if not isinstance(group_name, str):
        raise TypeError(
            "The argument for the variable `group_name` must be a string!")
    group_name = group_name.lower()
    subplot_num_shape_mapper = {2: [2, 1],
                                3: [3, 1],
                                4: [2, 2],
                                5: [3, 2],
                                6: [3, 2],
                                7: [4, 2],
                                8: [4, 2],
                                9: [3, 3],
                                10: [5, 2]}
    # Second, let's compile the articles that correspond to the group
    # that we are interested in.
    if group_name == "parents":
        # If the user is interested in taking a look at how many
        # articles belong to each parent class.
        group_count_dict = node_counter(last_tier="Tier1",
                                        data_frame=data_frame)

        # Create the bar plot.
        group_count_df = pd.DataFrame(group_count_dict)
        bar_obj = bar_plot(
            group_count_df,
            x_arg="Node_labels",
            y_arg="Node_counts",
            desired_orientation="h",
            plot_title="Distribution of Articles In Each Parent Tier")

        # Create the violin plot. More specifically, create the subplot
        # of violin plots.
        group_classes = group_count_dict["Node_labels"]
        num_of_group_classes = len(group_classes)
        num_of_subplots = np.ceil(num_of_group_classes / 3)
        if num_of_subplots == 1:
            # If all of the Violin Graphs that the user wishes to create
            # fit on one single plot that has at most 3 such graphs,
            # then there is NOT a need to call the `subplotter`
            # function. All we need to call is `violin_plot`.
            # violin_obj = violin_plot()
            pass
        else:
            # If we need more than one plot to display all of the Violin
            # graphs that the user would like to, then we now need to
            # use the `subplotter` function.
            subplot_shape = subplot_num_shape_mapper[num_of_subplots]
            group_article_dfs_list = [
                data_frame[data_frame.Tier1 == label] for label in group_classes]
            violin_obj = subplotter(data_frames_list=group_article_dfs_list,
                                    subplot_shape=subplot_shape)

    to_return = (bar_obj, violin_obj)

    return to_return
