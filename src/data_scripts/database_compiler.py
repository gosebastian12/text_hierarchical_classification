#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13

@author: Sebastian Gonzalez
"""

####################################
### Neccessary Import Statements ###
####################################
# Modules
import os
import json
import numpy as np
import pandas as pd
import psycopg2

# Mapper that we will be using below.
PATH_OF_SCRIPT = os.path.dirname(__file__)
REL_PATH_OF_JSON_MAPPER = "../../data/raw/iab_taxonomy-v2.json"
FULL_PATH_OF_JSON_MAPPER = os.path.join(PATH_OF_SCRIPT, REL_PATH_OF_JSON_MAPPER)
JSON_FILE = open(FULL_PATH_OF_JSON_MAPPER)
TABLE_TAXONOMY_MAPPER = json.load(JSON_FILE)
JSON_FILE.close()

####################################
### Define our Modular Functions ###
####################################
def connect_to_db(
        database: str,
        hostname: str,
        port: str,
        userid: str,
        passwrd: str):
    """
    Purpose
    -------
    The purpose of this function is to establish a connection between
    the user and a specified database so that SQL queries can be 
    performed within a Python framework. NOTE that the Python package 
    used to do this is Psycopg 2 (see references 1 and 2 in the 
    References section of this docstring for links about this pacakge).

    Parameters
    ----------
    database : str
      This string specifies the type of database that you are trying to
      connect to (i.e., Postgres).
    hostname : str
      This string specifies where the database is being hosted (i.e., a
      AWS link).
    port : str
      This string specifies what entrance port is needed to connect to
      the specified database.
    userid : str
      This string allows the user to specify any neccessary credentials
      they may need to be allowed to connect to the specified database.
      With this argument, the user specifies their username.
    passwrd : str
      This string allows the user to specify any neccessary credentials
      they may need to be allowed to connect to the specified database.
      With this argument, the user specifies their username.

    Returns
    -------
    to_return : (Psycopg connection object, Psycopg cursor object)
      The returned object is a **tuple** which contains a Psycopg 
      connection and cursor objects (see references 3 and 4 in the 
      References section of this docstring for links that describe these
      class objects). These are the objects that allow the user to perform
      SQL queries and other database management tasks in Python.

    References
    ----------
    1. https://pypi.org/project/psycopg2/
    2. https://www.psycopg.org/docs/
    3. https://www.psycopg.org/docs/connection.html
    4. https://www.psycopg.org/docs/cursor.html
    """
    # create string that will be used to connect to the database.
    conn_string = "host={} port={} dbname={} user={} password={}".format(
        hostname, port, database, userid, passwrd)
    # connect to the database with the connection string
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    # commits all queries you execute
    # instantiate cursor object that will be used to execute all 
    # queries.
    cursor = conn.cursor()

    return conn, cursor


def get_table(table_name: str, cursor, as_df=True, *args, **kwargs):
    """
    Purpose
    -------
    The purpose of this function is to perform a SQL query by simply
    specifying a few arguments. This function allows for the user to
    specify any specific columns they may want from the table they are
    obtaining data from.

    Parameters
    ----------
    table_name : str
      This string is what allows the user to specify which table they want
      to obtain data from. Note that this table must exist in the database
      for the query to be successful.
    cursor : Psycopg cursor object
      This object represents the connection that was previously made to
      a desired database. This is what will be used to execute the query
      to obtain data from the  specified table. See connect_to_db function
      documentation for more  information about this type of object.
    as_df : Bool; default True 
      This Boolean argument allows for the user to specify if they want
      the  resulting data to be placed in a Pandas DataFrame or to simply
      be returned as a list of tuples as is standard for SQL queries ran
      with Psycopg. NOTE that this argument defaults to True if it is not
      specified.
    *args : list
      The function is set up to allow for positional arguments to specify
      particular columns that the user only wants from the data table.
      Of course, an error will be raised if these columns do not exist
      in that data table.
    **kwargs : dict
      The function is set up to allow for keyword arguments to specify
      particular columns that the user only wants from the data table.
      Of course, an error will be raised if these columns do not exist 
      in that data table.

    Returns
    -------
    to_return : Pandas DataFrame or list
      This function can return one of two objects and which is dependent 
      on the value of the argument `as_df`. If its value is True (which
      is its default setting), then the function will attempt to convert
      the result into a Pandas DataFrame whose column labels are the exact
      same as that of the data table that a SQL query was sent to (the 
      code that makes up this attempt can be found in the `tuples_to_df`
      function below). If this attempt fails or the value of `as_df` is 
      set to False, then the returned object will simply be a list of 
      tuples (where each tuple represents a row) as is standard for Psycopg
      queries.

    References
    ----------
    1. https://www.w3schools.com/sql/sql_null_values.asp
    """
    to_return = None
    # define query
    if not args and not kwargs:
    	# If the user just wants all of the columns for the specified
    	# table that they are working with.
        query_str = "SELECT * FROM {} WHERE content IS NOT NULL".format(table_name)
        	# We don't want a row if there's no article for us to work
        	# with. Take advantage of this fact to save some time with
        	# this potentially large query.
    elif args:
        # If the user passed in POSITIONAL arguments to specify which
        # columns of the specified table they want to pull data for.
        formated_to_throw_in = "SELECT {} FROM {}".format(
            "{}, " * (len(args) - 1) + "{}", table_name)
        query_str = formated_to_throw_in.format(*args)
    else:
        # If the user passed in KEYWORD arguments to specify which
        # columns of the specified table they want to pull data for.
        formated_to_throw_in = "SELECT {} FROM {}".format(
            "{}, " * (len(kwargs) - 1) + "{}", table_name)
        query_str = formated_to_throw_in.format(*list(kwargs.values()))

    # Now, make the query!
    cursor.execute(query_str)
    rows = cursor.fetchall()

    # Perform any neccessary cleanup.
    if as_df:
        # if you would like for the query result to be placed into a 
        # Pandas DataFrame instead of recieving a list of tuples.
        try:
            # Throw the list of row tuples into the `tuples_to_df` 
            # function below. Since there are assertions in the code for 
            # that function, it is possible that it will raise an error.
            my_df = tuples_to_df(table_name, rows, cursor)
            to_return = my_df
        except AssertionError:
        	# Note that the AssertionError is coming from 
            print("""The list of tuples that was passed into the  
            	function did NOT meet the requirement for conversion to 
            	a DataFrame. \n\nAs a result, the object returned by  
            	this function is the list of row tuples.""")
            # raise
            # uncomment line above in the event that you would like see 
            # where exactly the error is occuring.
            to_return = rows
        # If such an error gets thrown, we know that, for some reason, something
        # is up with the list of tuples that we have gotten from our query.
    else:
        # if you just want to work with the rows of tuples.
        to_return = rows

    return to_return


def tuples_to_df(table_name: str, rows_tuples_list: list, cursor):
    """
    Purpose
    -------
    The purpose of this function is to take the list of tuples result
    from a Psycopg SQL query and convert it to the cleaner and easier to
    work with format that is a Pandas DataFrame.
    Pandas

    Parameters
    ----------
    table_name : str
      This string is what allows the user to specify which table they want
      to obtain data from. Note that this table must exist in the database
      for the query to be successful.
    rows_tuple_list : list
      This argument expect a list of tuples which was returned by the 
      `fetchall()` method of the Psycopg cursor object after the query was
      performed. This contains the data that will be placed into a Pandas
      DataFrame.
    cursor : Psycopg cursor object
      This object represents the connection that was previously made to
      a desired database. This is what will be used to execute the query
      to obtain data from the specified table. See connect_to_db function
      documentation for more information about this type of object.

    Returns
    -------
    to_return : Pandas DataFrame
      This represents the successful attempt to convert the list of a
      tuples object that we got from Psycopg into a more useful format.

    References
    ----------
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
    2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html
    """
    # We will use the exact columns name as they appear in the database
    # table in the DataFrame (unless the column name contains spaces in
    # which case we will replace those spaces with underscores).
    raw_names_of_colums = cursor.description
    names_of_colums = [col_name.name for col_name in raw_names_of_colums]

    # Now get the list of tuples into a dictionary
    # First, double check that each tuple does in fact have some value (even if it is NULL)
    # for each column. We do this by first checking that the first tuple has a value for each
    # column and then that each tuple is the same length as the first one. This will help
    # guarntee that nothing strange happens as we construct
    length_of_first_tup = len(rows_tuples_list[0])
    assert len(names_of_colums) == length_of_first_tup
    assert len(rows_tuples_list) == len(
        [row_tup for row_tup in rows_tuples_list if len(row_tup) == length_of_first_tup])

    # Now create a dictionary for each tuple where the keys are the column
    # name values defined above.
    rows_dicts_list = [dict(zip(names_of_colums, row_tup))
                       for row_tup in rows_tuples_list]
    # since we have ensured that the length of each tuple is the same as the number of
    # columns in the table, we will NOT lose any values when we use the zip
    # function.

    # Now we are ready to create the DataFrame
    rows_df = pd.DataFrame(rows_dicts_list).replace(
        to_replace="N/A", value=np.NaN)
    num_of_original_rows = rows_df.shape[0]
    num_of_original_columns = rows_df.shape[1]

    assert len(rows_tuples_list) == num_of_original_rows
    assert len(names_of_colums) == num_of_original_columns
    # More unit tests ;).

    try:
        # If we want to be able to perform the join below, we need to make sure that the
        # labels we get from the Database are cleaned.
        special_tables_checks = ["&" in rows_df.label[0],
                                 "-" in rows_df.label[0],
                                 "." in rows_df.label[0]]
        if np.any(special_tables_checks):
        	# if the label to be cleaned contains any annoying 
        	# characters that will mess up the way in which we are 
        	# cleaning it.
            label_series_0 = rows_df["label"].str.replace(r"\\\ ", " ").str.replace(r"\\", "")
        else:
            label_series_0 = rows_df["label"].str.replace(r"\\\ ", " ")
        
        # Clean values that will make a join with the taxonomy dataframe
        # possible.
        label_series_1 = label_series_0.replace("Childrens Literature", 
                                                "Children's Literature")
        label_series_2 = label_series_1.replace("Womens Fashion", 
                                                "Women's Fashion")
        label_series_3 = label_series_2.replace("Childrens Clothing", 
                                                "Children's Clothing")
        label_series_4 = label_series_3.replace("Mens Accessories", 
                                                "Men's Accessories")
        label_series_5 = label_series_4.replace("Mens Jewelry and Watches", 
                                                "Men's Jewelry and Watches")
        label_series_6 = label_series_5.replace("Mens Business Wear", 
                                                "Men's Business Wear")
        label_series_7 = label_series_6.replace("Mens Casual Wear", 
                                                "Men's Casual Wear")
        label_series_8 = label_series_7.replace("Mens Formal Wear", 
                                                "Men's Formal Wear")
        label_series_9 = label_series_8.replace("Mens Outerwear", 
                                                "Men's Outerwear")
        label_series_10 = label_series_9.replace("Mens Sportswear", 
                                                 "Men's Sportswear")
        label_series_11 = label_series_10.replace("Mens Underwear and Sleepwear", 
                                                  "Men's Underwear and Sleepwear")
        label_series_12 = label_series_11.replace("Mens Shoes and Footwear", 
                                                  "Men's Shoes and Footwear")
        label_series_13 = label_series_12.replace("Childrens TV", 
                                                  "Children's TV")
        label_series_14 = label_series_13.replace("Childrens Health", 
                                                  "Children's Health")
        label_series_15 = label_series_14.replace("Mens Health", 
                                                  "Men's Health")
        label_series_16 = label_series_15.replace("Womens Health", 
                                                  "Women's Health")
        label_series_17 = label_series_16.replace("Childrens Music", 
                                                  "Children's Music")
        label_series_18 = label_series_17.replace("Childrens Games and Toys", 
                                                  "Children's Games and Toys")
        label_series_19 = label_series_18.replace("Womens Accessories", 
                                                  "Women's Accessories")
        label_series_20 = label_series_19.replace("Womens Glasses", 
                                                  "Women's Glasses")
        label_series_21 = label_series_20.replace("Womens Handbags and Wallets", 
                                                  "Women's Handbags and Wallets")
        label_series_22 = label_series_21.replace("Womens Hats and Scarves", 
                                                  "Women's Hats and Scarves")
        label_series_23 = label_series_22.replace("Womens Jewelry and Watches", 
                                                  "Women's Jewelry and Watches")
        label_series_24 = label_series_23.replace("Womens Clothing", 
                                                  "Women's Clothing")
        label_series_25 = label_series_24.replace("Womens Business Wear", 
                                                  "Women's Business Wear")
        label_series_26 = label_series_25.replace("Womens Casual Wear", 
                                                  "Women's Casual Wear")
        label_series_27 = label_series_26.replace("Womens Formal Wear", 
                                                  "Women's Formal Wear")
        label_series_28 = label_series_27.replace("Womens Intimates and Sleepwear", 
                                                  "Women's Intimates and Sleepwear")
        label_series_29 = label_series_28.replace("Womens Outerwear", 
                                                  "Women's Outerwear")
        label_series_30 = label_series_29.replace("Womens Sportswear", 
                                                  "Women's Sportswear")
        label_series_31 = label_series_30.replace("Womens Shoes and Footwear", 
                                                  "Women's Shoes and Footwear")
        label_series_32 = label_series_31.replace("Mens Fashion", 
                                                  "Men's Fashion")
        label_series_33 = label_series_32.replace("Mens Clothing", 
                                                  "Men's Clothing")

        rows_df["cleaned_label"] = label_series_33
    except (AttributeError, KeyError):
        # If the original SQL result that the user wishes to convert 
        # into a DataFrame does NOT have the column `label`. In which 
        # case there is no way to  create the "cleaned_label" column 
        # done above. This is taken care of in the join_with_taxonomy
        # function defined below. 
        pass

    to_return = rows_df

    return to_return


def load_in_taxonomy(
        rel_path_to_taxonomy_file="../../data/raw/annotations_tracking_Taxonomy.csv",
        include_class_compilation=True):
    """
    Purpose
    -------
    The purpose of this function is to load in and then clean up (if 
    desired) the CSV that lists out the taxonomy that describes the 
    classification scheme we will be using throughout this project.

    Parameters
    ----------
    rel_path_to_taxonomy_file : str
      The default value is set to where the CSV file that we will be 
      loading live in relation to where this script lives. This of course
      can be over-written to the set-up that the user prefers.
    include_class_compilation : Boolean
      The default value is set to True. When True, this function will
      call the `class_sequence_compiler` function defined below in order
      to compile all of the tiers into one list of strings. This function
      will not be called if this argument's value is set to False.

    Returns
    -------
    to_return : Pandas DataFrame
      This object contains all of the data that was loaded in (and
      possibly also cleaned up) from the taxonomy CSV.

    References
    ----------
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    2. See the 1_Taxonomy_Exploration (root/notebooks/1_Taxonomy_Exploration.ipynb)
       notebook for why the cleaning steps in this script were taken.
    3. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
    4. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
    """
    # First things, first load in the csv file into a Pandas DataFrame.
    path_to_taxonomy_file = os.path.join(path_of_script, rel_path_to_taxonomy_file)

    raw_taxonomy_df = pd.read_csv(
        filepath_or_buffer=path_to_taxonomy_file,
        usecols=[
            "ID",
            "Parent",
            "Name",
            "Tier1",
            "Tier2",
            "Tier3",
            "Tier4"])

    # Now clean it so that we can actually use it below.
    taxonomy_df = raw_taxonomy_df.drop(index=698).reset_index(drop=True)
    	# See the notebook `1_Taxonomy_Exploration.ipynb` that is in the
    	# notebooks directory to see why this is being done.

    if include_class_compilation:
        # if the user wishes to add a new column that contains a list of strings that specify all of
        # the tiers that this instance belongs to.
        tier_lists_series = taxonomy_df.apply(class_sequence_compiler, axis=1)
        taxonomy_df["Tiers_list"] = tier_lists_series

        checked_names_series = taxonomy_df.apply(name_checker, axis=1)
        assert checked_names_series.sum() == taxonomy_df.shape[0]
        # If we pass this test, then we know that the `Name` column in 
        # our DataFrame is reliable. NOTE that we can only perform this 
        # test if we have compiled all of the (non-null) tier strings in 
        # a list; this is why this test is in this if-block.

    to_return = taxonomy_df

    return to_return


def name_checker(row):
    """
    Purpose
    -------
    The purpose of this function is to check whether or not the last 
    element of the list that exists in the `Tiers_list` column is 
    identical to the value that is in the `Name` column. NOTE THAT THIS 
    FUNCTION IS  WRITTEN TO BE USED WITH THE `apply()` METHOD OF A 
    PANDAS DATAFRAME. You have to make sure that the axis keyword of 
    that method is set to 1 since this function accepts a DataFrame Row.

    Parameters
    ----------
    row : Pandas DataFrame
      This is a row of a Pandas DataFrame which we will use to perform
      all of our operations. Again, remember to set the axis keyword of
      `apply()` to 1.

    Returns
    -------
    to_return : Boolean or np.NaN
      The returned object is either a Boolean or numpy `NaN` object.
      Whichis dependent on whether or not the `Tiers_list` column exists
      in the row that is being use; if it does not, then the result will
      be `NaN` because there is no way to make the comparisions done in
      the function. It it does, then the comparisions can be done and so 
      `True` will be returned if the elements are identical and False 
      otherwise.

    References
    ----------
    1. https://www.pythoncentral.io/one-line-if-statement-in-python-ternary-conditional-operator/
    """
    # Get the string that details the last Tier value and the name.
    did_it_work = True
    try:
        last_tier_str = row.Tiers_list[-1]
    except BaseException:
        print("The `Tiers_list` does NOT yet exist. Please ensure that the DataFrame has been used to call the `class_sequence_compiler` function.")
        did_it_work = False

    name_str = row.Name

    # Now compare the two (if you are able).
    to_return = last_tier_str == name_str if did_it_work else np.NaN
    return to_return


def class_sequence_compiler(row):
    """
    Purpose
    -------
    The purpose of this function is to take all of the Tier labels that
    exist in each row of a DataFrame and compile them into a list; the 
    length of this list is dependent on how many non-null values exist
    in these Tier columns. NOTE THAT THIS  FUNCTION IS  WRITTEN TO BE 
    USED WITH THE `apply()` METHOD OF A  PANDAS DATAFRAME. You have to 
    make sure that the axis keyword of that method is set to 1 since 
    this function accepts a DataFrame Row.

    Parameters
    ----------
    row : Pandas DataFrame row
      This is a row of a Pandas DataFrame which we will use to perform
      all of our operations. Again, remember to set the axis keyword of 
      `apply()` to 1.

    Returns
    -------
    to_return : list
      This list is a compilation of all of the non-null Tier labels that
      make up the class sequence for this particular row.
    """
    to_return = []
    # Compile all of the Tier labels.
    tier_1_str, tier_2, tier_3, tier_4 = row.Tier1, row.Tier2, row.Tier3, row.Tier4
    assert isinstance(tier_1_str, str)
    # just to make sure nothing weird is going on as we
    # extract the labels.
    to_return.append(tier_1_str)

    # Now determine how many tiers should be appended.
    tier_2_str, tier_3_str, tier_4_str = str(tier_2), str(tier_3), str(tier_4)
    nan_for_all_conditions = [tier_2_str.lower() == 'nan',
                              tier_3_str.lower() == 'nan',
                              tier_4_str.lower() == 'nan']
    if np.all(nan_for_all_conditions):
        # If we are working with a row that corresponds to a parent 
        # node. We do NOT need to do anything since we have already 
        # appended that tier label to the final Tiers list.
        pass
    elif np.all(nan_for_all_conditions[1::]):
        # if both tier 3 and tier 4 are NaN.
        to_return.append(tier_2_str)
    elif np.all(nan_for_all_conditions[2::]):
        # if only tier 4 has a NaN value.
        to_return.append(tier_2_str)
        to_return.append(tier_3_str)
    else:
        # if we are in one of the rare cases in which all 4 tiers do NOT have
        # NaN values.
        to_return.append(tier_2_str)
        to_return.append(tier_3_str)
        to_return.append(tier_4_str)

    return to_return


def join_with_taxonomy(table_name: str, cursor):
    """
    Purpose
    -------
    The purpose of this function is to match up each row instance that 
    was obtained by a SQL query with the Tier labels that live in the
    taxonomy DataFrame and JSON file. That way, we can know immediately 
    where exactly each article lives in the hierarchy classification 
    scheme.

    Parameters
    ----------
    table_name : str
      This string is what allows the user to specify which table they
      want to obtain data from. Note that this table must exist in the
      database for the query to be successful.
    cursor : Psycopg cursor object
      This object represents the connection that was previously made to 
      a desired database. This is what will be used to execute the query
      to obtain data from the specified table. See connect_to_db function
      documentation for more information about this type of object.

    Returns
    -------
    to_return : Pandas DataFrame
      This object contains all of the data that came from the SQL query
      for the specified table as well as the additional information that
      we can get from the taxonomy DataFrame returned by `load_in_taxonomy()`.

    References
    ----------
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
    3. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
    """
    to_return = None
    # First, get the taxonomy DataFrame.
    taxonomy_df = load_in_taxonomy()

    # Now get the table DataFrame.
    content_df = get_table(
        table_name=table_name,
        cursor=cursor,
        as_df=True)

    # Now let's investigate the schema of the table df to see if we are
    # ready to perform the join
    try:
        # if the obtained table did in fact come with a `label` column.
        content_df.drop(columns=["index", "label"])
    except (KeyError, AttributeError):
        # That is, if the returned table dataframe was not returned with
        # the label column in which case we will have to use the 
        # taxonomy JSON file to figure out what the `label` for these 
        # instances are.
        content_df.drop(columns="index")
            # We still want to drop this column. Every row of all of the
            # tables will have this column, so this will NOT raise an
            # error.
        table_name_to_map = table_name[table_name.index("_") + 1::].replace("_", ".")
        	# if you take a look at the taxonomy JSON file, you will
        	# see why this is neccessary.
        tiers_dict = np.any([ tax.get(table_name_to_map, None) for tax in TABLE_TAXONOMY_MAPPER ])
        	# we have to check the list comprehension because the mapper
        	# is a list of dictionaries.
        
        tiers_list = []
        for i, j in tiers_dict.items():
        	# it is certainly possible that we have matched with more
        	# than one key of the mapper.
            assert type(i) == str
            	# i represents the parent node of the class sequence.
            tiers_list.append(i)
            maybe_dict = j
            while type(maybe_dict) == dict:
            	# the taxonomy is defined so that it constantly points
            	# to dictionaries if there are still more tiers in the
            	# class sequence.
                for new_i, new_j in maybe_dict.items():
                    assert type(new_i) == str
                    tiers_list.append(new_i)
                    maybe_dict = new_j
            assert type(maybe_dict) == str
            tiers_list.append(maybe_dict)
            	# once maybe_dict is a str, we are ready to append it.
            
        label = tiers_list[-1]
        content_df["cleaned_label"] = [label]*content_df.shape[0]

    # Now do the join.
    if table_name == "table_26_3_7_2":
    	# For some reason, there are two different rows in the 
    	# taxonomy DataFrame that have "email" for its value of "Name".
    	taxonomy_df = taxonomy_df[taxonomy_df.Name == "Email"][0:1:]

    joined_df = content_df.merge(right=taxonomy_df,
    	                         how="inner",
    	                         left_on="cleaned_label",
    	                         right_on="Name").drop(columns="Name")
    assert joined_df.shape[0] == content_df.shape[0]
    # We want to make sure that every row (which corresponds to a unique article) gets
    # matched up with `Tier1`, `Tier2`, `Tier3`, and `Tier4` values.

    final_df = joined_df.rename(
        columns={
            "id": "Article_URL",
            "content": "Content",
            "title": "Title",
            "keywords": "Keywords",
            "description": "Description",
            "preprocessed_content": "Preprocessed_Content",
            "cleaned_label": "Label"})

    to_return = final_df

    return to_return


def load_in_full_data(
        rel_path_to_pickles="../../data/interim/table_dataframes"):
	"""
	Purpose
	-------
	The purpose of this function is to provide an easy tool for the user
	to quickly download all of the article data that we have. This handles
	all of the 

	Parameters
	----------
	rel_path_to_pickles : str
    This string represents the path to the directory where the pickled
    DataFrames that we wish we to load in are stroed RELATIVE TO THIS
    SCRIPT; not relative to whatever directory you are in when using this
    function.

	Returns
	-------
  to_return : Pandas DataFrame
    This function returns the result of loading in the pickled DataFrame
    and doing some minor clean-up to it.

	References
	----------
	1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
	2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
	"""
	to_return = None
	# First, get everything about the paths straightened out.
	script_path = PATH_OF_SCRIPT
	full_path_to_pickles = os.path.join(script_path, 
		                                rel_path_to_pickles)

	# Now, compile the names of all of these files and do some 
	# validation checks
	file_names = os.listdir(full_path_to_pickles)
	assert len([file for file in file_names if file[-4::] == ".pkl"]) == len(file_names)

	# Load in the files and put them all in one DataFrame.
	loaded_in_dfs_list = [ pd.read_pickle("{}/{}".format(full_path_to_pickles, file)) for file in file_names ]
	full_df = pd.concat(
		objs=loaded_in_dfs_list, 
		ignore_index=True).drop(columns=["index", "label", "level_0"]
		)

	# Clean up the obtained data.
	full_df["Preprocessed_Content"] = full_df.Preprocessed_Content.replace(
		"", np.NaN)
	final_full_df = full_df.dropna(subset=["Preprocessed_Content"])

	to_return = final_full_df
	return to_return





