import pandas as pd
from pyparsing import Any


def value_replacement(transform: callable, value: Any, default: Any = 0):
    """
    The function `value_replacement` takes a transformation function, a value, and an optional default
    value, and returns the result of applying the transformation function to the value or the default
    value if an exception occurs.

    :param transform: The `transform` parameter is a callable function that will be used to transform
    the `value` parameter
    :type transform: callable
    :param value: The `value` parameter is the input value that you want to transform using the provided
    `transform` function
    :type value: Any
    :param default: The `default` parameter in the `value_replacement` function is a keyword argument
    with a default value of 0. This means that if the `default` parameter is not provided when calling
    the function, it will default to 0. If an exception occurs during the transformation of the `value,
    defaults to 0
    :type default: Any (optional)
    :return: The function `value_replacement` takes three parameters: `transform`, `value`, and
    `default`. It tries to apply the `transform` function to the `value` parameter and return the
    result. If an exception occurs during the transformation, it returns the `default` value (which is
    set to 0 by default).
    """
    try:
        return transform(value)
    except Exception:
        return default


def prepare_binary_fields(df: pd.DataFrame, fields: list[str]):
    """
    The function `prepare_binary_fields` processes specified fields in a DataFrame to convert them into
    binary True/False values based on specific string mappings.

    :param df: A pandas DataFrame containing the data you want to process
    :type df: pd.DataFrame
    :param fields: The `fields` parameter is a list of strings representing the column names in a pandas
    DataFrame that you want to process in the `prepare_binary_fields` function
    :type fields: list[str]
    :return: The function `prepare_binary_fields` is returning the DataFrame `df` after processing the
    specified fields to convert them into binary values (True/False).
    """
    for field in fields:
        df[field].fillna(False, inplace=True)
        df.loc[df[field].isin(["si", "Si", "SI", "M"]), field] = True
        df.loc[df[field].isin(["no", "No", "NO", "Ninguno", "Ninguna", "0", "F"]), field] = False
        df.loc[~df[field].isin([True, False]), field] = True
    return df


def prepare_numeric_fields(df: pd.DataFrame, fields: list[str]):
    """
    The function `prepare_numeric_fields` processes numeric fields in a DataFrame by replacing commas
    with periods, converting certain non-numeric values to 0, filling NaN values with -1, and applying a
    value replacement function.

    :param df: A pandas DataFrame containing the data you want to process
    :type df: pd.DataFrame
    :param fields: The `fields` parameter in the `prepare_numeric_fields` function is a list of column
    names in a pandas DataFrame that you want to prepare as numeric fields. The function performs
    several operations on these fields to ensure they are in a suitable format for numerical
    calculations
    :type fields: list[str]
    :return: The function `prepare_numeric_fields` is returning the DataFrame `df` after performing
    operations to prepare the specified numeric fields as per the provided code snippet.
    """
    for field in fields:
        df[field] = df[field].replace(",", ".")
        df.loc[df[field].isin(["Ninguno", "Ninguna", "No"]), field] = 0
        df[field].fillna(-1, inplace=True)
        df[field] = df[field].apply(lambda x: str(x).replace(",", "."))
        df[field] = df[field].apply(lambda x: value_replacement(float, x, -1))
    return df


def prepare_time_fields(df: pd.DataFrame, fields: list[str]):
    """
    The function `prepare_time_fields` fills missing values in specified time fields with 0 and converts
    them to datetime objects.

    :param df: A pandas DataFrame containing the data you want to process
    :type df: pd.DataFrame
    :param fields: The `fields` parameter is a list of column names in a pandas DataFrame (`df`) that
    contain time-related data that needs to be prepared
    :type fields: list[str]
    :return: The function `prepare_time_fields` is returning the DataFrame `df` after filling NaN values
    with 0 and converting the specified fields to datetime format using the `value_replacement`
    function.
    """
    for field in fields:
        df[field].fillna(0, inplace=True)
        df[field] = df[field].apply(lambda x: value_replacement(pd.to_datetime, x, 0))
    return df


def prepare_categoric_fields(df: pd.DataFrame, fields: list[str]):
    """
    The `prepare_categoric_fields` function fills missing values in specified categorical fields with
    "N/A", while the `remove_useless_columns` function removes columns with only one unique value from a
    DataFrame.

    :param df: A pandas DataFrame containing the data that you want to work with. It could be your main
    dataset with multiple columns and rows
    :type df: pd.DataFrame
    :param fields: The `fields` parameter in the `prepare_categoric_fields` function is a list of column
    names that you want to fill missing values with "N/A"
    :type fields: list[str]
    :return: The `remove_useless_columns` function returns a new DataFrame containing only the columns
    that have more than one unique value in the original DataFrame `df`.
    """
    for field in fields:
        df[field].fillna("N/A", inplace=True)
    return df


def remove_useless_columns(df: pd.DataFrame, fields: list[str] = []):
    """
    The function removes columns from a DataFrame that have only one unique value.

    :param df: A pandas DataFrame containing the data from which you want to remove useless columns
    :type df: pd.DataFrame
    :param fields: The `fields` parameter in the `remove_useless_columns` function is a list of column
    names that you want to consider for removal. If `fields` is provided (not empty), only those columns
    will be checked for uniqueness. If `fields` is empty, all columns in the DataFrame `
    :type fields: list[str]
    :return: The `remove_useless_columns` function returns a new DataFrame containing only the columns
    that have more than one unique value in the original DataFrame `df`.
    """
    useful_fields = []
    columns = fields if len(fields) > 0 else df.columns
    for field in columns:
        if df[field].unique().shape[0] > 1:
            useful_fields += [field]
    return df[useful_fields]
