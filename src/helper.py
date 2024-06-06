import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class Helper:
    @staticmethod
    def print_unique_column_entries(df: pd.DataFrame, threshold: int = 10) -> None:
        """
        Print the unique entries for each column in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame whose columns will be analyzed.
        threshold (int): The maximum number of unique entries to print for each column.
                         If a column has more unique entries than this threshold, only
                         a message indicating the number of unique entries will be printed.

        Returns:
        None
        """
        for column in df.columns:
            unique_entries = df[column].unique()
            dtype_col = df[column].dtype
            if len(unique_entries) > threshold:
                print(f"Column '{column}' of type '{dtype_col}': Has more than {threshold} unique entries")
            else:
                print(f"Column '{column}' of type '{dtype_col}': {unique_entries}")

    @staticmethod
    def print_distinct_values(values, power, threshold):
        """
        Filters and returns distinct values from the input array that are far from each other based on a given threshold.

        This function iterates through the input array `values` and appends each value to the `result` list
        only if it is far (greater than the specified `threshold`) from all previously appended values in the `result` list.

        Parameters:
        values (array-like): A list or array of numerical values to be filtered.
        power (array-like): A list or array of power values corresponding to `values`.
        threshold (float): The minimum difference required between values to be considered distinct.

        Returns:
        list: A list of tuples where each tuple contains a value and its corresponding power value that are distinct based on the specified threshold.

        Example:
        values = [24, 24, 12, 617.6, 628.9, 12, 606.6, 315.9, 217.6, 318.8, 640.7]
        power = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        threshold = 100
        print_distinct_values(values, power, threshold)
        [(24, 1), (617.6, 4), (12, 3), (315.9, 8)]
        """
        result_frequencies = []
        result_power = []
        for k, value in enumerate(values):
            if all(abs(value - existing_value) > threshold for existing_value in result_frequencies):
                result_frequencies.append(np.round(value))
                result_power.append(np.round(power[k], 2))
        return list(zip(result_frequencies, result_power))

    @staticmethod
    def plot_time_series(data, x_col, y_col, hue_col=None, title=None, ylabel=None, xlabel=None, start=None, end=None,
                         scatter=False, figsize=(15, 3)):
        """
        Plots a time series data using seaborn.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The name of the column to use for the x-axis (typically the datetime).
        y_col (str): The name of the column to use for the y-axis (typically the value to plot).
        hue_col (str, optional): The name of the column to use for color coding. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        xlabel (str, optional): The label of the x-axis. Defaults to None.
        ylabel (str, optional): The label of the y-axis. Defaults to None.
        start (int, optional): The starting index for slicing the data. Defaults to None.
        end (int, optional): The ending index for slicing the data. Defaults to None.
        scatter (bool, optional): If True, scatter points will be added to the plot. Defaults to False.
        figsize (tuple, optional): The size of the figure. Defaults to (15, 3).

        Returns:
        None
        """
        plt.figure(figsize=figsize)
        if start is not None and end is not None:
            data = data.iloc[start:end, :]
        sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, markers=True)
        if scatter:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, size=1, legend=False)
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(x_col)
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(y_col)

        if hue_col is not None:
            plt.legend(title=hue_col)
        plt.show()

    @staticmethod
    def fill_nan_from_seasonality(series, window, distribution, threshold_quantile):
        """
        Fill NaN values in a series based on seasonal patterns.

        Parameters:
        series (pd.Series): The time series data.
        window (int): The size of the window to look back for seasonality.
        distribution (object): A distribution object with a `score` method.
        threshold_quantile (float): The threshold for determining an outlier.

        Returns:
        pd.Series: The series with NaN values filled.
        """
        series_copy = series.copy()
        for k in range(0, len(series_copy) - window):
            temp_value = series_copy.iloc[k - window]
            if np.isnan(series_copy.iloc[k + window]) and not np.isnan(temp_value):
                if np.exp(distribution.score(np.array(temp_value).reshape(-1, 1))) > threshold_quantile:
                    #if distribution.score(np.array(temp_value).reshape(-1, 1)) > threshold_quantile:
                    series_copy.iloc[k + window] = series_copy.iloc[k]
        if series_copy.equals(series):
            if np.isnan(series_copy.iloc[0]):
                series_copy.iloc[0] = np.mean(series)
            return series_copy
        else:
            return Helper.fill_nan_from_seasonality(series_copy, window, distribution, threshold_quantile)

    @staticmethod
    def fill_nan_from_distribution(series, distribution):
        """
        Fill NaN values in a series based on a given distribution.

        Parameters:
        series (pd.Series): The time series data.
        distribution (object): A distribution object with a `sample` method.

        Returns:
        pd.Series: The series with NaN values filled.
        """
        import sys
        mean_series = series.mean()
        if np.isnan(mean_series):
            sys.exit("The series contains only NaN values")
        if np.isnan(series.iloc[0]):
            series.iloc[0] = np.median(series)
        for k in range(len(series)):
            if np.isnan(series.iloc[k]):
                series.iloc[k] = distribution.sample()[0][0, 0]
        return series

    @staticmethod
    def calculate_difference_timeseries(
            df, property_column="property_name", property="Heating",
            value_column="temperature", date_column="datetime",
    ):
        """
        Calculate the temperature difference between heating and cooling properties.

        Parameters:
        df (pd.DataFrame): The DataFrame containing columns 'property_name', 'temperature', and 'datetime'.

        Returns:
        pd.DataFrame: A DataFrame with columns for datetime, temperature difference, cooling temperatures, and heating temperatures.
        """
        df = df.reset_index()
        result = []
        timeseries = []
        temp_cooling = []
        temp_heating = []
        for i in range(0, len(df) - 1, 2):
            if df.at[i, property_column] == property:
                difference = df.at[i, value_column] - df.at[i + 1, value_column]
                temp_cooling.append(df.at[i, value_column])
                temp_heating.append(df.at[i + 1, value_column])
            else:
                difference = df.at[i + 1, value_column] - df.at[i, value_column]
                temp_heating.append(df.at[i, value_column])
                temp_cooling.append(df.at[i + 1, value_column])
            result.append(difference)
            timeseries.append(df.at[i, date_column])

        df = pd.DataFrame({
            'datetime': timeseries,
            'temperature_difference': result,
            'cooling_temperature': temp_cooling,
            'heating_temperature': temp_heating
        })

        return df
