import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt

from app.core.dataset.data_preparation import prepare_numeric_fields
from app.model.abstract_wrapper import AbstractWrapper
from app.model.code_translator import CodeTranslator
from app.model.errors import AIXSequencesError
from app.model.model_wrapper import ModelWrapper


# The `AIXWrapper` class provides methods for calculating SHAP values, generating various types of
# analysis plots, and saving the analysis results for a given model.
class AIXWrapper(AbstractWrapper):
    model: ModelWrapper
    values: Optional[Any] = None
    values_filename: Optional[str] = "shap-values.pkl"
    translator_file: Optional[str] = "codes.csv"
    folder: Optional[str] = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.try_load_shap_values()

    def try_load_shap_values(self):
        values_path = os.path.join(self.folder, self.values_filename)

        try:
            self.values = self.load(values_path)
        except FileNotFoundError:
            return

    def calculate(self, start_idx: int = None, end_idx: int = None, force: bool = False, save_values: bool = False):
        if not force and self.values is not None:
            return

        explainer = shap.Explainer(self.model.predict, shap.maskers.Text(r"\W"))

        if self.model.sequences is None or self.model.sequences.shape[0] == 0:
            raise AIXSequencesError("empty sequences")

        seqs = self.model.sequences if start_idx is None or end_idx is None else self.model.sequences[start_idx:end_idx]
        self.values = explainer(seqs, silent=True)

        if save_values:
            values_path = os.path.join(self.folder, self.values_filename)
            self.save(self.values, values_path)

    def generate_analysis(self, start_idx: int = None, end_idx: int = None, max_display: int = 20, save: bool = False):
        """
        The function `generate_analysis` generates and displays a bar analysis within a specified range,
        with an option to save the analysis as an image.

        :param start_idx: The `start_idx` parameter specifies the starting index for generating the
        analysis. It is an integer value indicating the index from which the analysis should begin
        :type start_idx: int
        :param end_idx: The `end_idx` parameter in the `generate_analysis` method is used to specify the
        ending index for the analysis. It indicates the index at which the analysis should stop
        processing the data
        :type end_idx: int
        :param max_display: The `max_display` parameter specifies the maximum number of items to display
        in the analysis. It is used to limit the number of items shown in the analysis output to prevent
        overwhelming the user with too much information at once
        :type max_display: int
        :param save: The `save` parameter in the `generate_analysis` method is a boolean flag that
        indicates whether the generated analysis should be saved to a file or not. If `save` is set to
        `True`, the analysis will be saved as an image file named "{model_name}-bars.png". If `,
        defaults to False
        :type save: bool (optional)
        """
        if save:
            img_path = os.path.join(self.folder, f"{self.model.name}-bars.png")
            self.show_bar_analysis(start_idx, end_idx, max_display=max_display, to_file=img_path)
        else:
            self.show_bar_analysis(start_idx, end_idx, max_display=max_display)

    def get_features(self):
        """
        The function `get_features` extracts features from a dataset, calculates statistics like mean and
        standard deviation for each feature, and returns a dictionary containing this information.
        :return: A dictionary containing information about the features in the dataset. Each feature
        includes its title, values, length, mean, and standard deviation.
        """
        features = {}
        translator = CodeTranslator(self.translator_file)
        for i, row in enumerate(self.values.feature_names):
            for j, feature_name in enumerate(row):
                if feature_name not in features:
                    features[feature_name] = {}
                    features[feature_name]["title"] = translator(feature_name)
                    features[feature_name]["values"] = []

                features[feature_name]["values"] += [self.values.values[i][j]]

        for feat in features:
            features[feat]["len"] = len(features[feat]["values"])
            features[feat]["mean"] = np.mean(features[feat]["values"])
            features[feat]["std"] = np.std(features[feat]["values"])

        return features

    def get_features_dataframe(self):
        """
        The function `get_features_dataframe` creates a pandas DataFrame from features, prepares numeric
        fields, and adds additional columns.
        :return: A pandas DataFrame containing features extracted from the data, with additional columns
        for mean absolute value and data types converted to float32 and Int32.
        """
        features = self.get_features()

        df = pd.DataFrame.from_dict(features).transpose()
        df = prepare_numeric_fields(df, ["mean", "std"])
        df[["mean", "std"]] = df[["mean", "std"]].astype(pd.Float32Dtype)
        df["len"] = df["len"].astype("Int32")

        df["|mean|"] = df["mean"].abs()

        return df

    def generate_analysis_v2(self, max_display: int = 20, title="", ylabel="", save: bool = False):
        df = self.get_features_dataframe()
        df = df.sort_values(by=["|mean|"], ascending=False)
        df = df[["title", "mean"]].iloc[:max_display]
        palette = [
            shap.plots.colors.red_rgb if val > 0 else shap.plots.colors.blue_rgb for val in df["mean"].to_numpy()
        ]
        plt.figure(figsize=(5, max_display * 0.3))
        ax = sns.barplot(
            data=df,
            x="mean",
            y="title",
            orient="y",
            native_scale=True,
            palette=palette,
        )
        for container in ax.containers:
            ax.bar_label(container)
        ax.set(title=title, xlabel="SHAP value", ylabel=ylabel, xticks=[-1, -0.5, 0, 0.5, 1])

        if save:
            img_path = os.path.join(self.folder, f"{self.model.name}-bars.png")
            plt.savefig(img_path, bbox_inches="tight")

        plt.show()

    def show_violin(
        self, max_display: int = 20, title="", ylabel="", show_n: bool = False, save: bool = False, asc: bool = False
    ):
        """
        The `show_violin` function generates a violin plot to visualize feature distributions with
        optional customization for display and saving the plot.

        :param max_display: The `max_display` parameter in the `show_violin` method specifies the
        maximum number of features to display in the violin plot. Only the top `max_display` features
        based on their absolute mean values will be shown in the plot, defaults to 20
        :type max_display: int (optional)
        :param title: The `title` parameter in the `show_violin` method is used to specify the title of
        the violin plot that will be displayed. It allows you to provide a descriptive title for the
        plot to help users understand the context or purpose of the visualization
        :param ylabel: The `ylabel` parameter in the `show_violin` method is used to set the label for
        the y-axis of the violin plot that will be displayed. It is typically used to provide a
        description or name for the variable being visualized on the y-axis. You can customize this
        label based
        :param show_n: The `show_n` parameter in the `show_violin` method is a boolean flag that
        determines whether additional information about the number of data points in each violin plot
        should be displayed. If `show_n` is set to `True`, the number of data points in each violin plot
        will be, defaults to False
        :type show_n: bool (optional)
        :param save: The `save` parameter in the `show_violin` method is a boolean flag that determines
        whether the generated violin plot should be saved as an image file or not. If `save` is set to
        `True`, the plot will be saved as a PNG image file in the specified location, defaults to False
        :type save: bool (optional)
        :param asc: The `asc` parameter in the `show_violin` method is a boolean flag that determines
        whether the data should be sorted in ascending order or not. When `asc` is set to `True`, the
        data will be sorted in ascending order based on the absolute mean values of the features,
        defaults to False
        :type asc: bool (optional)
        """
        features = self.get_features()
        data = [
            [key, features[key]["values"], features[key]["title"], np.abs(features[key]["mean"])] for key in features
        ]
        data.sort(key=lambda x: x[3], reverse=not asc)
        data = data[:max_display]

        keys = [v[0] for v in data]
        dataset = [v[1] for v in data]
        labels = [v[2] for v in data]

        labels.reverse()
        dataset.reverse()
        keys.reverse()

        plt.figure(figsize=(8, len(labels) * 0.5))
        plt.yticks(labels=labels, ticks=np.arange(1, len(dataset) + 1), fontsize=12)
        # plt.xticks(ticks=np.arange(-1, 1, 0.5))
        ax = plt.violinplot(dataset=dataset, showmeans=True, vert=False, widths=0.8)
        for i, item in enumerate(ax["bodies"]):
            color = shap.plots.colors.blue_rgb if features[keys[i]]["mean"] < 0 else shap.plots.colors.red_rgb
            item.set(facecolor=color, edgecolor="black", linewidth=0.1)
            plt.axhline(y=i + 1, color="grey", linewidth=0.5, ls="dashed")

            label_attrs = [f"{features[keys[i]]['mean']:<.3f}"]
            if show_n:
                label_attrs += [f"n: {len(dataset[i])}"]

            plt.text(features[keys[i]]["mean"] + 0.02, i + 1 + 0.15, s=", ".join(label_attrs), fontsize=10)

        plt.title(title, loc="center")
        plt.ylabel(ylabel, fontsize=16)

        plt.axvline(x=0, color="black", linewidth=0.5, ls="dashed")

        if save:
            img_path = os.path.join(self.folder, f"{self.model.name}-violin.png")
            plt.savefig(img_path, bbox_inches="tight")

        plt.show()

    def generate_local_analysis(self, idx: int, max_display: int, save: bool = False):
        """
        This Python function generates local analysis based on the provided index and parameters, with
        an option to save the analysis results to files.

        :param idx: The `idx` parameter in the `generate_local_analysis` method is used to specify the
        index of the data point for which the local analysis is being generated
        :type idx: int
        :param max_display: The `max_display` parameter in the `generate_local_analysis` method
        specifies the maximum number of items to display in the analysis. It is used in various
        visualization methods such as `show_bar_analysis`, `show_waterfall_analysis`, and
        `show_bar_analysis_by_idx` to limit the number of items
        :type max_display: int
        :param save: The `save` parameter in the `generate_local_analysis` method is a boolean flag that
        determines whether to save the generated analysis results to files or not. If `save` is set to
        `True`, the analysis results will be saved to files with specific names based on the `base_name`
        and, defaults to False
        :type save: bool (optional)
        """

        if save:
            base_name = os.path.join(self.folder, f"{self.model.name}-local-{idx}")

            self.show_text_analysis(idx, idx + 1, f"{base_name}-text.html")
            self.show_bar_analysis(idx, idx + 1, to_file=f"{base_name}-bars.png")
            self.show_waterfall_analysis(idx, max_display=max_display, to_file=f"{base_name}-waterfall.png")
            self.show_bar_analysis_by_idx(idx, max_display=max_display, to_file=f"{base_name}-bar.png")

        else:
            self.show_text_analysis(idx, idx + 1)
            self.show_bar_analysis(idx, idx + 1, max_display=max_display)
            self.show_waterfall_analysis(idx, max_display=max_display)
            self.show_bar_analysis_by_idx(idx, max_display=max_display)

    def show_text_analysis(self, start_idx: int, end_idx: int, to_file: str = None):
        """
        The function `show_text_analysis` generates SHAP text plots for a specified range of indices and
        optionally saves the output to an HTML file.

        :param start_idx: The `start_idx` parameter specifies the starting index of the text analysis
        you want to display or save to a file
        :type start_idx: int
        :param end_idx: The `end_idx` parameter in the `show_text_analysis` method is used to specify
        the end index for selecting a range of values from the `self.values` attribute. This range
        starts from the `start_idx` (inclusive) and ends at `end_idx` (exclusive)
        :type end_idx: int
        :param to_file: The `to_file` parameter in the `show_text_analysis` method is an optional
        parameter that specifies the file path where the HTML output will be saved. If a value is
        provided for `to_file`, the method will generate an HTML representation of the text analysis and
        save it to the specified file
        :type to_file: str
        """
        if to_file:
            html = shap.plots.text(shap_values=self.values[start_idx:end_idx], separator=" ", display=False)
            with open(to_file, "w") as f:
                f.write(f"<html><body>{html}</body></html>")
        else:
            shap.plots.text(shap_values=self.values[start_idx:end_idx], separator=" ")

    def show_waterfall_analysis(
        self, idx: int, max_display: int = 20, figsize: tuple[int, int] = (1, 2), to_file: str = None
    ):
        """
        This function generates a waterfall plot for a specific index and optionally saves it to a file.

        :param idx: The `idx` parameter is an integer that represents the index of the data point for
        which you want to show the waterfall analysis
        :type idx: int
        :param max_display: The `max_display` parameter in the `show_waterfall_analysis` function
        specifies the maximum number of features to display in the waterfall plot. Only the top
        `max_display` features will be shown in the plot, making it easier to visualize the most
        important contributors to the prediction, defaults to 20
        :type max_display: int (optional)
        :param figsize: The `figsize` parameter in the `show_waterfall_analysis` function is used to
        specify the dimensions of the figure (plot) that will be created by the function. It is a tuple
        containing two integers representing the width and height of the figure in inches
        :type figsize: tuple[int, int]
        :param to_file: The `to_file` parameter in the `show_waterfall_analysis` function is used to
        specify the file path where the generated plot will be saved as an image file. If a value is
        provided for `to_file`, the plot will be saved to that file path. If `to_file` is
        :type to_file: str
        """
        plt.figure(figsize=figsize)
        if to_file:
            shap.plots.waterfall(self.values[idx], max_display=max_display, show=False)
            plt.savefig(to_file, bbox_inches="tight")
        else:
            shap.plots.waterfall(self.values[idx], max_display=max_display, show=True)

    def show_bar_analysis(
        self,
        start_idx: int = None,
        end_idx: int = None,
        max_display: int = 20,
        clustering_cutoff: int = 2,
        figsize: tuple[int, int] = (1, 2),
        to_file: str = None,
    ):
        """
        This function generates a bar plot analysis for a specified range of values with optional
        customization options and the ability to save the plot to a file.

        :param start_idx: The `start_idx` parameter specifies the starting index for the data range you
        want to analyze in the `show_bar_analysis` function. It indicates the index from which the
        analysis will begin
        :type start_idx: int
        :param end_idx: The `end_idx` parameter specifies the ending index for the range of values to be
        analyzed in the `show_bar_analysis` function. It indicates the index up to which the analysis
        will be performed on the values array starting from the `start_idx`
        :type end_idx: int
        :param max_display: The `max_display` parameter in the `show_bar_analysis` function determines
        the maximum number of features to display in the bar plot. If there are more features than the
        specified `max_display`, only the top `max_display` features will be shown in the plot, defaults
        to 20
        :type max_display: int (optional)
        :param clustering_cutoff: The `clustering_cutoff` parameter in the `show_bar_analysis` function
        determines the threshold for clustering similar features together in the SHAP bar plot. Features
        with a similarity score above the `clustering_cutoff` value will be grouped together in the
        plot. Lower values of `clustering_cutoff` will, defaults to 2
        :type clustering_cutoff: int (optional)
        :param figsize: The `figsize` parameter in the `show_bar_analysis` function is used to specify
        the width and height of the figure in inches for the bar plot visualization. It is a tuple
        containing two integers representing the width and height of the figure
        :type figsize: tuple[int, int]
        :param to_file: The `to_file` parameter in the `show_bar_analysis` function is used to specify
        the file path where the generated plot will be saved as an image file. If a value is provided
        for `to_file`, the plot will be saved to that file path. If `to_file` is set
        :type to_file: str
        """
        plt.figure(figsize=figsize)
        values = self.values if start_idx is None or end_idx is None else self.values[start_idx:end_idx]
        if to_file:
            shap.plots.bar(
                values,
                max_display=max_display,
                clustering_cutoff=clustering_cutoff,
                clustering=False,
                order=shap.Explanation.identity,
                show=False,
            )
            plt.savefig(to_file, bbox_inches="tight")
        else:
            shap.plots.bar(
                values,
                max_display=max_display,
                clustering_cutoff=clustering_cutoff,
                clustering=False,
                order=shap.Explanation.identity,
                show=True,
            )

    def show_bar_analysis_by_idx(
        self,
        idx: int,
        max_display: int = 20,
        clustering=None,
        clustering_cutoff: int = 0,
        figsize: tuple[int, int] = (10, 20),
        to_file: str = None,
    ):
        """
        This function generates a bar plot analysis for a specific index with optional clustering and
        saving the plot to a file.

        :param idx: The `idx` parameter in the `show_bar_analysis_by_idx` function is used to specify
        the index of the data point for which you want to display the bar analysis
        :type idx: int
        :param max_display: The `max_display` parameter specifies the maximum number of features to
        display in the bar plot. Only the top `max_display` features will be shown in the plot, defaults
        to 20
        :type max_display: int (optional)
        :param clustering: The `clustering` parameter in the `show_bar_analysis_by_idx` function is used
        for specifying the clustering method to be used in the SHAP bar plot. Clustering is a technique
        used to group similar data points together based on certain criteria. In this context, it can be
        used to group
        :param clustering_cutoff: The `clustering_cutoff` parameter in the `show_bar_analysis_by_idx`
        function is used to specify the cutoff for clustering the features in the SHAP bar plot.
        Features with a similarity greater than the `clustering_cutoff` value will be grouped together
        in the plot. This can help in visual, defaults to 0
        :type clustering_cutoff: int (optional)
        :param figsize: The `figsize` parameter in the `show_bar_analysis_by_idx` function is used to
        specify the dimensions of the figure (plot) that will be created. It expects a tuple of two
        integers representing the width and height of the figure in inches. By default, the `figsize`
        parameter
        :type figsize: tuple[int, int]
        :param to_file: The `to_file` parameter in the `show_bar_analysis_by_idx` function is a string
        parameter that specifies the file path where the generated plot will be saved. If a value is
        provided for `to_file`, the plot will be saved as an image file at the specified location. If
        `to
        :type to_file: str
        """
        plt.figure(figsize=figsize)
        if to_file:
            shap.plots.bar(
                self.values[idx],
                max_display=max_display,
                clustering_cutoff=clustering_cutoff,
                clustering=clustering,
                show=False,
            )
            plt.savefig(to_file, bbox_inches="tight")
        else:
            shap.plots.bar(
                self.values[idx],
                max_display=max_display,
                clustering_cutoff=clustering_cutoff,
                clustering=clustering,
                show=True,
            )
