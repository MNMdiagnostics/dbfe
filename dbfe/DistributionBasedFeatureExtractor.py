import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from .functions import generate_equal_bins, generate_quantile_bins, generate_clustering_bins, generate_supervised_bins, \
    create_features_from_bins, filter_values_to_range

from .validation import check_n_bins, check_y, check_values, check_cv


class DistributionBasedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, breakpoint_type='quantile', n_bins=4, prefix='dbfe', bw=0.5, resolution=200, log_scale=True, only_peaks=False, cv=None, value_range=None, bins=[], include_counts=True, include_fracs=True, include_total=True):
        """Constructor

        Args:
            breakpoint_type (str, optional): Type of breakpoint to generate. Available options are: equal, quantile, supervised, clustering. Defaults to 'quantile'.
            n_bins (int/float/'auto'/'sd'/'all', optional): Number of features to generate. int explicitly selects given number of features; float selects only the breakpoints corresponding with peaks higher than n_bins * <max_peak_height>; 'sd' selects only the breakpoints corresponding with peaks higher than the standard deviation of the density difference; 'auto' the same as 'sd'; 'all' takes all generated features (available only for supervised method). Defaults to 4.
            prefix (str, optional): Feature name prefix (the underscore after the prefix is added automatically). Defaults to 'cnv_del'.
            bw (float, optional): Bandwidth for kde. Used only in supervised breakpoints. Defaults to 0.5.
            resolution (int, optional): Resolution of kde. Used only in supervised breakpoints. Defaults to 200.
            log_scale (bool, optional): Whether log scale should be used on the x axis to calculate the bins. Used only in supervised breakpoints. Defaults to True.
            only_peaks (bool, optional): Whether the whole spectrum of lengths should be subdivided into features, or should only the actual peaks be selected. Used only in supervised breakpoints. Defaults to False.
            cv (int/object/None, optional): A parameter deciding whether the kde should be averaged over cv folds or if it should be done on the whole dataset. int - specifies the number of folds; object - a cross-validation objects specifying the details of cv; None - analysis done on the whole dataset. Defaults to None.
            value_range (tuple, optional): Defines the minimum and maximum length to be taken into account. None means no filtering, tuple of ints defines the minimum and maximum values, tuple of floats defines the minimum and maximum quantile.
            bins (list): A list manually predefined bins as tuples.
            include_counts (bool): Should the model generate count features. Defaults to True.
            include_fracs (bool): Should the model generate fraction features, i.e., features where the value represents the fraction of lengths in a given bin. Defaults to False.
            include_total (bool): Should the model generate a feature with the total number of lengths. Defaults to False.
        """
        self.breakpoint_type = breakpoint_type
        self.n_bins = n_bins
        self.prefix = prefix
        self.bw = bw
        self.resolution = resolution
        self.log_scale = log_scale
        self.only_peaks = only_peaks
        self.cv = cv
        self.value_range = value_range
        self.bins = bins
        self.include_counts = include_counts
        self.include_fracs = include_fracs
        self.include_total = include_total

    def fit(self, X, y):
        """Creates breakpoints based on the provided lengths and classes

        Args:
            X (Series): A series of lists of lengths
            y (Series): A series of labels

        Raises:
            Exception: Unsupported transformation type.

        Returns:
            self: A fitted extractor ready to transform new datasets according to trained breakpoints
        """
        if not self.bins:
            if self.n_bins == 1 and not self.breakpoint_type == 'supervised' and not self.only_peaks:
                self.bins = [(0, np.inf)]
            else:
                values = self.unstack(X)
                check_values(values)

                if self.value_range is not None:
                    values = filter_values_to_range(values, self.value_range)

                if self.breakpoint_type == 'equal':
                    self.bins = generate_equal_bins(values, self.n_bins, self.log_scale)
                elif self.breakpoint_type == 'quantile':
                    self.bins = generate_quantile_bins(values, self.n_bins)
                elif self.breakpoint_type == 'clustering':
                    self.bins, self.guassian_model = generate_clustering_bins(values, self.n_bins)
                elif self.breakpoint_type == 'supervised':
                    check_y(y)

                    self.cv = check_cv(self.cv)

                    self.n_bins = check_n_bins(self.n_bins)

                    self.all_bins, self.bins, self.density = generate_supervised_bins(values, y, self.n_bins, self.bw, self.resolution, self.log_scale, self.only_peaks, self.cv)
                else:
                    raise ValueError('Unsupported transformation type! Only \'quantile\', \'clustering\', and \'supervised\' are supported.')

        return self

    def transform(self, X):
        """Transforms the input series of lengths into features based on bins. Requires prior running of fit method.

        Args:
            X (Series): A series of lists of lengths to transform into features

        Returns:
            DataFrame: A DataFrame containing new features
        """

        result = pd.DataFrame(index=X.index)

        count_features = create_features_from_bins(X, self.bins, self.prefix).fillna(0)

        if self.breakpoint_type == 'supervised':
            all_features = create_features_from_bins(X, self.all_bins, self.prefix).fillna(0)
        else:
            all_features = count_features

        total_feature = all_features.sum(axis=1)
        total_feature.name = f'total_{self.prefix}'

        if self.include_counts:
            result = result.join(count_features)

        if self.include_total:
            result = result.join(total_feature)

        if self.include_fracs:
            frac_features = count_features.div(total_feature, axis=0)
            frac_features.columns = 'frac_' + frac_features.columns

            result = result.join(frac_features)

        return result

    def fit_transform(self, X, y):
        """First calls fit, then transform, and returns the result

        Args:
            X (Series): A series of lists of lengths
            y (Series): A series of classes

        Returns:
            DataFrame: A DataFrame containing new features
        """
        self.fit(X, y)

        return self.transform(X)

    def plot_data_with_breaks(self, X, y, plot_type='hist', plot_ax=None, subplot=False):
        """Plots lengths with indicated bins

        Args:
            X (Series): A series of lists of lengths to be plotted
            y (Series): Classes to differentiate on the plot
            plot_type (str): Type of plot to show. Available options: hist or kde.
            plot_ax (matplotlib.Axes): If provided, axis to plot on
            subplot (bool): If a plot is subplot
        """

        lengths = self.unstack(X)

        plot_data = lengths.to_frame().join(y, how='inner')
        plot_data.columns = ['Length', 'Class']

        if plot_type == 'hist':
            ax = sns.histplot(data=plot_data, x='Length', hue='Class', log_scale=self.log_scale, common_bins=True, common_norm=False, ax=plot_ax)

            breakpoints = np.unique([item for sublist in self.bins for item in sublist])
            for b in breakpoints:
                ax.axvline(b, color='gray', linestyle='--');
        elif plot_type == 'kde':
            if self.breakpoint_type == 'supervised':
                ax = sns.lineplot(data=self.density, x='Length', y='Density', hue='Class', ci='sd', ax=plot_ax);
                if self.log_scale:
                    ax.set(xscale='log');

                if self.only_peaks:
                    for b in self.bins:
                        if b[1] == np.inf:
                            _, right = plt.xlim()
                        else:
                            right = b[1]
                        ax.axvspan(b[0], right, alpha=0.25, color='lightgray');
                        ax.axvline(b[0], color='gray', linestyle='--');
                        ax.axvline(right, color='gray', linestyle='--');
                else:
                    breakpoints = np.unique([item for sublist in self.bins for item in sublist])
                    for b in breakpoints:
                        ax.axvline(b, color='gray', linestyle='--');
            else:
                ax = sns.kdeplot(data=plot_data, x='Length', hue='Class', log_scale=self.log_scale, gridsize=self.resolution, bw_method='silverman', cut=0, common_grid=True, common_norm=False, bw_adjust=self.bw*2, ax=plot_ax);

                if self.breakpoint_type == 'clustering':
                    for mu, var, w in zip(self.guassian_model.means_, self.guassian_model.covariances_, self.guassian_model.weights_):
                        ax.axvline(mu, color='lightgray', linestyle='-', ymin=0, ymax=w);

                breakpoints = np.unique([item for sublist in self.bins for item in sublist])
                for b in breakpoints:
                    ax.axvline(b, color='gray', linestyle='--');

        if not subplot:
            plt.title("variants");
            plt.show();

    @staticmethod
    def unstack(X):
        """Converts a Series of lists of lengths into a Series of lengths with non-unique index

        Args:
            X (Series): A Series of lists of lengths

        Returns:
            Series: A Series of lengths
        """
        result = pd.DataFrame({'sampleid': np.repeat(X.index, X.str.len()), 'LEN': np.concatenate(X.values)}).set_index('sampleid').LEN

        return result

    @staticmethod
    def stack(lengths):
        """Converts a Series of lengths with non-unique index into a Series of lists of lengths

        Args:
            X (Series): A Series of lengths

        Returns:
            Series: A Series of lists of lengths
        """
        result = lengths.groupby(lengths.index).apply(list)

        return result
