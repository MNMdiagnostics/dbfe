import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from .DistributionBasedFeatureExtractor import DistributionBasedFeatureExtractor

class MultiTypeDBFE(BaseEstimator, TransformerMixin):
    def __init__(self, svclass_col='SVCLASS', len_col='LEN', _verb=False, dbfe_cols=[], custom_dbfe_args=dict(), **dbfe_args):
        """Constructor

        Args:
            svclass_col (str): Name of the column containing svclass label.
            len_col (str): Name of the column containing length value.
            verb (bool): If print steps.
            dbfe_cols (list): A list of columns for DBFEing. When empty, all columns are DBFEied.
            custom_dbfe_args (dict): A dict with costum dbfe args assigned to columns. Those args overwrite dbfe_args for a given column.
            dbfe_args: Keyword arguments passed to DistributionBasedFeatureExtractor().

        """
        self.dbfe_args = dbfe_args
        self.svclass_col = svclass_col
        self.len_col = len_col
        self.extractors = dict()
        self._verb = _verb
        self.dbfe_cols = dbfe_cols
        self.custom_dbfe_args = custom_dbfe_args

    def fit(self, X, y):
        """Creates breakpoints based on the provided lengths and classes for each svclass

        Args:
            X (DataFrame): A DataFrame of lists of lengths for each svclass
            y (Series): A series of labels

        Raises:
            Exception: Unsupported transformation type.

        Returns:
            self: A wrapper with fitted extractor ready to transform new datasets according to trained breakpoints for each svclass
        """
        for col, X_lengths in self._iter_dbfe_columns(X):

            X_lengths, y_common = self._get_common(X_lengths, y)

            self.extractors[col] = DistributionBasedFeatureExtractor(**self._get_dbfe_args(col))

            self.extractors[col] = self.extractors[col].fit(X_lengths, y_common)
        return self

    def transform(self, X):
        """Transforms lengths for svclasses in input dataframe into features based on bins. Requires prior running of fit method.

        Args:
            X (DataFrame): A DataFrame with lists of lengths for each svclass to transform into features

        Returns:
            DataFrame: A DataFrame containing new features
        """
        X_out = X.copy()
        for col, X_lengths in self._iter_dbfe_columns(X):
            X_out_new = self.extractors[col].transform(X_lengths)
            X_out_new = X_out_new.rename(columns={col_name: f'{col}_{col_name}' for col_name in X_out_new.columns})
            X_out = X_out.join(X_out_new)
            X_out = X_out.drop(columns=col)
        X_out = X_out.fillna(0)
        return X_out

    def fit_transform(self, X, y):
        """Performs fit and then transform; optimised to not repeat same operations (that would happen when calling fit() and transform() methods)

        Args:
            X (Series): A Series of tuples of pairs (svclass, lengths)
            y (Series): A series of classes

        Returns:
            DataFrame: A DataFrame containing new features
        """
        self.fit(X, y)
        return self.transform(X)

    def plot_data_with_breaks(self, X, y, plot_type='hist', **fig_args):
        """Plots lengths with indicated bins for each

        Args:
            X (DataFrame): A DataFrame of lists of lengths for each svclass
            y (Series): Classes to differentiate on the plot
            plot_type (str): Type of plot to show. Available options: hist or kde.
            fig_args: Keyword arguments passed to pyplot.figure()
        """
        fig = plt.figure(**fig_args)
        nrows, ncols = self._get_layout(len(self.dbfe_cols) if self.dbfe_cols else X.columns.shape[0])
        i = 0
        for col, X_lengths in self._iter_dbfe_columns(X):

            X_lengths, y_common = self._get_common(X_lengths, y)

            ax = fig.add_subplot(nrows, ncols, i+1)
            i += 1
            self.extractors[col].plot_data_with_breaks(
                X_lengths,
                y_common,
                plot_type=plot_type,
                plot_ax=ax,
                subplot=True
            )
            ax.set_title(f'{col} variants')
        plt.show()

    def _verb_print(self, *args):
        """Prints when verb=True

        Args:
            *args (list/tuple): arguments for print() function
        """
        if (self._verb):
            print(*args)

    def _get_layout(self, nplots):
        """Get number of rows and columns for subplots layout
            Source: https://github.com/pandas-dev/pandas/blob/945c9ed766a61c7d2c0a7cbb251b6edebf9cb7d5/pandas/plotting/_matplotlib/tools.py#L110

        Args:
            nplots: number of plots

        Returns:
            (int, int): pair - (num_rows, num_cols)
        """
        layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
        try:
            return layouts[nplots]
        except KeyError:
            k = 1
            while k ** 2 < nplots:
                k += 1

            if (k - 1) * k >= nplots:
                return k, (k - 1)
            else:
                return k, k

    def _iter_dbfe_columns(self, X):
        """Generator extracting DBFEable columns from input dataframe

        Args:
            X (DataFrame): A DataFrame with columns with lists of lengths for svclasses

        Yields:
            (str, Series): name of a column for DBFE and the column
        """
        for col in X.columns:
            if self.dbfe_cols and col not in self.dbfe_cols:
                continue
            self._verb_print(f'Current column: {col}')
            dbfe_col = X[col]
            dbfe_col = dbfe_col[~dbfe_col.isna()]
            yield col, dbfe_col

    def _get_common(self, X, y):
        """Returns subsets of X and y for indexes present in both

        Args:
            X (DataFrame): A DataFrame to be processed
            y (Series): A Series to be processed

        Returns:
            (DataFrame, Series): Subsets of, respectively, X and y for indexes present in both
        """
        common_indices = np.intersect1d(np.unique(X.index), y.index)
        self._verb_print(f'Samples from y not found in X: {np.setdiff1d(y.index, np.unique(X.index))}')
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        return X, y

    def _get_dbfe_args(self, col):
        """Returns dbfe arguments for a given column

        Args:
            col (str): name of a column

        Returns:
            dict: Args for dbfe
        """
        try:
            return self.custom_dbfe_args[col]
        except KeyError:
            return self.dbfe_args

    @staticmethod
    def unstack(data, svclass_col='SVCLASS', len_col='LEN', id_col='sampleid'):
        """Converts a DataFrame of lists of lengths for each svclass to a Dataframe of svclasses and lengths with non-unique index

        Args:
            data (DataFrame): A DataFrame of lists of lengths for each svclass
            svclass_col (str): Name of the column which should contain svclass label.
            len_col (str): Name of the column which should contain length value.
            id_col (str): Name of the column which should contain ids.

        Returns:
            DataFrame: A DataFrame with svclasses and lengths
        """
        out_data = pd.DataFrame()
        for col in data.columns:
            svclass = col.split('_')[1] if '_' in col else col
            svclass_data = data[svclass]
            svclass_data = svclass_data[~svclass_data.isna()]
            out_data = pd.concat([out_data,
                                 pd.DataFrame({
                                     id_col: np.repeat(svclass_data.index, svclass_data.str.len()),
                                     svclass_col: svclass,
                                     len_col: np.concatenate(svclass_data)
                                 })
                                 ])
        return out_data.set_index(id_col)

    @staticmethod
    def stack(data, svclass_col='SVCLASS', len_col='LEN', source_name=None):
        """Converts a DataFrame of svclasses and lengths with non-unique index into a DataFrame of lists of lengths for each svclass

        Args:
            data (DataFrame): A DataFrame with svclasses and lengths
            svclass_col (str): Name of the column containing svclass label.
            len_col (str): Name of the column containing length value.
            source_name (str, optional): If provided, name of source file type (i.e. CNV or SV), that will be added to column prefixes

        Returns:
            DataFrame: A DataFrame of lists of lengths for each svclass
        """
        svclasses = np.unique(data[svclass_col])
        out_data = pd.DataFrame(index=np.unique(data.index))
        for svclass in svclasses:
            lengths = data[data[svclass_col]==svclass][len_col]
            lengths = lengths[lengths > 0]

            sv_col = DistributionBasedFeatureExtractor.stack(lengths)
            sv_col.name = f'{source_name}_{svclass}' if source_name else f'{svclass}'
            out_data = out_data.join(sv_col)

        return out_data
