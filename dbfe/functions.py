import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


def generate_equal_bins(lengths, n_bins=4, log_scale=True):
    """Generates bins of equal width

    Args:
        lengths (Series): A Series of lengths to be discretized
        n_bins (int, optional): Number of bins to generate. Defaults to 4.
        log_scale (bool, optional): Whether to generate equal bins in linear or logarithmic scale. Defaults to True.

    Returns:
        list: A list of bins
    """
    if log_scale:
        breakpoints = np.logspace(np.log10(lengths.min()), np.log10(lengths.max()), n_bins + 1)[1:-1]
    else:
        breakpoints = np.linspace(lengths.min(), lengths.max(), n_bins + 1)[1:-1]

    result = breakpoints_to_bins(breakpoints)

    return result


def generate_quantile_bins(lengths, n_bins=4):
    """Generates bins based on quantile binning. Uses sklearn.preprocessing.KBinsDiscretizer.

    Args:
        lengths (Series): A Series or DataFrame containing lengths to be discretized
        n_bins (int, optional): Number of bins to generate, e.g., 4 bins result in 3 breakpoints. Defaults to 4.

    Returns:
        array: An array of bins
    """

    data = lengths.to_frame()

    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    est.fit(data)

    breakpoints = np.around(est.bin_edges_[0][1:-1]).astype('int')
    
    result = breakpoints_to_bins(breakpoints)

    return result


def generate_clustering_bins(lengths, no_clusters, random_state):

    lengths_df = pd.DataFrame(lengths)

    model = GaussianMixture(n_components=no_clusters, init_params='kmeans', random_state=random_state)
    model.fit(lengths_df)
    cluster_assignment = model.predict(lengths_df)

    lengths_df['Cluster'] = cluster_assignment
    lengths_df.Cluster = lengths_df.Cluster.astype("category")

    breakpoints = lengths_df.groupby('Cluster').min().sort_values(by='LEN').LEN.to_numpy()[1:]

    result = breakpoints_to_bins(breakpoints)

    return result, model


def generate_supervised_bins(lengths, classes, max_bins='auto', bw=0.5, resolution=200, log_scale=True, only_peaks=False, cv=None):
    """Generates bins based on differences of length distributions between classes.

    Args:
        lengths (DataFrame/Series): Lengths with indexes to corresponding samples
        classes (DataFrame/Series): Classes with indexes to corresponding samples
        max_bins (bool, optional): A parameter indicating how the breakpoints should be filtered. \'auto\' means that only the peaks exceeding standard deviation should be considered; int selects a given number of bins.
        bw (float, optional): Bandwidth for kde. Used only in supervised breakpoints. Defaults to 0.5.
        resolution (int, optimal): Resolution of kde. Used only in supervised breakpoints. Defaults to 200.
        log_scale (bool, optional): Whether log scale should be used on the x axis to calculate the bins. Used only in supervised breakpoints. Defaults to True.
        only_peaks (bool, optional): Whether the whole spectrum of lengths should be subdivided into features, or should only the actual peaks be selected. Used only in supervised breakpoints. Defaults to False.
        cv (int/object/None, optional): A parameter deciding whether the kde should be averaged over cv folds or if it should be done on the whole dataset. int - specifies the number of folds; object - a cross-validation objects specifying the details of cv; None - analysis done on the whole dataset. Defaults to None.

    Returns:
        array: An array of breakpoints
    """

    density = calculate_density(lengths, classes, bw, resolution, log_scale, cv)

    d1_mean = density[density['Class'] == 1].groupby(['Length']).mean().Density
    d0_mean = density[density['Class'] == 0].groupby(['Length']).mean().Density

    density_diff = (d1_mean - d0_mean).reset_index()

    breakpoints, peaks = calculate_breakpoints_and_peaks_from_density_diff(density_diff)

    breakpoints = np.around(breakpoints).astype('int')

    bins = breakpoints_to_bins(breakpoints)

    n_bins = calculate_n_bins(max_bins, density_diff, peaks, bins)

    return bins, limit_bins(bins, peaks, n_bins, only_peaks), density


def calculate_n_bins(max_bins, density_diff, peaks, bins):
    if max_bins == 'auto' or max_bins == 'sd':
        n_bins = np.sum((np.abs(peaks) >= np.std(density_diff.Density))*1)
    elif isinstance(max_bins, int):
        n_bins = max_bins
    elif isinstance(max_bins, float):
        n_bins = np.sum((np.abs(peaks) >= max_bins * np.max(np.abs(density_diff.Density)))*1)
    elif max_bins == 'all':
        n_bins = len(bins)
    else:
        raise Exception('Unsupported \"max_bins\" value, but this should have been detected earlier!!!')

    return n_bins


def calculate_breakpoints_and_peaks_from_density_diff(density_diff):
    """Calculates breakpoints and peaks from difference of length distributions

    Args:
        density_diff (DataFrame): A DataFrame consisting of two columns: 'length' and 'den_diff'

    Returns:
        array, array: An array of breakpoints and an array of peaks in each bin resulting from breakpoints
    """

    breakpoints = []
    peaks = []

    curr_peak = 0
    for i in range(0, len(density_diff) - 1):
        if abs(density_diff.Density[i]) > abs(curr_peak):
            curr_peak = density_diff.Density[i]

        if density_diff.Density[i] * density_diff.Density[i + 1] < 0:
            breakpoints.append((density_diff.Length[i] + density_diff.Length[i + 1]) / 2)
            peaks.append(curr_peak)
            curr_peak = 0
        
    peaks.append(curr_peak)

    return breakpoints, peaks


def calculate_density(lengths, classes, bw=0.5, resolution=200, log_scale=True, cv=None):
    """Calculates difference of distributions between two classes based on kde of lengths in each class.

    Args:
        lengths (DataFrame/Series): Lengths with indexes to corresponding samples
        classes (DataFrame/Series): Classes with indexes to corresponding samples. Classes need to encoded to 1 (positive) and 0 (negative).
        bw (float, optional): Bandwidth for kde. Used only in supervised breakpoints. Defaults to 0.5.
        resolution (int, optimal): Resolution of kde. Used only in supervised breakpoints. Defaults to 200.
        log_scale (bool, optional): Whether log scale should be used on the x axis to calculate the bins. Used only in supervised breakpoints. Defaults to True.
        cv (int/object/None, optional): A parameter deciding whether the kde should be averaged over cv folds or if it should be done on the whole dataset. int - specifies the number of folds; object - a cross-validation objects specifying the details of cv; None - analysis done on the whole dataset. Defaults to None.

    Returns:
        DataFrame: A DataFrame consisting of two columns: 'length' and 'den_diff', where den_diff is the difference in distribution between samples in two classes of a given length
    """

    lengths = lengths.rename('Length')
    classes = classes.rename('Class')

    density = pd.DataFrame()

    if log_scale:
        x_grid = np.linspace(np.log(lengths.min()), np.log(lengths.max()), resolution)
    else:
        x_grid = np.linspace(lengths.min(), lengths.max(), resolution)

    if cv is None:
        density = density.append(calculate_density_in_class(x_grid, lengths, classes, 1, bw, log_scale), ignore_index=True)
        density = density.append(calculate_density_in_class(x_grid, lengths, classes, 0, bw, log_scale), ignore_index=True)
    else:
        for train_index, _ in cv.split(X=classes, y=classes):
            lengths_train = lengths[lengths.index.isin(classes[train_index].index)]
            classes_train = classes.iloc[train_index]

            density = density.append(calculate_density_in_class(x_grid, lengths_train, classes_train, 1, bw, log_scale), ignore_index=True)
            density = density.append(calculate_density_in_class(x_grid, lengths_train, classes_train, 0, bw, log_scale), ignore_index=True)

    return density


def calculate_density_in_class(x, values, classes, c, bw, log_scale):
    kde_skl = KernelDensity(bandwidth=bw)
    classes = classes[classes.index.isin(values.index.unique())]
    values_c = values.loc[classes[classes == c].index].to_numpy()

    if log_scale:
        values_c = np.log(values_c)

    kde_skl.fit(values_c[:, np.newaxis])
    log_pdf = kde_skl.score_samples(x[:, np.newaxis])
    density = np.exp(log_pdf)

    if log_scale:
        result = pd.DataFrame({'Length': np.exp(x), 'Density': np.array(density), 'Class': c})
    else:
        result = pd.DataFrame({'Length': x, 'Density': np.array(density), 'Class': c})

    return result


def limit_bins(bins, peaks, max_bins, only_peaks):

    bins_filtered = []

    order = np.argsort(-1*np.abs(peaks))

    for i in range(np.min([max_bins, len(bins)])):
        index = order[i]
        bins_filtered.append(bins[index])

    bins_filtered_ordered = sorted(bins_filtered, key=lambda tup: tup[0])

    result = []

    if only_peaks:
        result = bins_filtered_ordered
    else:
        prev_end = 0
        for r in bins_filtered_ordered:
            if prev_end != r[0]:
                result.append((prev_end, r[0]))

            result.append(r)

            prev_end = r[1]

        if bins_filtered_ordered[len(bins_filtered_ordered) - 1][1] != np.inf:
            result.append((bins_filtered_ordered[len(bins_filtered_ordered) - 1][1], np.inf))

    return result


def breakpoints_to_bins(breakpoints):
    """Converts breakpoints into bins defined by tuples

    Args:
        breakpoints (list): A list of bins

    Returns:
        list: A list of tuples representing bins
    """
    return list(zip([0, *breakpoints], [*breakpoints, np.inf]))


def create_sample_features_from_bins(sample_lengths, bins, prefix):
    """Converts lengths of a given sample into features based on the supplied bins

    Args:
        sample_lengths (DataFrame/Series): Lengths to be converted
        bins (array): Bins
        prefix (str): Prefix for feature name

    Returns:
        DataFrame: A DataFrame containing all feature values for a given sample
    """

    feature_names = []
    values = []

    for b in bins:
        feature_names.append('{}_{}_{}'.format(prefix, b[0], b[1]))
        values.append(sample_lengths[(sample_lengths >= b[0]) & (sample_lengths < b[1])].shape[0])

    row = pd.DataFrame([values], columns = feature_names, index = sample_lengths.index.unique())

    return row


def create_features_from_bins(samples_lengths, bins, prefix):
    """Converts lengths into features based on the supplied bins

    Args:
        sample_lengths (DataFrame/Series): Lengths to be converted
        bins (array): Bins
        prefix (str): Prefix for feature name

    Returns:
        DataFrame: A DataFrame containing all feature values for all samples
    """
    if not samples_lengths.index.is_unique:
        raise ValueError(
            "Cannot use a dataset w non-unique indices: ",
            samples_lengths[samples_lengths.index.duplicated()].index
        )

    result = pd.DataFrame()
    
    for index in samples_lengths.index:
        single_sample_lengths = pd.Series(samples_lengths.loc[index], index=[index]*len(samples_lengths.loc[index]))
        sample_features = create_sample_features_from_bins(single_sample_lengths, bins, prefix)

        if result.empty:
            result = sample_features
        else: 
            result = result.append(sample_features)

    return result


def is_int_tuple(range_tuple):
    """
    Checks whether a tuple contains only ints
    :param range_tuple: tuple of values
    :return: True if the tuple contains only ints, False otherwise
    """
    return all(isinstance(v, int) for v in range_tuple)


def is_float_tuple(range_tuple):
    """
    Checks whether a tuple contains only floats
    :param range_tuple: tuple of values
    :return: True if the tuple contains only floats, False otherwise
    """
    return all(isinstance(v, float) for v in range_tuple)


def filter_values_to_range(values, value_range):
    """
    Filters a list of values according to a min-max range
    :param values: list of values to filter
    :param value_range: min-max tuple; tuple of ints defines the minimum and maximum values, tuple of floats defines
    the minimum and maximum quantile.
    :return: filtered list of values
    """
    if value_range[0] >= value_range[1]:
        raise ValueError("Minimum of desired range must be smaller than maximum. Got %s." % str(value_range))
    if not isinstance(value_range, tuple) or len(value_range) != 2:
        raise ValueError("Incorrect range. Expected tuple of size 2. Got %s." % str(value_range))
    if is_float_tuple(value_range):
        min_quantile = values.quantile(value_range[0])
        max_quantile = values.quantile(value_range[1])
        values = values.loc[(values >= min_quantile) & (values <= max_quantile)]
    elif is_int_tuple(value_range):
        values = values.loc[(values >= value_range[0]) & (values <= value_range[1])]
    else:
        raise ValueError(
            "Unexpected type of range. Expected a pair of ints or floats. Got %s." % str(value_range))
    return values
