import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer

from matplotlib import pyplot as plt

def generate_quantile_breakpoints(lengths, n_bins=4):
    """Generates breakpoints based on quantile binning. Uses sklearn.preprocessing.KBinsDiscretizer.

    Args:
        lengths (DataFrame/Series): A Series or DataFrame containing lengths to be discretized
        n_bins (int, optional): Number of bins to generate, e.g., 4 bins result in 3 breakpoints. Defaults to 4.

    Returns:
        array: An array of breakpoints
    """

    data = lengths.to_frame()

    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    est.fit(data)

    breakpoints = np.around(est.bin_edges_[0][1:-1]).astype('int')
    
    return breakpoints

def generate_supervised_breakpoints(lengths, classes, filter=True):
    """Generates breakpoints based on differences of length distributions between classes.

    Args:
        lengths (DataFrame/Series): Lengths with indexes to corresponding samples
        classes (DataFrame/Series): Classes with indexes to corresponding samples
        filter (bool, optional): A flag indicating if breakpoints should be filtered based on the height of the peaks in bins or not. If False, all breakpoints are returned. Defaults to True.

    Returns:
        array: An array of breakpoints
    """

    density_diff = calculate_density_difference(lengths, classes)

    breakpoints, peaks = calculate_breakpoints_and_peaks_from_density_diff(density_diff)

    if filter:
        return filter_breakpoints(breakpoints, peaks, np.std(density_diff.den_diff))
    else:
        return breakpoints

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
        if abs(density_diff.den_diff[i]) > abs(curr_peak):
            curr_peak = density_diff.den_diff[i]

        if density_diff.den_diff[i] * density_diff.den_diff[i + 1] < 0:
            breakpoints.append((density_diff.length[i] + density_diff.length[i + 1]) / 2)
            peaks.append(curr_peak)
            curr_peak = 0
        
    peaks.append(curr_peak)

    return breakpoints, peaks

def calculate_density_difference(lengths, classes):
    """Calculates difference of distributions between two classes based on kde of lengths in each class.

    Args:
        lengths (DataFrame/Series): Lengths with indexes to corresponding samples
        classes (DataFrame/Series): Classes with indexes to corresponding samples

    Returns:
        DataFrame: A DataFrame consisting of two columns: 'length' and 'den_diff', where den_diff is the difference in distribution between samples in two classes of a given length
    """

    lengths = lengths.rename('Length')
    classes = classes.rename('Class')
    data = lengths.to_frame().join(classes)

    ax = sns.kdeplot(data=data, x='Length', hue='Class', log_scale=True, gridsize=512, bw_method='silverman', cut=0, common_grid=True, common_norm=False);

    x_axis = ax.get_lines()[0].get_data()[0];
    den_0 = ax.get_lines()[0].get_data()[1];
    den_1 = ax.get_lines()[1].get_data()[1];

    plt.clf();

    density_diff = pd.DataFrame({'length': x_axis, 'den_diff': np.array(den_0) - np.array(den_1)})

    return density_diff

def filter_breakpoints(breakpoints, peaks, peak_threshold):
    """Filters breakpoints based on the peaks in each bin. Only peaks higher than a given threshold pass.

    Args:
        breakpoints (array): Breakpoints
        peaks (array): Peaks in bins between breakpoints
        peak_threshold (float): A minimal height of a peak which leaves its adjacent breakpoints

    Returns:
        array: A filtered list of breakpoints
    """

    range_ends = [0, *breakpoints, np.inf]
    breakpoints_filtered = []

    for i in range(0, len(peaks)):
        if np.abs(peaks[i]) >= peak_threshold:
            breakpoints_filtered.append(range_ends[i])
            breakpoints_filtered.append(range_ends[i + 1])

    breakpoints_filtered = np.around(np.unique([0, np.inf, *breakpoints_filtered])[1:-1]).astype('int')

    return breakpoints_filtered

def create_sample_features_from_breakpoints(sample_lengths, breakpoints, feature_type, variant_type):
    """Converts lengths of a given sample into features based on the supplied breakpoints

    Args:
        sample_lengths (DataFrame/Series): Lengths to be converted
        breakpoints (array): Breakpoints
        feature_type (str): Type of feature (e.g., cnv or sv); used as a prefix for feature name
        variant_type (str): Type of variant (e.g., del, dup, translocation); used as a prefix for feature name

    Returns:
        DataFrame: A DataFrame containing all feature values for a given sample
    """

    feature_names = []
    values = []

    feature_name_prefix = '{}_{}_len'.format(feature_type, variant_type)
    range_ends = [0, *breakpoints, np.inf]

    for i in range(0, len(range_ends) - 1):
        feature_names.append('{}_{}_{}'.format(feature_name_prefix, range_ends[i], range_ends[i + 1]))
        values.append(sample_lengths[(sample_lengths >= range_ends[i]) & (sample_lengths < range_ends[i + 1])].shape[0])

    row = pd.DataFrame([values], columns = feature_names, index = sample_lengths.index.unique())

    return row

def create_features_from_breakpoints(samples_lengths, breakpoints, feature_type, variant_type):
    """Converts lengths into features based on the supplied breakpoints

    Args:
        sample_lengths (DataFrame/Series): Lengths to be converted
        breakpoints (array): Breakpoints
        feature_type (str): Type of feature (e.g., cnv or sv); used as a prefix for feature name
        variant_type (str): Type of variant (e.g., del, dup, translocation); used as a prefix for feature name

    Returns:
        DataFrame: A DataFrame containing all feature values for all samples
    """

    result = pd.DataFrame()
    
    for index in samples_lengths.index.unique():
        sample_lengths = samples_lengths[samples_lengths.index == index]

        sample_features = create_sample_features_from_breakpoints(sample_lengths, breakpoints, feature_type, variant_type)

        if result.empty:
            result = sample_features
        else: 
            result = result.append(sample_features)

    return result

def plot_data_with_breaks(data, breaks, plot_type='hist'):
    """Plots lengths with indicated breakpoints

    Args:
        data (DataFrame/Series): Lengths to be plotted
        breaks (array): An array of breakpoints to mark on the plot
        plot_type (str): Type of plot to show. Available options: hist or kde.
    """
    
    if plot_type == 'hist':
        logbreaks = np.logspace(np.log10(1.0), np.log10(1e8), 300)
        data.hist(bins=logbreaks);
        plt.gca().set_xscale("log")
    elif plot_type == 'kde':
        ax = sns.kdeplot(data=data, x='Length', hue='Class', log_scale=True, gridsize=512, bw_method='silverman', cut=0, common_grid=True, common_norm=False);

    for b in breaks:
        plt.axvline(b, color='orange', linestyle='--')

    plt.title("variants");
    plt.show();

def extract_features(feature_type, sv_classes, train_set, y_train, test_set=None, bp_type='supervised'):
    """A shortcut for feature extraction for a given type of variant with a given set of possible variant classes. Calculates separate features for each variant class.

    Args:
        feature_type (str): sv or cnv
        sv_classes (list): A list of possible variant classes (SVCLASS)
        train_set (DataFrame): A data frame with at least two attributes: LEN and SVCLASS
        y_train (Series/DataFrame): Classes for supervised breakpoints
        test_set (DataFrame, optional): Additional data frame to create features based on train_set. Defaults to None.
        bp_type (str, optional): Type of features to generate. Available options: supervised and quantile. Defaults to 'supervised'.

    Returns:
        DataFrame, DataFrame: A set of features for the input training set (and testing set, if provided, otherwise - None).
    """
    features_train = pd.DataFrame(index = train_set.index.unique())

    if test_set is not None:
        features_test = pd.DataFrame(index = test_set.index.unique())

    for sv_class in sv_classes:
        lengths_train = train_set[train_set.SVCLASS==sv_class].LEN

        if bp_type == 'supervised':
            breakpoints = generate_supervised_breakpoints(lengths_train, y_train)
        elif bp_type == 'quantile':
            breakpoints = generate_quantile_breakpoints(lengths_train)

        sv_class_features_train = create_features_from_breakpoints(lengths_train, breakpoints, feature_type, sv_class)

        features_train = features_train.join(sv_class_features_train)

        if test_set is not None:
            lengths_test = test_set[test_set.SVCLASS==sv_class].LEN
            sv_class_features_test = create_features_from_breakpoints(lengths_test, breakpoints, feature_type, sv_class)
            features_test = features_test.join(sv_class_features_test)

    if test_set is not None:
        return features_train.fillna(0), features_test.fillna(0)
    else:
        return features_train.fillna(0), None