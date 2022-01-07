import os

import pytest
import pandas as pd
import numpy as np
from dbfe import DistributionBasedFeatureExtractor

SEED = 23
SAMPLE_INDEX = ["SAMPLE_1", "SAMPLE_2", "SAMPLE_3", "SAMPLE_4", "SAMPLE_5",
                "SAMPLE_6", "SAMPLE_7", "SAMPLE_8", "SAMPLE_9", "SAMPLE_10"]
DUPLICATED_INDEX = ["SAMPLE_1", "SAMPLE_2", "SAMPLE_3", "SAMPLE_4", "SAMPLE_5",
                    "SAMPLE_1", "SAMPLE_7", "SAMPLE_8", "SAMPLE_9", "SAMPLE_5"]


@pytest.fixture
def sample_X():
    return pd.Series([[1, 10, 10], [100, 200, 200, 300], [1, 6, 7, 20], [10, 300, 500, 50, 40],
                      [10000, 2000, 5000, 6000, 700], [10, 30, 500, 50, 40], [810000, 82000, 58000, 68000, 8700],
                      [106000, 62000, 65000, 66000, 6700], [910000, 92000, 95000, 96000, 9700],
                      [100000, 20000, 50000, 60000, 10]],
                     name='foo', index=pd.Index(SAMPLE_INDEX, name='sampleid'))


@pytest.fixture
def sample_y():
    return pd.Series([1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                     name='bar', index=pd.Index(SAMPLE_INDEX, name='sampleid'))


def real_data(cancer_type, stat_type, sv_class, allowed_labels, positive_class_label):
    test_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    stat_vals = pd.read_csv(os.path.join(test_data_folder, f"{cancer_type}_{stat_type}.csv.gz"),
                                         index_col='SAMPLEID')
    stat_vals.loc[:, "SVCLASS"] = stat_vals.SVCLASS.str.upper()
    stat_vals = stat_vals.loc[stat_vals.SVCLASS == sv_class.upper(), :]
    stat_vals = stat_vals.groupby(stat_vals.index)['LEN'].apply(list).to_frame()
    stat_vals.loc[:, "COUNT"] = stat_vals.apply(lambda x: len(x.LEN), axis=1)

    labels = pd.read_csv(os.path.join(test_data_folder, f"{cancer_type}_labels.tsv"), sep='\t', index_col=0)
    labels = labels.loc[labels.CLASS_LABEL.isin(allowed_labels), :]
    labels = (labels == positive_class_label)*1
    stat_df = stat_vals.join(labels.CLASS_LABEL, how='inner')

    return stat_df.LEN, stat_df.CLASS_LABEL


def test_unstack(sample_X, sample_y):
    extractor = DistributionBasedFeatureExtractor(breakpoint_type="supervised", n_bins="auto")
    assert extractor.unstack(sample_X).shape[0] == 46


@pytest.mark.parametrize("method,bins,expected,only_peaks", [
    ("equal", 1, 1, False), ("equal", 2, 2, False), ("equal", 5, 5, False), ("equal", 10, 10, False), 
    ("quantile", 1, 1, False), ("quantile", 2, 2, False), ("quantile", 5, 5, False), ("quantile", 10, 10, False), 
    ("clustering", 1, 1, False), ("clustering", 2, 2, False), ("clustering", 5, 5, False), ("clustering", 10, 10, False),
    ("supervised", 1, 1, True), ("supervised", 2, 2, True), ("supervised", "auto", 2, True),
])
def test_bin_count(sample_X, sample_y, method, bins, expected, only_peaks):
    extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks)
    extractor.fit(sample_X, sample_y)

    assert len(extractor.bins) == expected


@pytest.mark.parametrize("method,bins", [
    ("equal", -1), ("equal", 0), ("equal", 2.5), ("equal", 11), ("equal", 'auto'),
    ("quantile", -1), ("quantile", 0), ("quantile", 2.5), ("quantile", 11), ("quantile", 'auto'),
    ("clustering", -1), ("clustering", 0), ("clustering", 2.5), ("clustering", 11), ("clustering", 'auto'),
    ("supervised", -1), ("supervised", 0), ("supervised", 2.5), ("supervised", 11)
])
def test_incorrect_bin_count(sample_X, sample_y, method, bins):
    try:
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins)
        extractor.fit(sample_X, sample_y)
        assert False
    except:
        assert True


@pytest.mark.parametrize("ys", [
    ([1, 1, 1, 1, 0, 1, 0, 0, 0, 2]), ([2, 2, 2, 2, 0, 2, 0, 0, 0, 0]),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ([1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    (['RES', 'RES', 'RES', 'RES', 'NONRES', 'RES', 'NONRES', 'NONRES', 'NONRES', 'NONRES'])
])
def test_incorrect_y(sample_X, ys):
    try:
        extractor = DistributionBasedFeatureExtractor(breakpoint_type='supervised')
        extractor.fit(sample_X, ys)
        assert False
    except:
        assert True


@pytest.mark.parametrize("Xs", [
    (pd.Series([[1, 10, 10], [100, 200, 200, 300], [1, 6, 7, 20], [10, 300, 500, 50, 40],
                      [-10000, 2000, 5000, 6000, 700], [10, 30, 500, 50, 40], [810000, 82000, 58000, 68000, 8700],
                      [106000, 62000, 65000, 66000, 6700], [910000, 92000, 95000, 96000, 9700],
                      [100000, 20000, 50000, 60000, 10]],
                     name='foo', index=pd.Index(SAMPLE_INDEX, name='sampleid'))),
    (pd.Series([[1, 10, 10], [100, 200, 200, 300], [1, 6, 7, 20], [10, 300, 500, 50, 40],
                      [0, 2000, 5000, 6000, 700], [10, 30, 500, 50, 40], [810000, 82000, 58000, 68000, 8700],
                      [106000, 62000, 65000, 66000, 6700], [910000, 92000, 95000, 96000, 9700],
                      [100000, 20000, 50000, 60000, 10]],
                     name='foo', index=pd.Index(SAMPLE_INDEX, name='sampleid')))
])
def test_incorrect_X(Xs, sample_y):
    try:
        extractor = DistributionBasedFeatureExtractor(breakpoint_type='supervised')
        extractor.fit(Xs, sample_y)
        assert False
    except:
        assert True


@pytest.mark.parametrize("method,bins", [("equal", 4), ("quantile", 4), ("clustering", 4), ("supervised", "auto")])
def test_real_data(method, bins):
    try:
        X, y = real_data("breast", "cnv", "del", ['ER+ HER2-', 'HER2+'], 'ER+ HER2-')
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins)
        extractor.fit(X, y)
        transformed_x = extractor.transform(X)
        assert len(transformed_x.columns) == len(transformed_x.columns.unique())
    except:
        assert False


@pytest.mark.parametrize("method,bins", [("equal", 4), ("quantile", 4), ("clustering", 4), ("supervised", "auto")])
def test_plots(method, bins):
    try:
        X, y = real_data("breast", "cnv", "del", ['ER+ HER2-', 'HER2+'], 'ER+ HER2-')
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins)
        extractor.fit(X, y)
        extractor.plot_data_with_breaks(X, y, plot_type='kde')
        assert True
    except:
        assert False

def test_nonunique_index(sample_X, sample_y):
    try:
        sample_X.index = DUPLICATED_INDEX
        sample_y.index = DUPLICATED_INDEX
        extractor = DistributionBasedFeatureExtractor(breakpoint_type="quantile", n_bins=3)
        extractor.fit_transform(sample_X, sample_y)
        assert False
    except ValueError as e:
        if "SAMPLE_1" in str(e) and "SAMPLE_5" in str(e):
            assert True
        else:
            assert False

@pytest.mark.parametrize("method,bins,only_peaks", [
    ("equal", 1, False), ("equal", 2, False), ("equal", 5, False),
    ("quantile", 1, False), ("quantile", 2, False), ("quantile", 5, False),
    ("clustering", 1, False), ("clustering", 2, False), ("clustering", 5, False),
    ("supervised", 1, True), ("supervised", 2, True), ("supervised", "auto", True),
])
def test_range_filtering(sample_X, sample_y, method, bins, only_peaks):
    extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks, value_range=(20, 10000))
    extractor.fit_transform(sample_X, sample_y)

    extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks, value_range=(0.1, 0.9))
    extractor.fit_transform(sample_X, sample_y)

@pytest.mark.parametrize("method,bins,only_peaks", [
    ("equal", 1, False), ("equal", 2, False), ("equal", 5, False),
    ("quantile", 1, False), ("quantile", 2, False), ("quantile", 5, False),
    ("clustering", 1, False), ("clustering", 2, False), ("clustering", 5, False),
    ("supervised", 1, True), ("supervised", 2, True), ("supervised", "auto", True),
])
def test_frac_features(sample_X, sample_y, method, bins, only_peaks):
    extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks,
                                                  include_counts=True, include_fracs=True, include_total=False)
    features = extractor.fit_transform(sample_X, sample_y)

    fracs_length = len([col for col in features.columns if col.startswith('frac')])
    counts_length = len([col for col in features.columns if not col.startswith('frac')])

    assert fracs_length == counts_length

    if not only_peaks:
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks, include_counts=True, include_total=True)
        features = extractor.fit_transform(sample_X, sample_y)

        total_col = [col for col in features.columns if col.startswith('total')]
        count_cols = [col for col in features.columns if not col.startswith('total')]

        assert all(features[count_cols].sum(axis=1) == features[total_col])

    extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks, include_counts=True, include_total=True, include_fracs=True)
    features = extractor.fit_transform(sample_X, sample_y)

    total_col = [col for col in features.columns if col.startswith('total')][0]
    frac_cols = [col for col in features.columns if col.startswith('frac')]
    count_cols = [col for col in features.columns if not col.startswith('total') and not col.startswith('frac')]

    computed_counts = features[frac_cols].multiply(features[total_col], axis=0).astype(int)
    computed_counts.columns = features[count_cols].columns

    all(computed_counts == features[count_cols])

    if not only_peaks:
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=bins, only_peaks=only_peaks, include_counts=False, include_total=False, include_fracs=True)
        features = extractor.fit_transform(sample_X, sample_y)

        assert all(np.round(features.sum(axis=1), 10) == 1)