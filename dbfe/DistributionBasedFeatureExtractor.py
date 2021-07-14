from sklearn.base import BaseEstimator, TransformerMixin

from .functions import generate_quantile_breakpoints, generate_supervised_breakpoints, create_features_from_breakpoints

class DistributionBasedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, breakpoint_type='quantile', n_bins=4, filter=True, feature_type='cnv', variant_type='del'):
        self.breakpoint_type = breakpoint_type
        self.n_bins = n_bins
        self.filter = filter
        self.feature_type = feature_type
        self.variant_type = variant_type

    def fit(self, X, y):
        if self.breakpoint_type == 'quantile':
            self.breakpoints = generate_quantile_breakpoints(X, self.n_bins)
        elif self.breakpoint_type == 'clustering':
            raise NotImplementedError()
        elif self.breakpoint_type == 'supervised':
            self.breakpoints = generate_supervised_breakpoints(X, y, self.filter)
        else:
            raise Exception('Unsupported transformation type! Only \'quantile\', \'clustering\', and \'supervised\' are supported.')

        return self

    def transform(self, X):
        features = create_features_from_breakpoints(X, self.breakpoints, self.feature_type, self.variant_type)

        return features

    def fit_transform(self, X, y):
        self.fit(X, y)

        return self.transform(X)
