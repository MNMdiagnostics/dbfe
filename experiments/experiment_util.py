import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [12, 8]

from sklearn.base import clone
from sklearn.metrics import  precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif
from sklearn.cluster import AgglomerativeClustering

from IPython.display import display
from tqdm.notebook import tqdm

from scipy.stats import friedmanchisquare, shapiro, f_oneway, wilcoxon, ttest_rel, ttest_ind
from scikit_posthocs import posthoc_nemenyi_friedman

from dbfe import DistributionBasedFeatureExtractor

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def get_dataset(cancer_type, stat_type, sv_class, allowed_labels, pos_class):
    stat_vals = pd.read_csv(f"data/{cancer_type.lower()}/{cancer_type.lower()}_{stat_type.lower()}.csv.gz",
                            index_col='SAMPLEID', dtype={"CHROM": "object", "LEN": "int", "SVCLASS": "object"})
    stat_vals.loc[:, "SVCLASS"] = stat_vals.SVCLASS.str.upper()
    stat_vals = stat_vals.loc[stat_vals.SVCLASS == sv_class.upper(), :]
    stat_vals = stat_vals.groupby(stat_vals.index)['LEN'].apply(list).to_frame()
    stat_vals.loc[:, "COUNT"] = stat_vals.apply(lambda x: len(x.LEN), axis=1)

    labels = pd.read_csv(f"data/{cancer_type.lower()}/labels.tsv", sep='\t', index_col=0)
    labels = labels.loc[labels.CLASS_LABEL.isin(allowed_labels), :]
    labels = (labels == pos_class) * 1
    stat_df = stat_vals.join(labels.CLASS_LABEL, how='inner')

    return stat_df


def split_dataset(stat_df, test_size=0.3, seed=23):
    X = stat_df.LEN
    y = stat_df.CLASS_LABEL

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def calculate_metrics(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    result = pd.DataFrame({'Accuracy': [accuracy],
                           'Precision': [precision],
                           'Recall': [recall], 'F1': [f1],
                           'AUROC': [auc_roc],
                           'Kappa': [kappa],
                           'MCC': [mcc],
                           'Balanced accuracy': [balanced_acc]})

    return result


def sensitivity_test(X, y, breakpoint_type, cancer_type, variant_group, variant_type, algorithms, class_labels,
                     pos_class, cv, param_name, param_values, preset_params_func=None, random_state=23):
    result = pd.DataFrame()

    if preset_params_func is not None:
        param_name = "Evaluation"
        param_values = ["Cross-validation"]

    for param_value in tqdm(param_values):
        fold = 0
        for train_index, val_index in cv.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            fold = fold + 1

            for algorithm_name, algorithm in algorithms.items():
                if preset_params_func is None:
                    if isinstance(param_value, dict):
                        tested_params = param_value
                    else:
                        tested_params = {}
                        tested_params[param_name] = param_value

                    if breakpoint_type == "supervised":
                        tested_params["n_bins"] = "auto"
                else:
                    tested_params = preset_params_func(breakpoint_type, cancer_type, class_labels, variant_group,
                                                       variant_type, algorithm_name)

                extractor = DistributionBasedFeatureExtractor(breakpoint_type=breakpoint_type, random_state=random_state, **tested_params)
                clf = clone(algorithm)
                pipe = make_pipeline(extractor, StandardScaler(), clf)
                pipe.fit(X_train, y_train)

                y_prob = pipe.predict_proba(X_val)
                y_pred = pipe.predict(X_val)
                row_result = calculate_metrics(y_val, y_pred, y_prob[:, 1])
                row_result['algorithm'] = algorithm_name
                row_result[param_name] = str(param_value)
                row_result['fold'] = fold

                result = result.append(row_result)

    cancer_variant_to_df_columns(result, breakpoint_type, cancer_type, class_labels, pos_class, variant_group,
                                 variant_type)

    return result


def multimethod_sensitivity_tests(cancer, variant, dbfe_methods, algorithms, cv, param_name, param_values,
                                  test_size, preset_params_func=None):
    results_for_variant = pd.DataFrame()
    allowed_classes, cancer_type, class_labels, pos_class, variant_group, variant_type = extract_cancer_varaint(cancer,
                                                                                                                variant)

    print(cancer, variant)
    # We only use training data for sensitivity analyses
    dataset = get_dataset(cancer_type, variant_group, variant_type, allowed_classes, pos_class=pos_class)
    X_train, X_test, y_train, y_test = split_dataset(dataset, test_size=test_size)

    # Perform a sensitivity test for each breakpoint type
    for breakpoint_type in dbfe_methods:
        result_breakpoint_type = sensitivity_test(X_train, y_train, breakpoint_type, cancer_type, variant_group,
                                                  variant_type, algorithms, class_labels, pos_class, cv,
                                                  param_name, param_values, preset_params_func)
        results_for_variant = results_for_variant.append(result_breakpoint_type)

    return results_for_variant


def cancer_variant_to_df_columns(result, breakpoint_type, cancer_type, class_labels, pos_class, variant_group,
                                 variant_type):
    result['cancer_type'] = cancer_type
    result['classes'] = str(class_labels)
    result['pos_class'] = pos_class
    result['variant_group'] = variant_group
    result['variant_type'] = variant_type
    result['breakpoint_type'] = breakpoint_type


def extract_cancer_varaint(cancer, variant):
    cancer_type = cancer[0]
    allowed_classes = cancer[1]
    variant_group = variant[0]
    variant_type = variant[1]
    class_labels = f"{allowed_classes[0]} vs {allowed_classes[1]}"
    pos_class = allowed_classes[0]
    return allowed_classes, cancer_type, class_labels, pos_class, variant_group, variant_type


def plot_sensitivity_results(df, param_name, results_folder, plot_name, style="approach", dashes=True):
    sns.set_theme(context="paper", style="ticks", font_scale=2, palette="tab10")
    df.loc[:, "dataset"] = df.loc[:, "cancer_type"] + "(" + df.loc[:, "classes"] + ")"
    df.loc[:, "variant"] = df.loc[:, "variant_group"] + ": " + df.loc[:, "variant_type"]
    df = df.rename(columns={"breakpoint_type": "approach", "AUROC": "AUC"})

    plot_df = df.groupby(["variant", "dataset", "algorithm", "approach", param_name]).mean().reset_index()

    g = sns.relplot(
        data=plot_df, x=param_name, y="AUC",
        col="algorithm", row="dataset",
        hue="variant", style=style,
        kind="line", dashes=dashes
    )

    g.set_titles("{row_name} | {col_name}")

    g.savefig(os.path.join(results_folder, plot_name + ".svg"))
    sns.set_context("notebook")


def get_best_cv_param(df, param_name):
    groups = df.groupby(["cancer_type", "classes", "variant_group", "variant_type", "algorithm", "breakpoint_type"])
    return df.loc[groups.AUROC.idxmax(), ["cancer_type", "classes", "variant_group", "variant_type", "algorithm",
                                          "breakpoint_type", param_name]]


def paper_ready_dbfe_plot(dataset, df, breakpoint_type, n_bins, labels, xlab, plot_name, results_folder, random_state=23):
    sns.set_context("paper", rc={"font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    extractor = DistributionBasedFeatureExtractor(breakpoint_type=breakpoint_type, n_bins=n_bins, only_peaks=True, random_state=random_state)
    extractor.fit(dataset.LEN, dataset.CLASS_LABEL)
    extractor.plot_data_with_breaks(dataset.LEN, dataset.CLASS_LABEL, plot_type='kde', plot_ax=ax, subplot=True)
    ax.tick_params(labelsize=22)
    ax.set_xlabel(xlab, size=24)
    ax.set_ylim([0.0, 0.4])
    plt.yticks(np.arange(0.0, 0.5, 0.1))
    plt.legend(title=None, loc='upper left', labels=labels)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='22')  # for legend title

    figure = ax.get_figure()
    figure.savefig(os.path.join(results_folder, plot_name + ".svg"))

    transformed_sample = pd.concat([extractor.transform(df.LEN), df.CLASS_LABEL], axis=1).tail()
    transformed_sample = pd.concat(
        [transformed_sample.drop(["CLASS_LABEL"], axis=1), transformed_sample.loc[:, "CLASS_LABEL"]], axis=1)
    transformed_sample.to_csv(os.path.join(results_folder, plot_name + "_sample.csv"))
    display(transformed_sample)

    sns.set_context("notebook")

    return extractor


def calculate_param_tests(full_df, param_name, metric, test=friedmanchisquare, post_hoc=True):
    algorithms = list(full_df.loc[:, "algorithm"].unique())
    breakpoint_types = list(full_df.loc[:, "breakpoint_type"].unique())

    for breakpoint_type in breakpoint_types:
        for algorithm in algorithms:
            print("========================================================")
            print("Approach: ", breakpoint_type, ", Classifier: ", algorithm)
            result_series = []
            df = full_df.loc[
                 (full_df.loc[:, "algorithm"] == algorithm) & (full_df.loc[:, "breakpoint_type"] == breakpoint_type), :]
            param_values = list(df.loc[:, param_name].unique())

            for param_value in param_values:
                print(param_value, " mean value: ",
                      round(df.loc[df.loc[:, param_name] == param_value, metric].mean(), 4))
                result_series.append(df.loc[df.loc[:, param_name] == param_value, metric])

            _, p = test(*result_series)
            print("Test p-value for " + metric + ": " + str(round(p, 4)))
            print()

            if post_hoc:
                print("Nemenyi post-hoc");

                results_df = pd.DataFrame(np.array(result_series).T, columns=param_values)
                print(posthoc_nemenyi_friedman(results_df))

            print("========================================================")
            print()


def calculate_approach_tests(full_df, metric, test=friedmanchisquare, post_hoc=True):
    algorithms = list(full_df.loc[:, "algorithm"].unique())
    breakpoint_types = list(full_df.loc[:, "breakpoint_type"].unique())

    for algorithm in algorithms:
        print("========================================================")
        print("Classifier: ", algorithm)
        result_series = []
        df = full_df.loc[(full_df.loc[:, "algorithm"] == algorithm), :]

        for breakpoint_type in breakpoint_types:
            print(breakpoint_type, " mean value: ",
                  round(df.loc[df.loc[:, "breakpoint_type"] == breakpoint_type, metric].mean(), 4))
            result_series.append(df.loc[df.loc[:, "breakpoint_type"] == breakpoint_type, metric])

        _, p = test(*result_series)
        print("--------------------------------------------------------")
        print("Test p-value for " + metric + ": " + str(round(p, 4)))
        print("--------------------------------------------------------")
        print()

        if post_hoc:
            print("Nemenyi post-hoc");

            results_df = pd.DataFrame(np.array(result_series).T, columns=breakpoint_types)
            print(posthoc_nemenyi_friedman(results_df))

        print("========================================================")
        print()


def feature_importance_tests(cancer, variants, dbfe_methods, algorithms, preset_params_func=None, random_state=23):
    result = pd.DataFrame()
    allowed_classes, cancer_type, class_labels, pos_class, _, __ = extract_cancer_varaint(cancer, (None, None))
    print(cancer)

    pam50_geneamps = pd.read_csv("data/pam50_geneamps.csv", index_col=0)

    X_train = None
    X_test = None
    algorithm_name = "RF"

    for breakpoint_type in dbfe_methods:
        for variant in variants:
            variant_group = variant[0]
            variant_type = variant[1]
            dataset = get_dataset(cancer_type, variant_group, variant_type, allowed_classes, pos_class=pos_class)
            X_train_variant, X_test_variant, y_train_variant, y_test_variant = split_dataset(dataset)
            tested_param = preset_params_func(breakpoint_type, cancer_type, class_labels, variant_group, variant_type,
                                              algorithm_name)
            extractor = DistributionBasedFeatureExtractor(
                prefix=breakpoint_type + "_" + variant_group + "_" + variant_type + "_", random_state=random_state,
                breakpoint_type=breakpoint_type, **tested_param)
            extractor.fit(X_train_variant, y_train_variant)

            if X_train is None:
                X_train = extractor.transform(X_train_variant)
                X_test = extractor.transform(X_test_variant)
                y_train = y_train_variant
                y_test = y_test_variant
            else:
                X_train = X_train.join(extractor.transform(X_train_variant))
                X_test = X_test.join(extractor.transform(X_test_variant))
                y_train = y_train.append(y_train_variant)
                y_test = y_test.append(y_test_variant)

    X_train = X_train.join(pam50_geneamps)
    X_test = X_test.join(pam50_geneamps)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.groupby(y_train.index).first().loc[X_train.index]
    y_test = y_test.groupby(y_test.index).first().loc[X_test.index]

    result["feature"] = X_train.columns
    result["F train"], result["p-value train"] = f_classif(X_train, y_train)
    result["F test"], result["p-value test"] = f_classif(X_test, y_test)
    result['cancer_type'] = cancer_type
    result['classes'] = str(class_labels)
    result['pos_class'] = pos_class

    return result


def best_param_value(best_df, param_name, breakpoint_type, cancer_type, class_labels, variant_group, variant_type,
                     algorithm):
    return best_df.loc[(best_df.cancer_type == cancer_type) & (best_df.classes == class_labels)
                       & (best_df.variant_group == variant_group) & (best_df.variant_type == variant_type)
                       & (best_df.breakpoint_type == breakpoint_type) & (
                                   best_df.algorithm == algorithm), param_name].values[0]


def multimethod_holdout_tests(cancer, variant, dbfe_methods, algorithms, preset_params_func=None, random_state=23):
    result = pd.DataFrame()
    allowed_classes, cancer_type, class_labels, pos_class, variant_group, variant_type = extract_cancer_varaint(cancer,
                                                                                                                variant)

    print(cancer, variant)
    dataset = get_dataset(cancer_type, variant_group, variant_type, allowed_classes, pos_class=pos_class)
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    for breakpoint_type in dbfe_methods:
        for algorithm_name, algorithm in algorithms.items():
            tested_param = preset_params_func(breakpoint_type, cancer_type, class_labels, variant_group, variant_type,
                                              algorithm_name)

            extractor = DistributionBasedFeatureExtractor(breakpoint_type=breakpoint_type, random_state=random_state, **tested_param)
            clf = clone(algorithm)
            pipe = make_pipeline(extractor, clf)
            pipe.fit(X_train, y_train)

            y_prob = pipe.predict_proba(X_test)
            y_pred = pipe.predict(X_test)

            row_result = calculate_metrics(y_test, y_pred, y_prob[:, 1])
            row_result['algorithm'] = algorithm_name
            cancer_variant_to_df_columns(row_result, breakpoint_type, cancer_type, class_labels, pos_class,
                                         variant_group, variant_type)

            result = result.append(row_result)

    return result


def multivariant_holdout_tests(cancer, variants, dbfe_methods, algorithms, preset_params_func=None, random_state=23):
    result = pd.DataFrame()
    allowed_classes, cancer_type, class_labels, pos_class, _, __ = extract_cancer_varaint(cancer, (None, None))
    print(cancer)

    pam50_geneamps = pd.read_csv("data/pam50_geneamps.csv", index_col=0)

    for breakpoint_type in dbfe_methods:
        for algorithm_name, algorithm in algorithms.items():
            X_train = None
            X_test = None

            for variant in variants:
                variant_group = variant[0]
                variant_type = variant[1]
                dataset = get_dataset(cancer_type, variant_group, variant_type, allowed_classes, pos_class=pos_class)
                X_train_variant, X_test_variant, y_train_variant, y_test_variant = split_dataset(dataset)
                tested_param = preset_params_func(breakpoint_type, cancer_type, class_labels, variant_group,
                                                  variant_type,
                                                  algorithm_name)
                extractor = DistributionBasedFeatureExtractor(prefix=variant_group + "_" + variant_type + "_",
                                                              random_state=random_state,
                                                              breakpoint_type=breakpoint_type, **tested_param)
                extractor.fit(X_train_variant, y_train_variant)

                if X_train is None:
                    X_train = extractor.transform(X_train_variant)
                    X_test = extractor.transform(X_test_variant)
                    y_train = y_train_variant
                    y_test = y_test_variant
                else:
                    X_train = X_train.join(extractor.transform(X_train_variant))
                    X_test = X_test.join(extractor.transform(X_test_variant))
                    y_train = y_train.append(y_train_variant)
                    y_test = y_test.append(y_test_variant)

            X_train = X_train.join(pam50_geneamps)
            X_test = X_test.join(pam50_geneamps)
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            y_train = y_train.groupby(y_train.index).first().loc[X_train.index]
            y_test = y_test.groupby(y_test.index).first().loc[X_test.index]

            # combined variants
            scaler = StandardScaler()
            X_train_variants = scaler.fit_transform(X_train.drop(pam50_geneamps.columns, axis=1))
            X_test_variants = scaler.transform(X_test.drop(pam50_geneamps.columns, axis=1))
            clf = clone(algorithm)
            clf.fit(X_train_variants, y_train)
            y_prob = clf.predict_proba(X_test_variants)
            y_pred = clf.predict(X_test_variants)

            row_result = calculate_metrics(y_test, y_pred, y_prob[:, 1])
            row_result['algorithm'] = algorithm_name
            row_result['features'] = X_train_variants.shape[1]
            cancer_variant_to_df_columns(row_result, breakpoint_type, cancer_type, class_labels, pos_class, "sv+cnv",
                                         "")
            result = result.append(row_result)

            # PAM 50 geneamps
            scaler = StandardScaler()
            X_train_geneamps = scaler.fit_transform(X_train.loc[:, pam50_geneamps.columns])
            X_test_geneamps = scaler.transform(X_test.loc[:, pam50_geneamps.columns])
            clf = clone(algorithm)
            clf.fit(X_train_geneamps, y_train)
            y_prob = clf.predict_proba(X_test_geneamps)
            y_pred = clf.predict(X_test_geneamps)

            row_result = calculate_metrics(y_test, y_pred, y_prob[:, 1])
            row_result['algorithm'] = algorithm_name
            row_result['features'] = X_train_geneamps.shape[1]
            cancer_variant_to_df_columns(row_result, breakpoint_type, cancer_type, class_labels, pos_class, "geneamps",
                                         "")
            result = result.append(row_result)

            # combined variants + PAM 50 geneamps
            scaler = StandardScaler()
            X_train_v_and_g = scaler.fit_transform(X_train)
            X_test_v_and_g = scaler.transform(X_test)
            clf = clone(algorithm)
            clf.fit(X_train_v_and_g, y_train)
            y_prob = clf.predict_proba(X_test_v_and_g)
            y_pred = clf.predict(X_test_v_and_g)

            row_result = calculate_metrics(y_test, y_pred, y_prob[:, 1])
            row_result['algorithm'] = algorithm_name
            row_result['features'] = X_train_v_and_g.shape[1]
            cancer_variant_to_df_columns(row_result, breakpoint_type, cancer_type, class_labels, pos_class,
                                         "sv+cnv+geneamps", "")
            result = result.append(row_result)

    return result


def human_expert_tests(cancer, variants, algorithms):
    result = pd.DataFrame()
    allowed_classes, cancer_type, class_labels, pos_class, _, __ = extract_cancer_varaint(cancer, (None, None))
    pam50_geneamps = pd.read_csv("data/pam50_geneamps.csv", index_col=0)

    for algorithm_name, algorithm in algorithms.items():
        X_train = None
        X_test = None

        for variant in variants:
            variant_group = variant[0]
            variant_type = variant[1]
            expert_bins = variant[2]
            dataset = get_dataset(cancer_type, variant_group, variant_type, allowed_classes, pos_class=pos_class)
            X_train_variant, X_test_variant, y_train_variant, y_test_variant = split_dataset(dataset)
            extractor = DistributionBasedFeatureExtractor(prefix=variant_group + "_" + variant_type + "_",
                                                          bins=expert_bins, include_counts=True, include_fracs=False,
                                                          include_total=False)
            extractor.fit(X_train_variant, y_train_variant)

            if X_train is None:
                X_train = extractor.transform(X_train_variant)
                X_test = extractor.transform(X_test_variant)
                y_train = y_train_variant
                y_test = y_test_variant
            else:
                X_train = X_train.join(extractor.transform(X_train_variant))
                X_test = X_test.join(extractor.transform(X_test_variant))
                y_train = y_train.append(y_train_variant)
                y_test = y_test.append(y_test_variant)

        X_train = X_train.join(pam50_geneamps)
        X_test = X_test.join(pam50_geneamps)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = y_train.groupby(y_train.index).first().loc[X_train.index]
        y_test = y_test.groupby(y_test.index).first().loc[X_test.index]

        # combined variants
        scaler = StandardScaler()
        X_train_variants = scaler.fit_transform(X_train.drop(pam50_geneamps.columns, axis=1))
        X_test_variants = scaler.transform(X_test.drop(pam50_geneamps.columns, axis=1))
        clf = clone(algorithm)
        clf.fit(X_train_variants, y_train)
        y_prob = clf.predict_proba(X_test_variants)
        y_pred = clf.predict(X_test_variants)

        row_result = calculate_metrics(y_test, y_pred, y_prob[:, 1])
        row_result['algorithm'] = algorithm_name
        row_result['features'] = X_train_variants.shape[1]
        cancer_variant_to_df_columns(row_result, "expert", cancer_type, class_labels, pos_class, "expert", "")
        result = result.append(row_result)

    return result


def reformat_feature_name(feature_name):
    name_parts = feature_name.split("__")

    if len(name_parts) == 1:
        return name_parts[0].split("_")[2]
    else:
        name_part = name_parts[0]
        range_part = name_parts[1]

        name_subparts = name_part.split("_")
        if len(name_subparts) == 3:
            return name_subparts[1].upper() + " " + name_subparts[2].upper() + " [" + range_part.replace("_", "-") + "] (" + name_subparts[0] + ")"
        else:
            return name_subparts[2].upper() + " " + name_subparts[3].upper() + " [" + range_part.replace("_", "-") + "] (" + name_subparts[1] + " " + name_subparts[0] + ")"


def reformat_feature_name_clustergram(feature_name):
    name_parts = feature_name.split("__")

    if len(name_parts) == 1:
        return name_parts[0].split("_")[2].upper() + " " + name_parts[0].split("_")[3].upper() + " total count"
    else:
        name_part = name_parts[0]
        range_part = name_parts[1]

        name_subparts = name_part.split("_")
        if len(name_subparts) == 3:
            return name_subparts[1].upper() + " " + name_subparts[2].upper() + " [" + range_part.replace("_", "-") + "] (count)"
        else:
            return name_subparts[2].upper() + " " + name_subparts[3].upper() + " [" + range_part.replace("_", "-") + "] (frac)"


def plot_feature_importance(df, cancer_type, class_labels, plot_name, results_folder, top_k=10,
                            importance_col="F train"):
    sns.set_context("paper", rc={"font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_df = df.loc[(df.cancer_type == cancer_type) & (df.classes == str(class_labels)), :].sort_values(importance_col,
                                                                                                         ascending=False).head(
        top_k)
    plot_df.feature = plot_df.feature.apply(reformat_feature_name)
    fig = sns.barplot(x="F train", y="feature", data=plot_df,
                      label="Total", color="#77AADD", ax=ax)
    ax.tick_params(labelsize=22)
    ax.set_xlabel("Feature importance", size=24)
    ax.set(ylabel="");
    ax.tick_params(axis="y", direction="in", left=False)

    if cancer_type == "breast":
        ax.set_title(cancer_type + " (" + str(class_labels) + ")")
    else:
        ax.set_title(cancer_type)

    figure = ax.get_figure()
    figure.savefig(os.path.join(results_folder, plot_name + ".svg"))

    sns.set_context("notebook")


def get_breast_clustering_data(method, variants, seed):
    breast_df = None

    for variant in variants:
        dataset = get_dataset("breast", variant[0], variant[1], ['HER2+', 'ER+ HER2-', 'TNBC'], pos_class=['HER2+'])
        X = dataset.LEN
        extractor = DistributionBasedFeatureExtractor(breakpoint_type=method, n_bins=4, random_state=seed,
                                                      prefix=method + "_" + variant[0] + "_" + variant[1] + "_")
        extractor.fit(X, None)

        if breast_df is None:
            breast_df = extractor.transform(X)
        else:
            breast_df = breast_df.join(extractor.transform(X))

    breast_df = breast_df.dropna()
    labels = pd.read_csv(f"data/breast/labels.tsv", sep='\t', index_col=0)
    labels = labels[~labels.index.duplicated(keep='first')]

    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(MinMaxScaler().fit_transform(breast_df))
    breast_df.loc[:, "UMAP Component 1"] = embedding[:, 0]
    breast_df.loc[:, "UMAP Component 2"] = embedding[:, 1]
    breast_df.loc[:, "Subtype"] = labels.loc[breast_df.index].CLASS_LABEL.values
    clustering = AgglomerativeClustering(n_clusters=3).fit(
        MinMaxScaler().fit_transform(breast_df.drop(["UMAP Component 1", "UMAP Component 2", "Subtype"], axis=1)))
    breast_df.loc[:, "Cluster"] = clustering.labels_

    return breast_df


def plot_clusters(df, results_folder, filename, hue=None, color=sns.color_palette(["#222222"]), palette=None):
    sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18})

    if palette is not None:
        palette = sns.color_palette(palette)

    fig = sns.relplot(
        data=df, x="UMAP Component 1", y="UMAP Component 2",
        kind="scatter", s=35, hue=hue,
        color=color, palette=palette
    );
    try:
        fig._legend.remove()
        plt.legend(title=hue, fontsize='12', title_fontsize='14')
    except:
        pass
    plt.tick_params(labelsize=16)
    fig.savefig(os.path.join(results_folder, filename + ".svg"))

    sns.set_context("notebook")


def create_clustergram(df, results_folder, filename, color_mapping):
    droped_df = df.drop(["UMAP Component 1", "UMAP Component 2", "Subtype", "Cluster"], axis=1)
    plot_df = pd.DataFrame(MinMaxScaler().fit_transform(droped_df), columns=droped_df.columns, index=droped_df.index)
    name_mapper = {}

    for c in plot_df.columns:
        name_mapper[c] = reformat_feature_name_clustergram(c)

    plot_df = plot_df.rename(columns=name_mapper)

    sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                                 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    row_colors = df.Cluster.map(color_mapping)

    fig = sns.clustermap(plot_df, row_colors=row_colors.to_numpy(),
                         dendrogram_ratio=(.05, 0.00001), cbar_pos=None, yticklabels=False,
                         method="ward", col_cluster=True, figsize=(16, 8))
    fig.savefig(os.path.join(results_folder, filename + ".svg"))
    sns.set_context("notebook")
