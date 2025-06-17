import itertools
import os
import glob
import copairs.compute_np as backend
import h5py
import numpy as np
import pandas as pd
from copairs.compute import cosine_indexed
from copairs.map import (
    aggregate,
    build_rank_list_multi,
    build_rank_lists,
    create_matcher,
    results_to_dframe,
)
from copairs.matching import dict_to_dframe
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity



def load_data(plate, filetype):
    """load all data from a single experiment into a single dataframe"""
    path = os.path.join("/gscratch/aims/datasets/cellpainting/jumpcp/profiles", f"{plate}", f"*_{filetype}")
    files = glob.glob(path)
    df = pd.concat(pd.read_csv(_, low_memory=False) for _ in files)
    return df



# def load_data(plate):
#     """load all data from a single experiment into a single dataframe"""
#     path = os.path.join(
#         "/gscratch/aims/datasets/cellpainting/jumpcp/output_emb/new_cell_clip",
#         f"{plate}.h5",
#     )
#     with h5py.File(path, "r") as img_file:
#         # Extract the datasets
#         embeddings = img_file["embeddings"][:]
#         well_position = img_file["well_position"][:]
#         broad_sample = img_file["broad_sample"][:]

#         embeddings_flat = embeddings.reshape(-1, 512)

#         # Create a DataFrame and store embeddings as a single column with numpy arrays
#         df = pd.DataFrame(
#             {
#                 "embeddings": [np.array(embedding) for embedding in embeddings_flat],
#                 "well_position": well_position.astype(str),
#                 "Metadata_broad_sample": broad_sample.astype(str),
#             }
#         )
#         df["Metadata_Plate"] = plate

#     target1_metadata = pd.read_csv(
#         "/gscratch/aims/mingyulu/cell_painting/label_data/jumpcp/JUMP-Target-1_compound_metadata_additional_annotations.tsv",
#         sep="\t",
#     )
#     target1_metadata = target1_metadata.rename(
#         columns={col: f"Metadata_{col}" for col in target1_metadata.columns}
#     )

#     df = pd.merge(
#         target1_metadata,
#         df,
#         left_on="Metadata_broad_sample",
#         right_on="Metadata_broad_sample",
#         how="right",
#     )
#     return df

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]


def remove_negcon_and_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=["Metadata_broad_sample"])
        .reset_index(drop=True)
    )
    return df


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = df.dropna(subset=["Metadata_broad_sample"]).reset_index(drop=True)
    return df


def concat_profiles(df1, df2):
    """Concatenate dataframes"""
    if df1.shape[0] == 0:
        df1 = df2.copy()
    else:
        frames = [df1, df2]
        df1 = pd.concat(frames, ignore_index=True, join="inner")

    return df1


def create_replicability_df(
    replicability_map_df,
    replicability_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality,
    cell,
    timepoint,
):
    _replicability_map_df = replicability_map_df
    _replicability_fr_df = replicability_fr_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_replicability_fr_df)],
    )
    _replicability_fr_df = concat_profiles(_replicability_fr_df, _fr_df)

    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _replicability_map_df = concat_profiles(_replicability_map_df, _map_df)

    _replicability_fr_df["fr"] = _replicability_fr_df["fr"].astype(float)
    _replicability_map_df["mean_average_precision"] = _replicability_map_df[
        "mean_average_precision"
    ].astype(float)

    return _replicability_map_df, _replicability_fr_df


def create_matching_df(
    matching_map_df,
    matching_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality,
    cell,
    timepoint,
):
    _matching_map_df = matching_map_df
    _matching_fr_df = matching_fr_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_matching_fr_df)],
    )
    _matching_fr_df = concat_profiles(_matching_fr_df, _fr_df)

    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _matching_map_df = concat_profiles(_matching_map_df, _map_df)

    _matching_fr_df["fr"] = _matching_fr_df["fr"].astype(float)
    _matching_map_df["mean_average_precision"] = _matching_map_df[
        "mean_average_precision"
    ].astype(float)

    return _matching_map_df, _matching_fr_df


def create_gene_compound_matching_df(
    gene_compound_matching_map_df,
    gene_compound_matching_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality_1,
    modality_2,
    cell,
    timepoint1,
    timepoint2,
):
    _gene_compound_matching_map_df = gene_compound_matching_map_df
    _gene_compound_matching_fr_df = gene_compound_matching_fr_df

    _modality_1 = modality_1
    _modality_2 = modality_2
    _cell = cell
    _timepoint_1 = timepoint1
    _timepoint_2 = timepoint2
    _time_1 = time_point(_modality_1, _timepoint_1)
    _time_2 = time_point(_modality_2, _timepoint_2)

    _description = f"{_modality_1}_{cell}_{_time_1}-{_modality_2}_{cell}_{_time_2}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality1": f"{_modality_1}_{_time_1}",
            "Modality2": f"{_modality_2}_{_time_2}",
            "Cell": _cell,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_gene_compound_matching_fr_df)],
    )
    _gene_compound_matching_fr_df = concat_profiles(_gene_compound_matching_fr_df, _fr_df)

    _map_df["Description"] = f"{_description}"
    _map_df["Modality1"] = f"{_modality_1}_{_time_1}"
    _map_df["Modality2"] = f"{_modality_2}_{_time_2}"
    _map_df["Cell"] = f"{_cell}"
    _gene_compound_matching_map_df = concat_profiles(
        _gene_compound_matching_map_df, _map_df
    )

    _gene_compound_matching_fr_df["fr"] = _gene_compound_matching_fr_df["fr"].astype(float)
    _gene_compound_matching_map_df[
        "mean_average_precision"
    ] = _gene_compound_matching_map_df["mean_average_precision"].astype(float)

    return _gene_compound_matching_map_df, _gene_compound_matching_fr_df


def consensus(profiles_df, group_by_feature, feature_type):
    """
    Computes the median consensus profiles.
    Parameters:
    -----------
    profiles_df: pandas.DataFrame
        dataframe of profiles
    group_by_feature: str
        Name of the column
    Returns:
    -------
    pandas.DataFrame of the same shape as `plate`
    """

    metadata_df = get_metadata(profiles_df).drop_duplicates(subset=[group_by_feature])

    if feature_type == "emb":
        feature_cols = [group_by_feature, "embeddings"]
        profiles_df = (
            profiles_df[feature_cols]
            .groupby([group_by_feature])
            .apply(lambda x: pd.Series({
                "embeddings": np.median(np.stack(x["embeddings"].values), axis=0)
            }))
            .reset_index()
        )

    else:
        feature_cols = [group_by_feature] + get_featurecols(profiles_df)
        profiles_df = (
            profiles_df[feature_cols].groupby([group_by_feature]).median().reset_index()
        )

    profiles_df = metadata_df.merge(profiles_df, on=group_by_feature)

    return profiles_df

    def cleanup(self):
        """
        Remove rows and columns that are all NaN
        """
        keep = list((self.truth_matrix.sum(axis=1) > 0))
        self.corr["keep"] = keep
        self.map1["keep"] = keep
        self.truth_matrix["keep"] = keep

        self.corr = self.corr.loc[self.corr.keep].drop(columns=["keep"])
        self.map1 = self.map1.loc[self.map1.keep].drop(columns=["keep"])
        self.truth_matrix = self.truth_matrix.loc[self.truth_matrix.keep].drop(
            columns=["keep"]
        )


def time_point(modality, time_point):
    """
    Convert time point in hr to long or short time description
    Parameters:
    -----------
    modality: str
        perturbation modality
    time_point: int
        time point in hr
    Returns:
    -------
    str of time description
    """
    if modality == "compound":
        if time_point == 24:
            time = "short"
        else:
            time = "long"
    elif modality == "orf":
        if time_point == 48:
            time = "short"
        else:
            time = "long"
    else:
        if time_point == 96:
            time = "short"
        else:
            time = "long"

    return time


def convert_pvalue(pvalue):
    """
    Convert p value format
    Parameters:
    -----------
    pvalue: float
        p value
    Returns:
    -------
    str of p value
    """
    if pvalue < 0.05:
        pvalue = "<0.05"
    else:
        pvalue = f"{pvalue:.2f}"
    return pvalue


def add_lines_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add lines to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """
    y_value = ""
    if percentile == 5:
        y_value = "fifth_percentile"
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
    fig.add_shape(
        type="line",
        x0=locations["line"][color_order.index(df_row[color_column])]["x0"],
        y0=df_row[y_value],
        x1=locations["line"][color_order.index(df_row[color_column])]["x1"],
        y1=df_row[y_value],
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
        row=row,
        col=col,
    )
    return fig


def add_text_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add text to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """

    y_value = ""
    y_percent_value = ""
    y_offset = 0
    if percentile == 5:
        y_value = "fifth_percentile"
        y_percent_value = "percent_fifth_percentile"
        y_offset = -0.08
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
        y_percent_value = "percent_ninetyfifth_percentile"
        y_offset = 0.08
    fig.add_annotation(
        x=locations["text"][color_order.index(df_row[color_column])]["x"],
        y=df_row[y_value] + y_offset,
        text=f"{df_row[y_percent_value]*100:.02f}%",
        showarrow=False,
        font=dict(
            size=16,
        ),
        row=row,
        col=col,
    )
    return fig


def calculate_mAP(result, pos_sameby, threshold):
    """
    Calculate the mean average precision
    Parameters
    ----------
    result : pandas.DataFrame of average precision values output by copairs
    pos_sameby : str of columns that define positives
    threshold : float of threshold for q-value
    Returns
    -------
    agg_result : pandas.DataFrame of mAP values grouped by pos_sameby columns
    """
    agg_result = aggregate(result, pos_sameby, threshold=0.05).rename(
        columns={"average_precision": "mean_average_precision"}
    )
    return agg_result


def calculate_fraction_retrieved(agg_result):
    """
    Calculate the fraction of labels retrieved
    Parameters
    ----------
    agg_result : pandas.DataFrame of mAP values
    Returns
    -------
    fraction_retrieved : float of fraction positive
    """
    fraction_retrieved = len(agg_result.query("above_q_threshold==True")) / len(agg_result)
    return fraction_retrieved


def compute_similarities(pairs, feats, batch_size, anti_match=False):
    dist_df = pairs[["ix1", "ix2"]].drop_duplicates().copy()
    dist_df["dist"] = cosine_indexed(feats, dist_df.values, batch_size)
    if anti_match:
        dist_df["dist"] = np.abs(dist_df["dist"])
    return pairs.merge(dist_df, on=["ix1", "ix2"])


def run_pipeline(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    null_size,
    anti_match=False,
    multilabel_col=None,
    batch_size=20000,
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    matcher = create_matcher(
        meta, pos_sameby, pos_diffby, neg_sameby, neg_diffby, multilabel_col
    )

    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = dict_to_dframe(dict_pairs, pos_sameby)
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = set(itertools.chain.from_iterable(dict_pairs.values()))
    neg_pairs = pd.DataFrame(neg_pairs, columns=["ix1", "ix2"])
    pos_pairs = compute_similarities(pos_pairs, feats, batch_size, anti_match)
    neg_pairs = compute_similarities(neg_pairs, feats, batch_size, anti_match)
    if multilabel_col and multilabel_col in pos_sameby:
        rel_k_list = build_rank_list_multi(pos_pairs, neg_pairs, multilabel_col)
    else:
        rel_k_list = build_rank_lists(pos_pairs, neg_pairs)
    ap_scores = rel_k_list.apply(backend.compute_ap)
    ap_scores = np.concatenate(ap_scores.values)
    null_dists = backend.compute_null_dists(rel_k_list, null_size)
    p_values = backend.compute_p_values(null_dists, ap_scores, null_size)
    result = results_to_dframe(meta, rel_k_list.index, p_values, ap_scores, multilabel_col)
    return result


class PrecisionScores(object):
    """
    Calculate the precision scores for information retrieval.
    """

    def __init__(
        self,
        profile1,
        profile2,
        group_by_feature,
        mode,
        identify_perturbation_feature,
        within=False,
        anti_correlation=False,
        against_negcon=False,
    ):
        """
        Parameters:
        -----------
        profile1: pandas.DataFrame
            dataframe of profiles
        profile2: pandas.DataFrame
            dataframe of profiles
        group_by_feature: str
            Name of the feature to group by
        mode: str
            Whether compute replicability or matching
        identity_perturbation_feature: str
            Name of the feature that identifies perturbations
        within: bool, default: False
            Whether profile1 and profile2 are the same dataframe.
        anti_correlation: bool, default: False
            Whether both anti-correlation and correlation are used in the calculation.
        against_negcon: bool, default:  False
            Whether to calculate precision scores with respect to negcon.
        """
        self.sample_id_feature = "Metadata_sample_id"
        self.control_type_feature = "Metadata_control_type"
        self.feature = group_by_feature
        self.mode = mode
        self.identify_perturbation_feature = identify_perturbation_feature
        self.within = within
        self.anti_correlation = anti_correlation
        self.against_negcon = against_negcon

        self.profile1 = self.process_profiles(profile1)
        self.profile2 = self.process_profiles(profile2)

        if self.mode == "replicability":
            self.map1 = self.profile1[
                [self.feature, self.sample_id_feature, self.control_type_feature]
            ].copy()
            self.map2 = self.profile2[
                [self.feature, self.sample_id_feature, self.control_type_feature]
            ].copy()
        elif self.mode == "matching":
            self.map1 = self.profile1[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.sample_id_feature,
                    self.control_type_feature,
                ]
            ].copy()
            self.map2 = self.profile2[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.sample_id_feature,
                    self.control_type_feature,
                ]
            ].copy()

        self.corr = self.compute_correlation()
        self.truth_matrix = self.create_truth_matrix()
        self.cleanup()

        self.ap = self.calculate_average_precision_per_sample()
        self.map = self.calculate_average_precision_score_per_group(self.ap)
        self.mmap = self.calculate_mean_average_precision_score(self.map)

    def process_profiles(self, _profile):
        """
        Add sample id column to profiles.
        Parameters:
        -----------
        _profile: pandas.DataFrame
            dataframe of profiles
        Returns:
        -------
        pandas.DataFrame which includes the sample id column
        """

        _metadata_df = pd.DataFrame()
        _profile = _profile.reset_index(drop=True)
        _feature_df = get_featuredata(_profile)
        if self.mode == "replicability":
            _metadata_df = _profile[[self.feature, self.control_type_feature]]
        elif self.mode == "matching":
            _metadata_df = _profile[
                [
                    self.identify_perturbation_feature,
                    self.feature,
                    self.control_type_feature,
                ]
            ]
        width = int(np.log10(len(_profile))) + 1
        _perturbation_id_df = pd.DataFrame(
            {
                self.sample_id_feature: [
                    f"sample_{i:0{width}}" for i in range(len(_metadata_df))
                ]
            }
        )
        _metadata_df = pd.concat([_metadata_df, _perturbation_id_df], axis=1)
        _profile = pd.concat([_metadata_df, _feature_df], axis=1)
        return _profile

    def compute_correlation(self):
        """
        Compute correlation.
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values.
        """

        _profile1 = get_featuredata(self.profile1)
        _profile2 = get_featuredata(self.profile2)
        _sample_names_1 = list(self.profile1[self.sample_id_feature])
        _sample_names_2 = list(self.profile2[self.sample_id_feature])
        _corr = cosine_similarity(_profile1, _profile2)
        if self.anti_correlation:
            _corr = np.abs(_corr)
        _corr_df = pd.DataFrame(_corr, columns=_sample_names_2, index=_sample_names_1)
        _corr_df = self.process_self_correlation(_corr_df)
        _corr_df = self.process_negcon(_corr_df)
        return _corr_df

    def create_truth_matrix(self):
        """
        Compute truth matrix.
        Returns:
        -------
        pandas.DataFrame of binary truth values.
        """

        _truth_matrix = self.corr.unstack().reset_index()
        _truth_matrix = _truth_matrix.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature, 0], axis=1)
        _truth_matrix = _truth_matrix.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _truth_matrix["value"] = [
            len(np.intersect1d(x[0].split("|"), x[1].split("|"))) > 0
            for x in zip(
                _truth_matrix[f"{self.feature}_x"], _truth_matrix[f"{self.feature}_y"]
            )
        ]
        if self.within and self.mode == "replicability":
            _truth_matrix["value"] = np.where(
                _truth_matrix["level_0"] == _truth_matrix["level_1"],
                0,
                _truth_matrix["value"],
            )
        elif self.within and self.mode == "matching":
            _truth_matrix["value"] = np.where(
                _truth_matrix[f"{self.identify_perturbation_feature}_x"]
                == _truth_matrix[f"{self.identify_perturbation_feature}_y"],
                0,
                _truth_matrix["value"],
            )

        _truth_matrix = (
            _truth_matrix.pivot("level_1", "level_0", "value")
            .reset_index()
            .set_index("level_1")
        )
        _truth_matrix.index.name = None
        _truth_matrix = _truth_matrix.rename_axis(None, axis=1)
        return _truth_matrix

    def calculate_average_precision_per_sample(self):
        """
        Compute average precision score per sample.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """
        _score = []
        for _sample in self.corr.index:
            _y_true, _y_pred = self.filter_nan(
                self.truth_matrix.loc[_sample].values, self.corr.loc[_sample].values
            )

            # compute corrected average precision
            random_baseline_ap = _y_true.sum() / len(_y_true)
            _score.append(average_precision_score(_y_true, _y_pred) - random_baseline_ap)

        _ap_sample_df = self.map1.copy()
        _ap_sample_df["ap"] = _score
        if self.against_negcon:
            _ap_sample_df = (
                _ap_sample_df.query(f'{self.control_type_feature}!="negcon"')
                .drop(columns=[self.control_type_feature])
                .reset_index(drop=True)
            )
        else:
            _ap_sample_df = _ap_sample_df.drop(
                columns=[self.control_type_feature]
            ).reset_index(drop=True)

        return _ap_sample_df

    def calculate_average_precision_score_per_group(self, precision_score):
        """
        Compute average precision score per sample group.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """

        _precision_group_df = (
            precision_score.groupby(self.feature)
            .apply(lambda x: np.mean(x))
            .reset_index()
            .rename(columns={"ap": "mAP"})
        )
        return _precision_group_df

    @staticmethod
    def calculate_mean_average_precision_score(precision_score):
        """
        Compute mean average precision score.
        Returns:
        -------
        mean average precision score.
        """

        return precision_score.mean().values[0]

    def process_negcon(self, _corr_df):
        """
        Keep or remove negcon
        Parameters:
        -----------
        _corr_df: pandas.DataFrame
            pairwise correlation dataframe
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values
        """
        _corr_df = _corr_df.unstack().reset_index()
        _corr_df["filter"] = 1
        _corr_df = _corr_df.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _corr_df = _corr_df.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)

        if self.against_negcon:
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.feature}_x"] != _corr_df[f"{self.feature}_y"],
                0,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_x"] == "negcon",
                1,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_y"] == "negcon",
                0,
                _corr_df["filter"],
            )
        else:
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_x"] == "negcon",
                0,
                _corr_df["filter"],
            )
            _corr_df["filter"] = np.where(
                _corr_df[f"{self.control_type_feature}_y"] == "negcon",
                0,
                _corr_df["filter"],
            )

        _corr_df = _corr_df.query("filter==1").reset_index(drop=True)

        if self.mode == "replicability":
            self.map1 = (
                _corr_df[["level_1", f"{self.feature}_y", f"{self.control_type_feature}_y"]]
                .copy()
                .rename(
                    columns={
                        "level_1": self.sample_id_feature,
                        f"{self.feature}_y": self.feature,
                        f"{self.control_type_feature}_y": self.control_type_feature,
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[["level_0", f"{self.feature}_x", f"{self.control_type_feature}_x"]]
                .copy()
                .rename(
                    columns={
                        "level_0": self.sample_id_feature,
                        f"{self.feature}_x": self.feature,
                        f"{self.control_type_feature}_x": self.control_type_feature,
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
        elif self.mode == "matching":
            self.map1 = (
                _corr_df[
                    [
                        "level_1",
                        f"{self.identify_perturbation_feature}_y",
                        f"{self.feature}_y",
                        f"{self.control_type_feature}_y",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "level_1": self.sample_id_feature,
                        f"{self.feature}_y": self.feature,
                        f"{self.control_type_feature}_y": self.control_type_feature,
                        f"{self.identify_perturbation_feature}_y": f"{self.identify_perturbation_feature}",
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[
                    [
                        "level_0",
                        f"{self.identify_perturbation_feature}_x",
                        f"{self.feature}_x",
                        f"{self.control_type_feature}_x",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "level_0": self.sample_id_feature,
                        f"{self.feature}_x": self.feature,
                        f"{self.control_type_feature}_x": self.control_type_feature,
                        f"{self.identify_perturbation_feature}_x": f"{self.identify_perturbation_feature}",
                    }
                )
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )

        _corr_df = (
            _corr_df.pivot("level_1", "level_0", 0).reset_index().set_index("level_1")
        )
        _corr_df.index.name = None
        _corr_df = _corr_df.rename_axis(None, axis=1)
        return _corr_df

    @staticmethod
    def filter_nan(_y_true, _y_pred):
        """
        Filter out nan values from y_true and y_pred
        Parameters:
        -----------
        _y_true: np.array of truth values
        _y_pred: np.array of predicted values
        Returns:
        --------
        _y_true: np.array of truth values
        _y_pred: np.array of predicted values
        """
        arg = np.argwhere(~np.isnan(_y_pred))
        return _y_true[arg].flatten(), _y_pred[arg].flatten()

    def process_self_correlation(self, corr):
        """
        Process self correlation values (correlation between the same profiles)
        Parameters:
        -----------
        corr: pd.DataFrame of correlation values
        Returns:
        --------
        _corr: pd.DataFrame of correlation values
        """
        _corr = corr.unstack().reset_index().rename(columns={0: "corr"})
        _corr = _corr.merge(
            self.map2, left_on="level_0", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        _corr = _corr.merge(
            self.map1, left_on="level_1", right_on=self.sample_id_feature, how="left"
        ).drop([self.sample_id_feature], axis=1)
        if self.within and self.mode == "replicability":
            _corr["corr"] = np.where(
                _corr["level_0"] == _corr["level_1"], np.nan, _corr["corr"]
            )
        elif self.within and self.mode == "matching":
            _corr["corr"] = np.where(
                _corr[f"{self.identify_perturbation_feature}_x"]
                == _corr[f"{self.identify_perturbation_feature}_y"],
                np.nan,
                _corr["corr"],
            )

        _corr = _corr.pivot("level_1", "level_0", "corr").reset_index().set_index("level_1")
        _corr.index.name = None
        _corr = _corr.rename_axis(None, axis=1)

        return _corr

    def cleanup(self):
        """
        Remove rows and columns that are all NaN
        """
        keep = list((self.truth_matrix.sum(axis=1) > 0))
        self.corr["keep"] = keep
        self.map1["keep"] = keep
        self.truth_matrix["keep"] = keep

        self.corr = self.corr.loc[self.corr.keep].drop(columns=["keep"])
        self.map1 = self.map1.loc[self.map1.keep].drop(columns=["keep"])
        self.truth_matrix = self.truth_matrix.loc[self.truth_matrix.keep].drop(
            columns=["keep"]
        )
