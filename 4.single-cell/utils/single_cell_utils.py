import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pycytominer.cyto_utils import infer_cp_features


def load_compartment_site(compartment, connection, imagenumber):
    query = f"select * from {compartment} where ImageNumber = {imagenumber}"
    df = pd.read_sql_query(query, connection)
    return df


def prefilter_features(df, flags):
    remove_cols = []
    for filter_feature in flags:
        remove_cols += [x for x in df.columns if filter_feature in x]
    remove_cols = list(set(remove_cols))
    return remove_cols


def filter_cells(df, drop_prop=0.1):
    missing_count = df.isna().sum(axis=1)
    missing_prop = missing_count / df.shape[1]
    drop_cells = missing_prop > drop_prop
    drop_cells = drop_cells[drop_cells].index.tolist()
    return drop_cells


def normalize_sc(sc_df, scaler_method="standard"):
    sc_df = sc_df.reset_index(drop=True)
    cp_features = infer_cp_features(sc_df)
    meta_df = sc_df.drop(cp_features, axis="columns")
    meta_df.columns = [
        x if x.startswith("Metadata_") else f"Metadata_{x}" for x in meta_df.columns
    ]
    sc_df = sc_df.loc[:, cp_features]

    if scaler_method == "standard":
        scaler = StandardScaler()

    sc_df = pd.DataFrame(
        scaler.fit_transform(sc_df), index=sc_df.index, columns=sc_df.columns
    )
    sc_df = meta_df.merge(sc_df, left_index=True, right_index=True)
    return sc_df


def process_data(
    connection, imagenumber, image_df, feature_filter, random_sample="all", seed=123,
):
    # Load compartments
    cell_df = load_compartment_site("cells", connection, imagenumber)
    if random_sample != "all":
        cell_df = cell_df.sample(frac=random_sample, axis="index")

    cyto_df = load_compartment_site("cytoplasm", connection, imagenumber)
    nuc_df = load_compartment_site("nuclei", connection, imagenumber)

    # Merge tables
    merged_df = cell_df.merge(
        cyto_df,
        left_on=["TableNumber", "ImageNumber", "ObjectNumber"],
        right_on=["TableNumber", "ImageNumber", "Cytoplasm_Parent_Cells"],
        how="inner",
    ).merge(
        nuc_df,
        left_on=["TableNumber", "ImageNumber", "Cytoplasm_Parent_Nuclei"],
        right_on=["TableNumber", "ImageNumber", "ObjectNumber"],
        how="inner",
    )

    # Filter features
    drop_features = prefilter_features(merged_df, feature_filter)
    merged_df = merged_df.drop(drop_features, axis="columns")

    # Merge with the image information
    merged_df = image_df.merge(
        merged_df, on=["TableNumber", "ImageNumber"], how="right"
    )

    return merged_df


def process_sites(
    connection,
    imagenumbers,
    image_df,
    feature_filter,
    seed=123,
    scaler_method="standard",
    random_sample="all",
    normalize=True,
):
    data_df = {}
    for imagenumber in imagenumbers:
        data_df[imagenumber] = process_data(
            connection=connection,
            imagenumber=imagenumber,
            image_df=image_df,
            feature_filter=feature_filter,
            random_sample=random_sample,
            seed=seed,
        )

    data_df = pd.concat(data_df).reset_index(drop=True)

    if normalize:
        data_df = normalize_sc(data_df, scaler_method=scaler_method)

    return data_df
