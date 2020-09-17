import sys
import pathlib
import sqlite3
import pandas as pd

from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features, output

from utils.single_cell_utils import process_sites

sys.path.append("../0.generate-profiles")
from scripts.profile_util import load_config

# Determine which plates to process
process_which_plates = {
    "2019_11_22_Batch7": {
        "plates": ["217766", "217768"],
        "cell_line_column": "Metadata_clone_number",
    },
    "2020_07_02_Batch8": {
        "plates": ["218360", "218361", "218362", "218363"],
        "cell_line_column": "Metadata_clone_number",
    },
}

# Set constants
output_base_dir = pathlib.Path("data/profiles/")
feature_filter = ["Object", "Location", "Count", "Parent"]
scaler_method = "standard"
normalize_single_cells = True
seed = 123

feature_select_opts = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blacklist",
    "drop_outliers",
]
corr_threshold = 0.8
na_cutoff = 0

# Load locations of single cell files
config = pathlib.Path("../0.generate-profiles/profile_config.yaml")
pipeline, single_cell_files = load_config(config, append_sql_prefix=False)

for batch in process_which_plates:
    cell_line_column = process_which_plates[batch]["cell_line_column"]
    for plate in process_which_plates[batch]["plates"]:
        print(f"Now processing... Batch: {batch}, Plate: {plate}")
        # Generate output files
        output_dir = pathlib.Path(f"{output_dir}/{batch}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = pathlib.Path(
            f"{output_dir}/{plate}_singlecell_normalized_feature_select.csv.gz"
        )

        # Load platemap
        workspace_dir = pipeline["workspace_dir"]
        batch_dir = pathlib.Path(workspace_dir, "backend", batch)
        metadata_dir = pathlib.Path("../0.generate-profiles", "metadata", batch)

        barcode_plate_map_file = pathlib.Path(metadata_dir, "barcode_platemap.csv")
        barcode_plate_map_df = pd.read_csv(barcode_plate_map_file)

        plate_map_name = barcode_plate_map_df.query(
            "Assay_Plate_Barcode == @plate"
        ).Plate_Map_Name.values[0]

        plate_map_file = pathlib.Path(metadata_dir, "platemap", f"{plate_map_name}.txt")
        plate_map_df = pd.read_csv(plate_map_file, sep="\t")
        plate_map_df.columns = [
            x if x.startswith("Metadata_") else f"Metadata_{x}"
            for x in plate_map_df.columns
        ]

        plate_column = pipeline["aggregate"]["plate_column"]
        well_column = pipeline["aggregate"]["well_column"]

        # Establish connection to sqlite file
        single_cell_sqlite = single_cell_files[batch]["plates"][plate]
        conn = sqlite3.connect(single_cell_sqlite)

        # Load Image details
        image_cols = f"TableNumber, ImageNumber, {plate_column}, {well_column}"
        image_query = f"select {image_cols} from image"
        image_df = (
            pd.read_sql_query(image_query, conn)
            .merge(plate_map_df, left_on=well_column, right_on="Metadata_well_position")
            .drop(["Metadata_well_position"], axis="columns")
        )

        all_image_numbers = image_df.ImageNumber.unique()
        assert len(all_image_numbers) == image_df.shape[0]

        # Load all single cell profiles
        sc_df = process_sites(
            connection=conn,
            imagenumbers=all_image_numbers,
            image_df=image_df,
            feature_filter=feature_filter,
            seed=seed,
            scaler_method=scaler_method,
            normalize=normalize_single_cells,
        )

        # Apply feature selection
        sc_df = feature_select(
            sc_df,
            operation=feature_select_opts,
            na_cutoff=na_cutoff,
            corr_threshold=corr_threshold,
        )

        # Output file
        sc_df.to_csv(output_file, compression="gzip", sep=",", index=False)
        print("Done.")
