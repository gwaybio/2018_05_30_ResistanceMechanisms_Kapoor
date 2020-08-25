#!/usr/bin/env python
# coding: utf-8

# ## Process and apply models to single cell profiles from other data batches
# 
# We will apply classifiers to these data to prioritize samples that we predict to have specific drug-tolerance mechanisms.

# In[1]:


import sys
import joblib
import pathlib
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pycytominer.cyto_utils import infer_cp_features

from utils.ml_utils import model_apply
from utils.single_cell_utils import process_sites, normalize_sc

sys.path.append("../0.generate-profiles")
from scripts.profile_util import load_config


# In[2]:


np.random.seed(1234)


# In[3]:


# Set constants
batch = "2019_03_20_Batch2"
plate = "207106_exposure320"

feature_filter = ["Object", "Location", "Count", "Parent"]
scaler_method = "standard"
seed = 123
n_sample_sites_per_well = 3


# In[4]:


# Load locations of single cell files
config = pathlib.Path("../0.generate-profiles/profile_config.yaml")
pipeline, single_cell_files = load_config(config, append_sql_prefix=False, local=False)


# In[5]:


# Load models
model_file = pathlib.Path("models", "multiclass_cloneAE_wildtype.joblib")
top_model = joblib.load(model_file)

shuffle_model_file = pathlib.Path("models", "multiclass_cloneAE_wildtype_shuffled.joblib")
top_shuffle_model = joblib.load(shuffle_model_file)


# In[6]:


# Load platemap and metadata
workspace_dir = pipeline["workspace_dir"]
batch_dir = pathlib.Path(workspace_dir, "backend", batch)
metadata_dir = pathlib.Path("../0.generate-profiles", "metadata", batch)

barcode_plate_map_file = pathlib.Path(metadata_dir, "barcode_platemap.csv")
barcode_plate_map_df = pd.read_csv(barcode_plate_map_file)

plate_map_name = (
    barcode_plate_map_df
    .query("Assay_Plate_Barcode == @plate")
    .Plate_Map_Name
    .values[0]
)

plate_map_file = pathlib.Path(metadata_dir, "platemap", f"{plate_map_name}.txt")
plate_map_df = pd.read_csv(plate_map_file, sep="\t")
plate_map_df.columns = [x if x.startswith("Metadata_") else f"Metadata_{x}" for x in plate_map_df.columns]
plate_map_df.head()


# ## Load single cell data

# In[7]:


plate_column = pipeline["aggregate"]["plate_column"]
well_column = pipeline["aggregate"]["well_column"]


# In[8]:


# Establish connection to sqlite file
single_cell_sqlite = single_cell_files[batch]["plates"][str(plate)]
conn = sqlite3.connect(single_cell_sqlite)


# In[9]:


image_cols = f"TableNumber, ImageNumber, {plate_column}, {well_column}"
image_query = f"select {image_cols} from image"
image_df = (
    pd.read_sql_query(image_query, conn)
    .merge(
        plate_map_df,
        left_on=well_column,
        right_on="Metadata_well_position"
    )
    .drop(["Metadata_well_position"], axis="columns")
)

print(image_df.shape)
image_df.head()


# In[10]:


# Assert that image number is unique
assert len(image_df.ImageNumber.unique()) == image_df.shape[0]


# In[11]:


# Randomly sample three sites per well to reduce number of single cells to store
sampled_image_df = image_df.groupby("Metadata_Well").apply(pd.DataFrame.sample, n=n_sample_sites_per_well)

sampled_image_df.head()


# In[12]:


get_ipython().run_cell_magic('time', '', 'sc_df = process_sites(\n    connection=conn,\n    imagenumbers=sampled_image_df.ImageNumber.tolist(),\n    image_df=image_df,\n    feature_filter=feature_filter,\n    seed=seed,\n    scaler_method=scaler_method,\n    normalize=True\n)')


# In[13]:


print(sc_df.shape)
sc_df.head()


# In[14]:


# Load test set data and reindex to match feature list
test_file = pathlib.Path("data", "single_cell_test.tsv.gz")
test_df = pd.read_csv(test_file, sep="\t")

cp_feature_order = infer_cp_features(test_df)

print(test_df.shape)
test_df.head()


# In[15]:


coef_file = pathlib.Path("coefficients/single_cell_multiclass_coefficients.tsv")
coef_df = pd.read_csv(coef_file, sep="\t")

print(coef_df.shape)
coef_df.head()


# In[16]:


# Assert the feature order and the model are equivalent
assert cp_feature_order == coef_df.feature.tolist()


# In[17]:


# Reindex features in the proper order before saving
meta_features = infer_cp_features(sc_df, metadata=True)
reindex_features = meta_features + cp_feature_order
sc_reindexed_df = sc_df.reindex(reindex_features, axis="columns")

print(sc_reindexed_df.shape)
sc_reindexed_df.head()


# In[18]:


# Output file
sc_output_file = pathlib.Path(f"data/single_cell_{batch}_plate_{plate}_random_cells.tsv.gz")
sc_reindexed_df.to_csv(sc_output_file, sep="\t", compression="gzip", index=False)


# ## Apply Models

# In[19]:


y_recode = {"WT parental": 0, "Clone A": 1, "Clone E": 2}
y_recode_reverse = {y: x for x, y in y_recode.items()}


# In[20]:


sc_df = sc_reindexed_df.reindex(cp_feature_order, axis="columns")
meta_df = sc_reindexed_df.reindex(meta_features, axis="columns")


# In[21]:


real_scores_df = model_apply(
    model=top_model,
    x_df=sc_df.fillna(0),
    meta_df=meta_df,
    y_recode=y_recode_reverse,
    data_fit="other_batch",
    shuffled=False,
    predict_proba=False
)

output_file = pathlib.Path(f"scores/{batch}_{plate}_othersinglecells.tsv.gz")
real_scores_df.to_csv(output_file, sep="\t", compression="gzip", index=False)

print(real_scores_df.shape)
real_scores_df.head()


# In[22]:


shuffled_scores_df = model_apply(
    model=top_shuffle_model,
    x_df=sc_df.fillna(0),
    meta_df=meta_df,
    y_recode=y_recode_reverse,
    data_fit="other_batch",
    shuffled=True,
    predict_proba=True
)

output_file = pathlib.Path(f"scores/{batch}_{plate}_shuffled_model_othersinglecells.tsv.gz")
shuffled_scores_df.to_csv(output_file, sep="\t", compression="gzip", index=False)

print(shuffled_scores_df.shape)
shuffled_scores_df.head()

