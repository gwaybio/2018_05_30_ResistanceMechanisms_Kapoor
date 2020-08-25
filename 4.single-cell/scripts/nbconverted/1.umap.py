#!/usr/bin/env python
# coding: utf-8

# In[1]:


import umap
import pathlib
import warnings
import numpy as np
import pandas as pd
import plotnine as gg

from utils.data_utils import load_data
from utils.umap_utils import apply_umap, plot_umap_cell_line, plot_umap_well


# In[2]:


np.random.seed(123)


# ## Load Data

# In[3]:


data_dict = load_data(
    return_meta=True,
    shuffle_row_order=True,
    holdout=True,
    othertreatment=True
)

print(data_dict["train"]["x"].shape)
print(data_dict["test"]["x"].shape)
print(data_dict["holdout"]["x"].shape)
print(data_dict["othertreatment"]["x"].shape)

data_dict["test"]["x"].head(3)


# ## Apply and visualize UMAP

# In[4]:


embedding_dict = {}

for data_fit in ["train", "test", "holdout", "othertreatment"]:
    embedding_dict[data_fit] = apply_umap(
        data_dict[data_fit]["x"], data_dict[data_fit]["meta"]
    )


# In[5]:


cell_line_column = "Metadata_clone_number"
well_column = "Metadata_Well"

cell_line_labels = {"Clone A": "Clone A", "Clone E": "Clone E", "WT parental": "WT parental"}
cell_line_colors = {"Clone A": "#1b9e77", "Clone E": "#d95f02", "WT parental": "#7570b3"}


# In[6]:


for data_fit, embedding_df in embedding_dict.items():
    fig_file = pathlib.Path("figures", "umap", f"single_cell_umap_{data_fit}.png")
    
    cell_gg = plot_umap_cell_line(
        embedding_df,
        fig_file,
        cell_line_column,
        cell_line_labels,
        cell_line_colors
    )
    
    print(cell_gg)
    
    fig_file = pathlib.Path("figures", "umap", f"single_cell_umap_well_{data_fit}.png")
    well_gg = plot_umap_well(embedding_df, fig_file, well_column)
    print(well_gg)


# In[7]:


treatment_gg = (
    gg.ggplot(embedding_dict["othertreatment"], gg.aes(x="x", y="y")) +
    gg.geom_point(gg.aes(color="Metadata_treatment"), size = 0.2, shape = ".", alpha = 0.2) +
    gg.theme_bw() 
)

fig_file = pathlib.Path("figures", "umap", "single_cell_othertreatment.png")
treatment_gg.save(filename=fig_file, height=4, width=5, dpi=500)

treatment_gg

