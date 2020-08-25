import umap
import pandas as pd
import plotnine as gg


def apply_umap(x_df, meta_df):
    reducer = umap.UMAP(random_state=123)
    embedding_df = reducer.fit_transform(x_df)

    # Setup plotting logic
    embedding_df = pd.DataFrame(embedding_df, columns=["x", "y"])
    embedding_df = embedding_df.merge(meta_df, left_index=True, right_index=True)

    return embedding_df


def plot_umap_cell_line(
    embedding_df, fig_file, cell_line_column, color_labels, color_values
):
    cell_line_gg = (
        gg.ggplot(embedding_df, gg.aes(x="x", y="y"))
        + gg.geom_point(gg.aes(color=cell_line_column), size=0.2, shape=".", alpha=0.2)
        + gg.theme_bw()
        + gg.scale_color_manual(
            name="Cell Line", labels=color_labels, values=color_values
        )
    )

    cell_line_gg.save(filename=fig_file, height=4, width=5, dpi=500)
    return cell_line_gg


def plot_umap_well(embedding_df, fig_file, well_column):
    well_gg = (
        gg.ggplot(embedding_df, gg.aes(x="x", y="y"))
        + gg.geom_point(gg.aes(color=well_column), size=0.2, shape=".", alpha=0.2)
        + gg.theme_bw()
    )

    well_gg.save(filename=fig_file, height=4, width=5, dpi=500)
    return well_gg
