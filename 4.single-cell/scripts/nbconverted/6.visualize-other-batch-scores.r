suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))

scores <- list()

batch <- "2019_11_20_Batch6"
plate <- "217762"

models <- c("real", "shuffled")

for (model in models) {
    file <- file.path(
        "scores", paste0(batch, "_", plate, "_", model, "_model_othersinglecells.tsv.gz")
    )
    scores[[model]] <- readr::read_tsv(file, col_types = readr::cols())
}


real_df <- scores[["real"]] %>%
    reshape2::melt(
        measure.vars = c("WT parental", "Clone A", "Clone E"),
        variable.name = "model",
        value.name = "probability"
    )

shuffle_df <- scores[["shuffled"]] %>%
    reshape2::melt(
        measure.vars = c("WT parental", "Clone A", "Clone E"),
        variable.name = "model",
        value.name = "probability"
    )

df <- dplyr::bind_rows(real_df, shuffle_df)

df$Metadata_treatment <- factor(df$Metadata_treatment, levels = c("DMSO", "bortezomib"))

head(df, 3)

append_model <- function(string) paste0("Model: ", string)

ggplot(df, aes(x = factor(Metadata_treatment), y = probability, fill = shuffled)) +
    geom_boxplot(outlier.size = 0.2) +
    facet_grid("model~Metadata_clone_number",
               labeller = labeller(model = as_labeller(append_model))) +
    xlab("Treatment") +
    ylab("Model probability") +
    ggtitle(paste0("Batch: ", batch, "\nPlate: ", plate)) +
    scale_fill_manual(name = "Data",
                      labels = c("True" = "Shuffled", "False" = "Real"),
                      values = c("True" = "#D96A2F", "False" = "#6ABEBE")) +
    theme_bw() +
    theme(
        axis.text.x = element_text(angle = 90),
        strip.background = element_rect(colour = "black", fill = "#fdfff4")
    ) +
    geom_hline(yintercept = 0.33, linetype = "dashed", color = "black")

output_fig <- file.path(
    "figures", "predictions", paste0(batch, "_", plate, "_predictions_boxplot.png")
    )
ggsave(output_fig, dpi = 500, height = 6, width = 9.5)
