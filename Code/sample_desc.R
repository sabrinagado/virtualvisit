library(apaTables)
library(tidyverse)
library(readr)

# Get dataset
data <- read_delim("Projects/Virtual Visit/virtualvisit/Data/scores_summary.csv", delim = ";", escape_double = FALSE, locale = locale(decimal_mark = ","), trim_ws = TRUE)

# Correlation Table
subset_cor <- subset(data, select = c("ASI3", "SPAI", "SIAS", "AQ-K", "SSQ", "IPQ", "MPS"))
apa.cor.table(subset_cor, filename = "Projects/Virtual Visit/virtualvisit/Data/correlation_scores.doc", show.sig.stars=TRUE)
