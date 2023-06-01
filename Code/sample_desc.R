library(apaTables)
library(tidyverse)
library(readr)

# Get dataset
data <- read_delim("Projects/Virtual Visit/virtualvisit/Data/scores_summary.csv", delim = ";", escape_double = FALSE, locale = locale(decimal_mark = ","), trim_ws = TRUE)

# Demographics
length(data$age)
mean(data$age)
sd(data$age)
range(data$age)

data %>%
  count(gender) %>%
  mutate(freq = n/sum(n)*100)

# Correlation Table
subset_cor <- subset(data, select = c("ASI3", "SPAI", "SIAS", "AQ-K", "SSQ", "IPQ", "MPS"))
apa.cor.table(subset_cor, filename = "test.doc", show.sig.stars=TRUE)
