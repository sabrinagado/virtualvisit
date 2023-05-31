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


# Filter dataset
# Filter dataset
data_filt <- data %>%
  filter(exclude_walk != 1)  %>% # exclude participants who were suspicious about ET and reported it influenced their behavior
  filter(exclude_walk != 2)  %>% # exclude participants who reported the ET influenced their behavior
  filter(exclude_lab != 1)  %>% # exclude participants who were suspicious about the confederate and reported it influenced their behavior
  filter(exclude_lab != 2)  %>% # exclude participants who reported the confederate influenced their behavior
  filter(exclude_lab != 4) # exclude participants who knew the confederate

# Demographics
length(data_filt$age)
mean(data_filt$age)
sd(data_filt$age)
range(data_filt$age)

data_filt %>%
  count(gender) %>%
  mutate(freq = n/sum(n)*100)

# Correlation Table
subset_cor <- subset(data_filt, select = c("SAD_screening", "SPAI", "SIAS", "AQ-K", "VAS_start_anxiety", "VAS_start_nervous", "VAS_start_distress", "VAS_start_stress"))
apa.cor.table(subset_cor, filename = "test.doc", show.sig.stars=TRUE)
