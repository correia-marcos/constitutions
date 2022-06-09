###############################################################################
# Structural Topic Model
# Creator: Marcos Paulo R. Correia
# Researchers: Eric Alstom and Marcos Paulo
# In this (only) R script we create the STM approach in order to measure the
# amount of change between constitutions.
# See: http://www.structuraltopicmodel.com/
###############################################################################

# Load the required packages
library(dplyr)
library(tidytext)
library(tidyverse)
library(stm)
library(quanteda) 
library(tm)
library(MASS)

# Reading df and removing/changing columns. Remember that the document
# column is based on already heavly preprocessed textual data
data <- read.csv("results/csv/constitutions_for_stm.csv")
data$X <- NULL
data$codes <- NULL
data$length_preprocess <- NULL 
names(data)[1] <- 'country'

# Processing for necessary format
processed <- textProcessor(data$document, metadata = data,
                           removestopwords = FALSE, removenumbers = FALSE,
                           stem = FALSE, removepunctuation = FALSE)

# Preprocessed documents (like doc2bow)
out <- prepDocuments(processed$documents, processed$vocab,
                     processed$meta)

# Necessary and just copy and poste
docs <- out$documents
vocab <- out$voca
meta <- out$meta

# STM model
stm_constitutios <- stm(documents = out$documents, vocab = out$vocab,
                        K = 50, max.em.its = 80,
                        data = out$meta, init.type = "Spectral",
                        verbose = TRUE, seed=42)


# Just a small check
td_beta <- tidy(stm_constitutios)

# Taking Dataframe
results <- stm_constitutios[["theta"]]
write.matrix(results, file="stm(50topics).csv")
             