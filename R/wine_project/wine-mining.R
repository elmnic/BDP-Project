library(tm)
library(SnowballC)
library(dplyr)
library(tidyverse)
library(slam)
library(stringr)
library(mltools)
library(data.table)

toSpace <- function(x, pattern) {
    return (gsub(pattern, " ", x))
}
toBlank <- function(x, pattern) {
    return (gsub(pattern, "", x))
}

# Needed for preprocessing dataset #
file <- "./winemag-data-130k-v2.csv"
csv <- read.csv(file, header = TRUE)

# Needed for machine learning #
keepCols <- c("doc_id", "text", "points", "taster_name")
dataML <- csv[keepCols]

# Remove empty taster
dataML <- dataML[!(dataML$taster_name == ""),]
# dataML 

# Clean and process text
i <- sapply(dataML, is.factor)
dataML[i] <- lapply(dataML[i], as.character)
dataML$text <- lapply(dataML$text, tolower)
dataML$text <- lapply(dataML$text, toSpace, "-")
dataML$text <- lapply(dataML$text, toSpace, "–")
dataML$text <- lapply(dataML$text, toSpace, "%")
dataML$text <- lapply(dataML$text, toSpace, "“")
dataML$text <- lapply(dataML$text, toSpace, "”")
dataML$text <- lapply(dataML$text, toSpace, "\\d")
dataML$text <- lapply(dataML$text, toBlank, "[[:punct:]]")
dataML$text <- lapply(dataML$text, removeWords, stopwords("english"))
dataML$text <- lapply(dataML$text, toBlank, "^\\s+")
dataML$text <- lapply(dataML$text, stemDocument)
dataML$text <- as.character(dataML$text)

# Save for later
textData <- dataML$text
dataML$text <- textData

# Create DT-matrix
word_corpus <- Corpus(VectorSource(dataML$text))
dtm <- DocumentTermMatrix(word_corpus)
inspect(dtm[1:4, 1:5])

# Sum all terms and retrieve top 100 most used
agg_dtm <- col_sums(dtm) %>%
    sort(decreasing = T)
agg_dtm <- agg_dtm[1:100]
agg_dtm[1:10]
keepWords <- names(agg_dtm)
head(agg_dtm, 5)

# Filter the top 100 terms from the 'text' column
dataML$text <- lapply(dataML$text, FUN = function(x) {
    lapply(strsplit(x, "\\s"), FUN = function(y) {
        paste(intersect(y, keepWords), collapse=" ")
        })
    })

# Flatten text into one string
flattenedText <- unlist(dataML$text) %>%
    strsplit(split=" ") %>%
    unlist()

# Quick test
'buttercream' %in% textDTM$dimnames$Terms
'wine' %in% textDTM$dimnames$Terms

# Unlist strings and remove rows with empty string
dataML$text <- unlist(dataML$text)
dataML <- dataML[dataML$text != " ",]


# Attempt to do One Hot Encoding on the 100 terms
# oneHot <- one_hot(dataML$text, cols=keepWords, sparsifyNAs = T)
# oneHot <- lapply(dataML$text, strsplit, " ") %>%
#             keras::pad_sequences(maxlen = 100, value="") %>%
#             mltools:one_hot(cols=keepWords, sparsifyNAs=T)
# oneHot <- lapply(dataML$text, kerasR::one_hot, 100)
# oneHot <- map(dataML$text, one_hot(n=100))
# tbl <- dataML$text %>% 
#         strsplit(split = " ")
# tbl <- as.data.table(tbl)
# ohe_tbl <- mltools::one_hot(tbl, cols = 'auto', sparsifyNAs = T)
# for (i in 1:nrow(dataML)) {
#     dataML$text[i] <- mltools::one_hot(dataML$text[i], cols=keepWords, sparsifyNAs = T)
# }

# Write to csv file
#write_csv(textFrame, "ohe_terms_100.csv")
write.csv(dataML, "wine_ml.csv", row.names=F)
#write.csv(keepWords, "keepWords.csv", row.names=F, col.names=F)
