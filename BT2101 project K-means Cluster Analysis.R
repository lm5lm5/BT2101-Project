#BT2101 project K-means Cluster Analysis
data = USArrests
data = na.omit(data)

library(tidyverse)
library(cluster)    # clustering algorithms
library(factoextra)
library('rpart')
library('rpart.plot')
library(mlbench)
library(ggplot2)

df = as_tibble(read.csv('heart.csv')) 
colnames(df)[1] <- "age"

## cluster analysis using all variables
set.seed(456)
k2_all <- kmeans(df[1:13], centers = 2, nstart = 50)
k2_all$cluster[k2_all$cluster == 2] = 0
table1 = confusionMatrix(data = factor(k2_all$cluster),factor(df$target), positive = '1') ##accuracy is about 57.8%
fourfoldplot(table1$table)
str(k2_all)
fviz_cluster(k2_all, data = df[1:13])

## cluster analysis using selected variables: sex, cp, exang, oldpeak, ca
set.seed(456)
df_select = df[,c(2,3,9,10,12)]
k2 <- kmeans(df_select, centers = 2, nstart = 50)
k2$cluster[k2$cluster == 1] = 0
k2$cluster[k2$cluster == 2] = 1
table2 = confusionMatrix(data = factor(k2$cluster), factor(df$target), positive = '1') ##accuracy is about 79.5%, with high sensitivity
fourfoldplot(table2$table)
str(k2)
fviz_cluster(k2, data = df_select)

df_select %>%
  mutate(cluster = k2$cluster, No = row.names(df)) %>%
  ggplot(aes(cp, ca, color = factor(cluster), label = No)) +
  geom_text()
