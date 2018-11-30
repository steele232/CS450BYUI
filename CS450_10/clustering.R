library(datasets)
myData = state.x77

column.names = c("Population", "Income", "Illiteracy", 
                 "Life Exp", "Murder", "HS Grad", "Frost", "Area")

# ------------------------------------------------------------------------
# Agglomerative Hierachical Clustering
# ------------------------------------------------------------------------

# ------------------ 
# Agglomerative Hierachical Clustering
# ------------------ 
# first compute a distance matrix
distance <- dist(as.matrix(myData))

# now perform the clustering
hc <- hclust(distance)

# finally, plot the dendrogram
plot(hc)



# ------------------ 
# Agglomerative w/ Scaling
# ------------------ 
data.scaled <- scale(myData)

distance <- dist(as.matrix(data.scaled))
hc <- hclust(distance)
plot(hc)

# ------------------ 
# Agglomerative w/ Scaling && "Area" Removed
# ------------------ 
data.scaled.area.removed <- data.scaled[,1:7]

distance <- dist(as.matrix(data.scaled.area.removed))
hc <- hclust(distance)
plot(hc)



# ------------------ 
# Agglomerative w/ Scaling && ONLY "FROST"
# ------------------ 
data.scaled.area.removed <- data.scaled[,7]

distance <- dist(as.matrix(data.scaled.area.removed))
hc <- hclust(distance)
plot(hc)



# ------------------------------------------------------------------------
# K-Means Clustering
# ------------------------------------------------------------------------
library(cluster)
# ------------------ 
# K-Means (Just Getting Started)
# ------------------ 
# useing scaled...
# data.scaled #...

# Cluster into k=3 clusters:
myClusters <- kmeans(data.scaled, 3)

# Summary of the clusters
summary(myClusters)

# Centers (mean values) of the clusters
myClusters$centers

# Cluster assignments
myClusters$cluster

# Within-cluster sum of squares and total sum of squares across clusters
myClusters$withinss
myClusters$tot.withinss


# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(data.scaled, myClusters$cluster, color=TRUE, 
         shade=TRUE, labels=2, lines=0)



# ------------------ 
# K Means for many different Ks
# ------------------ 

# Cluster into k=i clusters:
# and accumulate the within-cluster sum of squares error for each k-value
errValues <- NULL
for (i in 1:25) {
  myClusters <- kmeans(data.scaled, i)
  errValues[i] <- myClusters$tot.withinss
  # It was fun to plot every single one, but I don't necessarily want to run it every time.
  # clusplot(data.scaled, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
}

plot(errValues)




# ------------------ 
# K Means for chosen K=6
# ------------------ 

myClusters <- kmeans(data.scaled, 6)
clusplot(data.scaled, myClusters$cluster, color=TRUE, 
         shade=TRUE, labels=2, lines=0)

# Answer the remaining questions with this data (k=6)

# 1) List the states in each cluster.
myClusters$cluster
sort(myClusters$cluster == 1) # this shows all the states with cluster as 1 (sorted to the end)
sort(myClusters$cluster == 2) #
sort(myClusters$cluster == 3) #
sort(myClusters$cluster == 4) #
sort(myClusters$cluster == 5) #
sort(myClusters$cluster == 6) #

# 2) Use "clusplot" to plot a 2D representation of the clustering.
# Already done above...

# 3) Analyze the centers of each of these clusters. 
# Can you identify any insight into this clustering?

myClusters$centers

# Cluster 1 Is by far the very highest in population
# Cluster 6 is by far the highest in income,
# Cluster 5 is the lowest in income and highest in illiteracy
# Clusters 5 and 6 have the lowest life expectancy 
# Clusters 1,5 and 6 have the highest murder
# Cluster 6 has the highest frost, but Cluster 2 isn’t too far behind it. 
# Cluster 6 has the highest area by far, and Cluster 1 also has a lot of area.






# ------------------------------------------------------------------------
# ABOVE AND BEYOND :: K-Means Clustering w/ World Happiness Report (2017)
# ------------------------------------------------------------------------

`2017` <- read.csv("~/py/CS450BYUI/CS450_10/world-happiness-report/2017.csv")
# View(`2017`)
dimnames(`2017`)
myData <- `2017`
column.names <- dimnames(myData)
column.names <- column.names[2]
data.scaled <- myData
data.scaled[,2:12] <- scale(myData[,2:12])
# View(`2017`)
# Yay, it's scaled now!
data.scaled <- data.scaled[,-2] # remove "happiness ranking" because it's kind of backwards

# Turn the first column into the row names
# https://stackoverflow.com/questions/5555408/convert-the-values-in-a-column-into-row-names-in-an-existing-data-frame-in-r
data.scaled2 <- data.scaled[,-1]
rownames(data.scaled2) <- data.scaled[,1]
data.scaled <- data.scaled2

# Cluster into k=i clusters:
# and accumulate the within-cluster sum of squares error for each k-value
errValues <- NULL

for (i in 1:5) {
  myClusters <- kmeans(data.scaled, i)
  errValues[i] <- myClusters$tot.withinss
  # It was fun to plot every single one, but I don't necessarily want to run it every time.
  clusplot(data.scaled, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
}
# plot(errValues)


# ------------------ 
# K Means for chosen K=6
# ------------------ 

# Do it for the chosen number of K
myClusters <- kmeans(data.scaled, 6)
clusplot(data.scaled, myClusters$cluster, color=TRUE, 
         shade=TRUE, labels=2, lines=0)

# 3) Analyze the centers of each of these clusters. 
# Can you identify any insight into this clustering?

myClusters$centers



# ------------------ 
# Do all the same stuff EXCEPT LEAVE OUT GDP
# ------------------ 
# THINKING::
# So if I leave out GDP, will I discover anything else about Happiness? 
# Will I be able to make any claims about GDP's involvement in happiness?
# Should I try an ANOVA or something?
# A Linear Regression?
# Which feature has a greatest correlation (least error ss) between happiness index?


# At this point, "Economy..GDP.per.Capita" is at index 4, 
# so we can remove it there.
data.scaled <- data.scaled[,-4]

errValues <- NULL

for (i in 10:50) {
  myClusters <- kmeans(data.scaled, i)
  errValues[i] <- myClusters$tot.withinss
  # It was fun to plot every single one, but I don't necessarily want to run it every time.
  # clusplot(data.scaled, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
}
plot(errValues)



