library(datasets)
myData = state.x77

column.names = c("Population", "Income", "Illiteracy", 
                 "Life Exp", "Murder", "HS Grad", "Frost", "Area")


# As an Overview, let's plan to 
# 1) Follow the prescribed methods
# 2) Then Try clustering on each individual column and see the results


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













