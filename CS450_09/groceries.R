# install.packages('arules')

library(arules)
data(Groceries)

# save to 'gro' instead of 'Groceries' for ease of typing
gro <- Groceries

print("Dimensions of Groceries: ")
dim(gro)

print("Categories used in market basket analysis: ")
dimnames(gro)

# 'inspect' makes it easiest to see what's going on
print("Inspecting the head of the dataset: ")
inspect(head(gro))

# See the size of each row in the 
size(head(gro))

# another way to inspect, but this is less pretty, I won't use it anymore.
# LIST(head(gro))

#######################
# READY to EXPERIMENT #
#######################


rules <- apriori(gro,  parameter = list(supp= 0.01, confidence=.01, minlen=2))
rules_conf <- sort (rules, by="lift", decreasing=TRUE)
inspect(head(rules_conf, 12))
