


import pandas as pd

# *********** LOAD DATA ***********
headers = ["age", "workclass", "fnlwgt", "education", 
            "education-num", "marital-status", "occupation", 
            "relationship", "race", "sex", "capital-gain", 
            "capital-loss", "hours-per-week", "native-country", "answer"]
df = pd.read_csv("./adult.csv", header=None skip_initial_)
df.columns = headers
print(df.head(20))



# Replace missing values "?" with 
# --> Strip out that row (rows with ?)
# --> Replace with mean XXX
print((df[headers] == ' ?').sum()) # has missing data
df = df.drop(df[df.workclass == " ?"].index)
df = df.drop(df[df.occupation == " ?"].index)
df = df.drop(df[df["native-country"] == " ?"].index) # would be intereesting to see that there is very very little correlation between missing native-country and the other 2 commonly missing values
print((df[headers] == ' ?').sum()) # no more missing data


# Encode Categorical Data
# --> Strip out categorical 'education' because numeric 'education-num' covers it fine
# --> 
