import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
sns.set()
data = pd.read_csv("C:\\Users\\arvin\\Desktop\\major proj\\Reach Out.csv")


data_dropped = data.drop(["Timestamp","Name","Interests"],axis = 1)

data_dropped.head()

for i in data_dropped.columns:
    print(i,data_dropped[i].isnull().sum(),sep = "=>")

groups = data_dropped.groupby(["Occupation"])["Age"].count()

data_dropped["Gender"] = data_dropped["Gender"].map({"Female":0,"Male":1})

x = data_dropped.iloc[:,0:-1]

x.head()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
y_pred = kmeans.fit_predict(x.iloc[:,0:1].sort_values(["Age"],ascending = True))

#Persisting the model with 4 Clusters
with open("clustering_with_age.pkl","wb") as file:
    pickle.dump(kmeans,file)


plt.scatter(x["Age"],x["Age"],c = y_pred,cmap = "rainbow")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()

data1 = data.drop(["Timestamp","Name"],axis = 1)

# To flatten the column Interests
data2 = data1.Interests.str.split(";").apply(pd.Series)
data2.index = data1.set_index(["Age","Gender","Occupation","Interests"]).index
data3 = data2.stack().reset_index(["Age","Gender","Occupation","Interests"])

data4 = data3.drop(["Interests"],axis = 1)

#Creating Encoding logic
desc_welfare = data4.groupby([0])["Age"].count().sort_values(ascending = False).index
asc_welfare = data4.groupby([0])["Age"].count().sort_values().index

#This is currently used
map_welfare_asc = {k:v for v,k in enumerate(desc_welfare,start = 1)}
id_interest_dict = dict(zip(map_welfare_asc.values(),map_welfare_asc.keys()))

#Persisting interest_id dictionary
with open("interest_id_dict.pkl","wb") as file:
    pickle.dump(map_welfare_asc,file)

#Persisting id_dictionary dictionary
with open("id_interest_dict.pkl","wb") as file:
    pickle.dump(id_interest_dict,file)

#Not recommended for use
map_welfare_desc = {k:v for v,k in enumerate(asc_welfare,start = 1)}

data4[0] = data4[0].map(map_welfare_asc)
data4.head()

"""---------------------------------------Age Wise Interests Extrapolation-------------------------------"""

"""num_clusters = 4
km_w_asc_map = KMeans(n_clusters = num_clusters)
km_w_asc_map_pred = km_w_asc_map.fit_predict(data4.iloc[:,[0,-1]])
data4["Clusters"] = km_w_asc_map_pred
data4 = data4.reset_index(drop = True)"""

plt.scatter(data4[0],data4.Age,c = km_w_asc_map_pred,cmap = "rainbow",alpha=0.6)
plt.ylabel("Age")
plt.xlabel("Interests")
plt.xticks(sorted(data4[0].unique()),list(map_welfare_asc.keys()),rotation = "vertical")
plt.show()




#Getting the interest cluster wise
data6 = data3.copy()
data6.drop(["Interests","Gender"],axis = 1,inplace= True)
data6["Clusters"] = list(data4["Clusters"].copy())
age_cluster_wise =data6.groupby(["Clusters",0],as_index=False)["Occupation"].count()
all_interests_by_clusters = {}
top_interests_by_clusters = {}
for i in range(num_clusters):
    all_interests_by_clusters[i] = list(age_cluster_wise[age_cluster_wise["Clusters"]==i].sort_values(["Occupation"],ascending = False).iloc[:,1])
    top_interests_by_clusters[i] = list(age_cluster_wise[age_cluster_wise["Clusters"]==i].sort_values(["Occupation"],ascending = False).iloc[:4,1])

#persisting cluster wise all interests
with open("all_interests_by_clusters.pkl","wb") as file:
    pickle.dump(all_interests_by_clusters,file)

#persisting cluster wise top interests
with open("top_interests_by_clusters.pkl","wb") as file:
    pickle.dump(top_interests_by_clusters,file)

"""------------------------Gender Wise Interests Extrapolation-------------------------------------------"""
#Gender wise  INterests
age_interests = data4.groupby([0,"Gender"],as_index=False)["Occupation"].count()
male = list(age_interests[age_interests["Gender"]=="Male"]["Occupation"])
female = list(age_interests[age_interests["Gender"]=="Female"]["Occupation"])
female.append(0)


#Age Encoding
data5 = data4.copy()
data5["Gender"] = data5["Gender"].map({"Male":1,"Female":0})

#Visualising gender vs interests
plt.scatter(data5.Gender,data5[0])
plt.show()

#Visualising gender vs Interests 2.0
plt.bar(range(1,15),male,color = "red",label = "Male")
plt.bar(range(1,15),female,color = "blue",label = "Female")
plt.xlabel("Gender")
plt.ylabel("Interests")
plt.xticks(range(1,15),list(map_welfare_asc.keys()),rotation = "vertical")
plt.show()

"""Gathering Top interests of Respective Genders"""
male_interests_names = dict(zip(map_welfare_asc.keys(),male))
male_top_interests = sorted(male_interests_names , key = male_interests_names.get,reverse = True)[:5]
#Persisting male_top_interests
with open("male_top_interests.pkl","wb") as file:
    pickle.dump(male_top_interests,file)

female_interests_names = dict(zip(map_welfare_asc.keys(),female))
female_top_interests = sorted(female_interests_names , key = female_interests_names.get,reverse = True)[:5]
#Persisting female_top_interests
with open("female_top_interests.pkl","wb") as file:
    pickle.dump(female_top_interests,file)
