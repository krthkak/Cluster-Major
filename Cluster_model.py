import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
y_pred = kmeans.fit_predict(x.iloc[:,0].values.reshape(-1,1))




plt.scatter(x["Age"],x["Age"],c = y_pred,cmap = "rainbow")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()




data1 = data.drop(["Timestamp","Name"],axis = 1)




data1.head()


# To flatten the column Interests


data2 = data1.Interests.str.split(";").apply(pd.Series)
data2.index = data1.set_index(["Age","Gender","Occupation","Interests"]).index
data3 = data2.stack().reset_index(["Age","Gender","Occupation","Interests"])




data3.head()


data4 = data3.drop(["Interests"],axis = 1)


data4.head()

desc_welfare = data4.groupby([0])["Age"].count().sort_values(ascending = False).index
asc_welfare = data4.groupby([0])["Age"].count().sort_values().index

map_welfare_asc = {k:v for v,k in enumerate(desc_welfare,start = 1)}
map_welfare_desc = {k:v for v,k in enumerate(asc_welfare,start = 1)}

map_welfare_asc

data4[0] = data4[0].map(map_welfare_asc)

data4.head()

km_w_asc_map = KMeans(n_clusters = 4)
km_w_asc_map_pred = km_w_asc_map.fit_predict(data4.iloc[:,[0,-1]])

plt.scatter(data4[0],data4.Age,c = km_w_asc_map_pred,cmap = "rainbow",alpha=0.6)
plt.ylabel("Age")
plt.xlabel("Interests")
plt.xticks(sorted(data4[0].unique()),list(map_welfare_asc.keys()),rotation = "vertical")
plt.show()




import pickle
with open("clustering_with_age.pkl","wb") as file:
    pickle.dump(km_w_asc_map,file)



#Gender wise  INterests
age_interests = data4.groupby([0,"Gender"],as_index=False)["Occupation"].count()
male = list(age_interests[age_interests["Gender"]=="Male"]["Occupation"])
female = list(age_interests[age_interests["Gender"]=="Female"]["Occupation"])


#Age Encoding
data5 = data4.copy()
data5["Gender"] = data5["Gender"].map({"Male":1,"Female":0})

#Visualising gender vs interests

plt.scatter(data5.Gender,data5[0])
plt.show()

#Visualising gender vs Interests 2.0
ax1 = plt.subplot(111)
ax1.bar(range(1,15),male,width = 0.3,color = "red",label = "Male",align='center')
ax1.bar(range(1,15),female,width = 0.3,color = "blue",label = "Female",align='center')
plt.xlabel("Gender")
plt.ylabel("Interests")
plt.xticks(range(1,15),list(map_welfare_asc.keys()),rotation = "vertical")
plt.legend([male_plot,female_plot])
plt.show()


