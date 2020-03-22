from flask import Flask,request,jsonify
import pandas
import numpy as np
import pickle
from sklearn.cluster import KMeans

app = Flask(__name__)

cluster_model = None
with open("clustering_with_age.pkl","rb") as file:
    cluster_model = pickle.load(file)

cluster_top_interests = None
with open("top_interests_by_clusters.pkl","rb") as file:
    cluster_top_interests = pickle.load(file)

male_interests = None
with open("male_top_interests.pkl","rb") as file:
    male_interests = pickle.load(file)

female_interests = None
with open("female_top_interests.pkl","rb") as file:
    female_interests = pickle.load(file)

interest_id_dict = None
with open("interest_id_dict.pkl","rb") as file:
    interest_id_dict = pickle.load(file)

def get_interests(age,gender):
    cluster = (cluster_model.predict(np.array(age))[0])
    cluster_interests = cluster_top_interests[cluster]
    interests = list()
    if gender.lower() == "male":
        interests = list(set(cluster_interests) & set(male_interests))
    if gender.lower() == "female":
        interests = list(set(cluster_interests) & set(female_interests))


    output = {}
    output["Cluster"] = int(cluster)
    if len(interests)!=0:
        output["Final Interests"] = interests
    else:
        output["Final Interests"] = cluster_interests

    interest_id = list(map(lambda x:interest_id_dict[x],output["Final Interests"]))
    output["Interests Ids"] = interest_id
    output["Interests-Id"] = interest_id_dict
    return jsonify(output)



@app.route('/',methods = ["GET","POST"])
def home():
    return "Cluster Api is working"

@app.route('/suggest-category',methods = ["GET","POST"])
def predict():
    age = int(request.args.get("age"))
    gender = request.args.get("gender")
    return get_interests(age,gender)


if __name__ == "__main__":
    app.run(debug=True)