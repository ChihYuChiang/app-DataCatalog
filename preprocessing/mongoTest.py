import pymongo
import json
with open('./scraper/ref/credential_mongo.json', 'r') as f:
    CREDENTIAL_MONGO = json.load(f)
    MONGO_URI = CREDENTIAL_MONGO['MONGO_URI']
    MONGO_DATABASE = CREDENTIAL_MONGO['MONGO_DATABASE']

client = pymongo.MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]

result = db['Kaggle_List'].find({
        
    }, {
        '_id': False,
        'title': True
    })
list(result)