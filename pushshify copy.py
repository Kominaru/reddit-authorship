from time import sleep
import requests
import json
import csv
import datetime
from psaw import PushshiftAPI

api = PushshiftAPI()

def getPushshiftData(after, before, query=None, sub="All"):
    url = f"https://api.pushshift.io/reddit/search/comment/{f'?q={query}&' if query is not None else '?'}size=1000&after={str(after)}&before={before}&subreddit={sub}"
    print(url)
    while True:
        try:
            r = requests.get(url)
            data = json.loads(r.text)
            
            return data['data']
        except:
            print("Errored, retrying in 30s")
            sleep(30)

def collectSubData(subm):
    subData = list() #list to store data points
    body = subm['body'].replace('\n',' ').replace('\t',' ').replace('\r',' ').replace('\r\n',' ')
    author = subm['author']
    com_id = subm['id']
    score = subm['score']
    created = datetime.datetime.fromtimestamp(subm['created_utc']) #1520561700.0
    permalink = subm['permalink']
    subreddit = subm['subreddit']
    parent_id = subm['parent_id']
    post_id=subm['link_id']
    
    subData.append((com_id,subreddit,body,author,score,created,permalink,parent_id,post_id))
    subStats[com_id] = subData

#Subreddit to query
sub="Coronavirus"
#before and after dates
before = datetime.datetime(2021, 3, 30, 0)
after = datetime.datetime(2021, 1, 1, 0)
query = ""
subCount = 0
subStats = {}

data = getPushshiftData(after, before, sub=sub,query=query)# Will run until all posts have been gathered 
# from the 'after' date up until before date
while len(data) > 0:
    for submission in data:
        collectSubData(submission)
        subCount+=1
    # Calls getPushshiftData() with the created date of the last submission
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    sleep(2)
    after = data[-1]['created_utc']

    data = getPushshiftData(after, before, sub=sub,query=query)
    
print(len(data))

print(str(len(subStats)) + " submissions have added to list")
print("1st entry is:")
print(list(subStats.values())[0][0][1] + " created: " + str(list(subStats.values())[0][0][5]))
print("Last entry is:")
print(list(subStats.values())[-1][0][1] + " created: " + str(list(subStats.values())[-1][0][5]))

def updateSubs_file():
    upload_count = 0
    print("input filename of submission file, please add .csv")
    filename = input()
    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file: 
        a = csv.writer(file, delimiter=',')
        headers = ["Post ID","Subreddit","Body","Author","Score","Publish Date","Permalink","Parent_id","Post_id"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub][0])
            upload_count+=1
            
        print(str(upload_count) + " submissions have been uploaded")

updateSubs_file()