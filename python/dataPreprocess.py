import gzip
from collections import defaultdict
from datetime import datetime
import json


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# Location of the dataset
datasetLocation = '/home/kavach/Dev/NewResearch/E2P/sasRec/SASRec.pytorch/python/data/raw/Magazine_Subscriptions_5.json.gz'

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

# Add the name of the dataSet
dataset_name = 'Magazine_Subscriptions'

# change the location here also if needed
f = open('/home/kavach/Dev/NewResearch/E2P/sasRec/SASRec.pytorch/python/data/'+'reviews_' + dataset_name + '.txt', 'w')
for l in parse(datasetLocation):
    line += 1
    f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    countU[rev] += 1
    countP[asin] += 1
f.close()

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
for l in parse(datasetLocation):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([time, itemid])
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

# Add the location below of where to save the file
f = open('/home/kavach/Dev/NewResearch/E2P/sasRec/SASRec.pytorch/python/data/' + dataset_name +'.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()