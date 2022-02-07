import os
import numpy as np
import pickle

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
# print(all_features_to_idx)
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

path = "train_part/"
dirs = os.listdir( path )
# print(dirs)
X = []
y = []

for file in dirs:
  count = 0
  with open("train_part/"+file, encoding="utf-8") as f:

    for line in f:

      # one line one data
      line = line.strip()
      # features is a list
      features = line.split("\x01")
        
      new_y = []
      for label, idx in labels_to_idx.items():
          # print("label {} has value {}".format(label, features[idx]))
          new_y.append(features[idx])
      y.append(new_y)

    
    count += 1
  if count == 1: break

y = np.asarray(y)


# for i in range(len(X)):
#   X[i] = X[i].split("\t")


y1 = y[:,0].tolist()
y2 = y[:,1].tolist()
y3 = y[:,2].tolist()
y4 = y[:,3].tolist()

print(y1)
for i in range(len(y1)):
  if y1[i] == '':
    y1[i] = 0
  else:
    y1[i] = 1

for i in range(len(y2)):
  if y2[i] == '':
    y2[i] = 0
  else:
    y2[i] = 1

for i in range(len(y3)):
  if y3[i] == '':
    y3[i] = 0
  else:
    y3[i] = 1

for i in range(len(y4)):
  if y4[i] == '':
    y4[i] = 0
  else:
    y4[i] = 1

