import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import decomposition
from xgboost import XGBClassifier
import pickle
import lightgbm as lgb
from sklearn.decomposition import PCA

def datarize2(path):
  df_rows = []
  with open(path, encoding="utf-8") as f:
    for line in f.readlines():
      line = line.strip()
      features = line.split("\x01")
      df_rows.append(features)

  df = pd.DataFrame(df_rows)
  columns = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
              "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
              "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
              "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
              "enaging_user_account_creation", "engagee_follows_engager"]
  df.columns = columns
  return df


def encoding2(df):

  media_tokens = [] # Photo, Vedio, Gif
  for m in df['present_media']:
    try:
      token = m.split('\t')
      p_num = token.count('Photo')
      v_num = token.count('Video')
      g_num = token.count('Gif')
      media_count = [p_num, v_num, g_num]
    except: media_count = [0, 0, 0]
    media_tokens.append(media_count)
  media_df = pd.DataFrame(media_tokens, columns=['Photo', 'Vidio', 'Gif'])

  type_tokens = [] # Retweet, Quote, Reply, TopLevel
  for t in df['tweet_type']:
    if t=='Retweet':
      token = [1, 0, 0, 0]
    elif t=='Quote':
      token = [0, 1, 0, 0]
    elif t=='Reply':
      token = [0, 0, 1, 0]
    elif t=='TopLevel':
      token = [0, 0, 0, 1]
    else: token =[0, 0, 0, 0]
    type_tokens.append(token)
  type_df = pd.DataFrame(type_tokens, columns=['Retweet', 'Quote', 'Reply', 'TopLevel'])

  langs = list(df['language'].value_counts()[:10].index) # Top 10 langs and others
  langs.append('others')
  print(langs)
  lang_tokens = []
  for lang in df['language']:
    token = [0]*11
    if lang not in langs:
      idx =10
    else:
      idx = langs.index(lang)
    token[idx]+=1
    lang_tokens.append(token)
  lang_df = pd.DataFrame(lang_tokens, columns=langs)

  is_present_links=[]
  for l in df['present_links']:
    if type(l)!=float:
      is_present_links.append([1])
    else:
      is_present_links.append([0])
  is_present_links = pd.DataFrame(is_present_links)

  true_false_cols = df[['engaged_with_user_is_verified', 'enaging_user_is_verified', 'engagee_follows_engager']]
  true_false_cols = pd.get_dummies(
      true_false_cols, 
      columns=['engaged_with_user_is_verified', 'enaging_user_is_verified', 'engagee_follows_engager'],
      prefix=['engaged_with_user_is_verified', 'enaging_user_is_verified', 'engagee_follows_engager'],
      prefix_sep="_",
      dummy_na=False,
      drop_first=False
  )

  numeric_df = df[['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'enaging_user_follower_count', 'enaging_user_following_count']]

  texts = df['text_tokens'].str.split()
  text_len_tokens=[]
  for text in texts:
    text_len_tokens.append(len(text))
  text_len_df = pd.DataFrame(text_len_tokens)

  docs_vectors = []
  for text in texts:
    temp = []
    for word in text:
      try:
        word_vec = embeddings[word]
        temp.append(word_vec)
      except:
        pass
    doc_vector = np.mean(temp, axis=0)
    docs_vectors.append(doc_vector)
  text_df = pd.DataFrame(docs_vectors)


  x_test = pd.concat([text_df, text_len_df, numeric_df, true_false_cols, is_present_links, lang_df, type_df, media_df], axis=1)
  return x_test

def pca(df):
  pca = PCA(n_components=2)
  PCA_X = pca.fit(df.values).transform(df.values)
  return PCA_X

# path = '/content/drive/My Drive/網路搜索與探勘/recsys challenge 2020/validation data'
# filepaths = [os.path.join(path,p) for p in os.listdir(path)[2:]]


# 載入模型
# embeddings = pickle.load(open("/content/drive/My Drive/網路搜索與探勘/recsys challenge 2020/model/word_vec.pickle.dat", "rb"))
# model = pickle.load(open("/content/drive/My Drive/網路搜索與探勘/recsys challenge 2020/model/like.pickle.dat", "rb"))


# 前處理
df = datarize2("val.tsv")
df2 = encoding2(df)
df2 = pca(df2)

# 分類
y_pred = model.predict_proba(df2)
result_df = df[['tweet_id', 'enaging_user_id']]
result_df['prediction'] = y_pred[:,1]
result_df.columns = ['Tweet_Id', 'User_Id', 'Prediction']
result_df.to_csv(f'/content/drive/My Drive/網路搜索與探勘/recsys challenge 2020/prediction data/like_pred{i}.csv', index=False, header=False)