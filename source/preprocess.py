import pickle
import os
import pandas as pd
import zipfile
import tqdm
import json
import warnings
import numpy as np
import torch

torch.manual_seed(2023)

warnings.filterwarnings("ignore")
os.chdir('data/')
print('What splitting strategy you want to use?(global_temporal, user_temporal)')
split_strategy = input()

# %% global_temporal
if split_strategy == 'global_temporal':
    
    # %%% preprocess movielens1M
    movielens = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None)
    movielens.columns = ['userid', 'movieid', 'rating', 'time']
    movielens = movielens.sort_values('time')
    user_support = pd.read_csv('ml-1m/users.dat', sep = '::', header = None).iloc[:, 0:2]
    user_support.columns = ['userid', 'gender']
    item_support_panel = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, encoding = "ISO-8859-1").iloc[:, [0, 2]]
    item_support_panel.columns = ['movieid', 'genres']
    item_support_panel = item_support_panel[item_support_panel['movieid'].isin(list(movielens['movieid']))]
    item_support = {}
    for i in range(len(item_support_panel)):
        row = item_support_panel.iloc[i, :]
        item_support[row['movieid']] = row['genres'].split('|')
        
    train_cut, val_cut = int(movielens.shape[0]*0.8), int(movielens.shape[0]*0.9)
    movielens_train, movielens_val, movielens_test = \
        movielens.iloc[:train_cut, :], \
        movielens.iloc[train_cut:val_cut, :], \
        movielens.iloc[val_cut:, :]
        
    ## drop timestamp
    train, val, test = \
        movielens_train[['userid', 'movieid', 'rating']], \
        movielens_val[['userid', 'movieid', 'rating']], \
        movielens_test[['userid', 'movieid', 'rating']]
        
    ## normalize to tensors
    user_support['id'] = range(user_support.shape[0])
    user_support['if_male'] = np.where(user_support['gender']=='M', 1, 0)
    train, val, test = \
        train.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        val.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        test.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid')
    user_gender = torch.tensor(user_support[['id', 'if_male']].values)
    item_tag = {}
    for i, key in enumerate(item_support.keys()):
        item_tag[i] = item_support[key]
    train['id_'] = train['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    val['id_'] = val['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    test['id_'] = test['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    user_unknown_train = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_'])
        user_unknown_train[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_val = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_'])
        user_unknown_val[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_test = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_']) + list(test[test['id']==i]['id_'])
        user_unknown_test[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    train, val, test = \
        torch.tensor(train[['id', 'id_', 'rating']].values), \
        torch.tensor(val[['id', 'id_', 'rating']].values), \
        torch.tensor(test[['id', 'id_', 'rating']].values)
        
    ## save
    torch.save(train, 'ml-1m/train.pt')
    torch.save(val, 'ml-1m/val.pt')
    torch.save(test, 'ml-1m/test.pt')
    torch.save(user_gender, 'ml-1m/user_gender.pt')
    with open('ml-1m/user_unknown_train.pickle', 'wb') as handle:
        pickle.dump(user_unknown_train, handle)
    with open('ml-1m/user_unknown_val.pickle', 'wb') as handle:
        pickle.dump(user_unknown_val, handle)
    with open('ml-1m/user_unknown_test.pickle', 'wb') as handle:
        pickle.dump(user_unknown_test, handle)
    with open('ml-1m/item_tag.pickle', 'wb') as handle:
        pickle.dump(item_tag, handle)
    
    # %%% preprocess lastfm1k: interaction
    ## load the lastfm data
    lastfm = pd.read_csv("lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv",
                         sep = '\t', header = None, usecols = [0,1,2,3,4,5])
    lastfm.columns = ['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname']
    artid_name = lastfm[['artid', 'artname']].drop_duplicates().reset_index(drop=True)
    
    ## prune to leave 3 columns
    lastfm = lastfm[['userid', 'artid', 'timestamp']]
    
    ## convert timestamp
    lastfm['timestamp'] = pd.to_datetime(lastfm['timestamp'])
    
    ## drop duplicates
    lastfm_ = lastfm.groupby(by = ['userid', 'artid']).min().reset_index(drop = False)
    lastfm = lastfm_.sort_values(by = 'timestamp')
    
    ## perform refinement: user interaction >= thres, item interaction >= thres
    thres = 20
    dense_item0 = list(lastfm['artid'].unique())
    dense_user0 = list(lastfm['userid'].unique())
    
    while True:    
    
        ### perform refinement by user
        count_user =  ((lastfm[lastfm['artid'].isin(dense_item0)]).groupby('userid').count())['artid']
        dense_user = list((count_user[count_user > thres]).index)
        dense_user0 = dense_user
        
        ### perform refinement by item
        count_item =  ((lastfm[lastfm['userid'].isin(dense_user0)]).groupby('artid').count())['userid']
        dense_item = list((count_item[count_item > thres]).index)
        if len(dense_item) == len(dense_item0):
            break
        else:
            dense_item0 = dense_item
            
    lastfm = lastfm[(lastfm['userid'].isin(dense_user))&(lastfm['artid'].isin(dense_item))]
    artid_name = artid_name[artid_name['artid'].isin(dense_item)].reset_index(drop=True)
    
    # %%% preprocess lastfm1k: user support
    user_support = pd.read_csv('lastfm-dataset-1K/userid-profile.tsv', sep = '\t')
    user_support = user_support[user_support['#id'].isin(list(lastfm['userid'].unique()))][['#id', 'gender']]
    user_support.columns = ['userid', 'gender']
    user_support = user_support.dropna()
    
    # %%% preprocess lastfm1k: item support
    item_support_artist = []
    item_support_tags = []
    
    archive = zipfile.ZipFile('lastfm-dataset-1K/lastfm_train.zip', 'r')
    files = archive.infolist()
    name_exist = list(artid_name['artname'])
    for file in tqdm.tqdm(files):
        if not file.is_dir():
            content = archive.read(file.filename)
            try:
                dic = json.loads(content.decode('utf-8'))
                if dic['artist'] in name_exist:
                    if len(dic['tags']) > 0:
                        item_support_artist.append(dic['artist'])
                        item_support_tags.append(dic['tags'])
            except:
                continue
            
    archive = zipfile.ZipFile('lastfm-dataset-1K/lastfm_test.zip', 'r')
    files = archive.infolist()
    name_exist = list(artid_name['artname'])
    for file in tqdm.tqdm(files):
        if not file.is_dir():
            content = archive.read(file.filename)
            try:
                dic = json.loads(content.decode('utf-8'))
                if dic['artist'] in name_exist:
                    if len(dic['tags']) > 0:
                        item_support_artist.append(dic['artist'])
                        item_support_tags.append(dic['tags'])
            except:
                continue
            
    item_support_artist_unique = []
    item_support_tags_unique = []
    item_support_artist_index = [i for i in range(len(item_support_artist))]
    panel = pd.DataFrame({'name': item_support_artist,
                          'id': item_support_artist_index})
    panel = panel.drop_duplicates(subset = 'name')
    item_support_artist_unique = list(panel['name'])
    item_support_tags_unique = [item_support_tags[i] for i in list(panel['id'])]
    item_support_artid_unique = [list(artid_name['artid'])[list(artid_name['artname']).index(name)] \
                                 for name in item_support_artist_unique]
    
    # %%% preprocess lastfm1k: interaction final refinement
    ## initialize
    lastfm = lastfm[(lastfm['artid'].isin(item_support_artid_unique))&\
                    (lastfm['userid'].isin(list(user_support['userid'])))]
    dense_item0 = list(lastfm['artid'].unique())
    dense_user0 = list(lastfm['userid'].unique())
    
    while True:    
    
        ### perform refinement by user
        count_user =  ((lastfm[lastfm['artid'].isin(dense_item0)]).groupby('userid').count())['artid']
        dense_user = list((count_user[count_user > thres]).index)
        dense_user0 = dense_user
        
        ### perform refinement by item
        count_item =  ((lastfm[lastfm['userid'].isin(dense_user0)]).groupby('artid').count())['userid']
        dense_item = list((count_item[count_item > thres]).index)
        if len(dense_item) == len(dense_item0):
            break
        else:
            dense_item0 = dense_item
            
    lastfm = lastfm[(lastfm['userid'].isin(dense_user))&(lastfm['artid'].isin(dense_item))]
    artid_name = artid_name[artid_name['artid'].isin(dense_item)].reset_index(drop=True)
    
    ## finalize user support, item support
    user_support = user_support[user_support['userid'].isin(list(lastfm['userid'].unique()))]
    item_support = {}
    for i in range(len(item_support_artid_unique)):
        artid, tags = item_support_artid_unique[i], item_support_tags_unique[i]
        item_support[artid] = tags
    diff = list(set(item_support.keys()).difference(set(list(lastfm['artid'].unique()))))
    for artid in diff:
        del item_support[artid]
    item_support_copy = item_support.copy()
    item_support = {}
    for key in item_support_copy.keys():
        item_support[key] = [item[0] for item in item_support_copy[key] if int(item[1]) > 50]
    
    # %%% preprocess lastfm1k: split & save
    ## perform the raw split by timestamp
    train_cut, val_cut = int(lastfm.shape[0]*0.8), int(lastfm.shape[0]*0.9)
    lastfm_train, lastfm_val, lastfm_test = \
        lastfm.iloc[:train_cut, :], \
        lastfm.iloc[train_cut:val_cut, :], \
        lastfm.iloc[val_cut:, :]
    
    ## drop timestamp
    train, val, test = \
        lastfm_train[['userid', 'artid']], \
        lastfm_val[['userid', 'artid']], \
        lastfm_test[['userid', 'artid']]
        
    ## normalize to tensors
    user_support['id'] = range(user_support.shape[0])
    user_support['if_male'] = np.where(user_support['gender']=='m', 1, 0)
    train, val, test = \
        train.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        val.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        test.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid')
    user_gender = torch.tensor(user_support[['id', 'if_male']].values)
    item_tag = {}
    for i, key in enumerate(item_support.keys()):
        item_tag[i] = item_support[key]
    train['id_'] = train['artid'].replace(list(item_support.keys()), range(len(item_support)))
    val['id_'] = val['artid'].replace(list(item_support.keys()), range(len(item_support)))
    test['id_'] = test['artid'].replace(list(item_support.keys()), range(len(item_support)))
    user_unknown_train = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_'])
        user_unknown_train[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_val = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_'])
        user_unknown_val[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_test = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_']) + list(test[test['id']==i]['id_'])
        user_unknown_test[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    train, val, test = \
        torch.tensor(train[['id', 'id_']].values), \
        torch.tensor(val[['id', 'id_']].values), \
        torch.tensor(test[['id', 'id_']].values)
        
    ## save
    torch.save(train, 'lastfm-dataset-1K//train.pt')
    torch.save(val, 'lastfm-dataset-1K//val.pt')
    torch.save(test, 'lastfm-dataset-1K//test.pt')
    torch.save(user_gender, 'lastfm-dataset-1K//user_gender.pt')
    with open('lastfm-dataset-1K//user_unknown_train.pickle', 'wb') as handle:
        pickle.dump(user_unknown_train, handle)
    with open('lastfm-dataset-1K//user_unknown_val.pickle', 'wb') as handle:
        pickle.dump(user_unknown_val, handle)
    with open('lastfm-dataset-1K//user_unknown_test.pickle', 'wb') as handle:
        pickle.dump(user_unknown_test, handle)
    with open('lastfm-dataset-1K//item_tag.pickle', 'wb') as handle:
        pickle.dump(item_tag, handle)
        
# %% user_temporal
elif split_strategy == 'user_temporal':
    
    # %%% preprocess movielens1M
    movielens = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None)
    movielens.columns = ['userid', 'movieid', 'rating', 'time']
    movielens = movielens.sort_values('time')
    user_support = pd.read_csv('ml-1m/users.dat', sep = '::', header = None).iloc[:, 0:2]
    user_support.columns = ['userid', 'gender']
    item_support_panel = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, encoding = "ISO-8859-1").iloc[:, [0, 2]]
    item_support_panel.columns = ['movieid', 'genres']
    item_support_panel = item_support_panel[item_support_panel['movieid'].isin(list(movielens['movieid']))]
    item_support = {}
    for i in range(len(item_support_panel)):
        row = item_support_panel.iloc[i, :]
        item_support[row['movieid']] = row['genres'].split('|')
        
    user_list = list(user_support['userid'])
    movielens_train, movielens_val, movielens_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for user in tqdm.tqdm(user_list):
        sub_movielens = movielens.loc[movielens['userid']==user, :]
        train_cut, val_cut = int(sub_movielens.shape[0]*0.8), int(sub_movielens.shape[0]*0.9)
        sub_movielens_train, sub_movielens_val, sub_movielens_test = \
            sub_movielens.iloc[:train_cut, :], \
            sub_movielens.iloc[train_cut:val_cut, :], \
            sub_movielens.iloc[val_cut:, :]
        movielens_train, movielens_val, movielens_test = \
            pd.concat((movielens_train, sub_movielens_train), axis = 0), \
            pd.concat((movielens_val, sub_movielens_val), axis = 0), \
            pd.concat((movielens_test, sub_movielens_test), axis = 0)
        
    ## drop timestamp
    train, val, test = \
        movielens_train[['userid', 'movieid', 'rating']], \
        movielens_val[['userid', 'movieid', 'rating']], \
        movielens_test[['userid', 'movieid', 'rating']]
        
    ## normalize to tensors
    user_support['id'] = range(user_support.shape[0])
    user_support['if_male'] = np.where(user_support['gender']=='M', 1, 0)
    train, val, test = \
        train.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        val.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        test.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid')
    user_gender = torch.tensor(user_support[['id', 'if_male']].values)
    item_tag = {}
    for i, key in enumerate(item_support.keys()):
        item_tag[i] = item_support[key]
    train['id_'] = train['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    val['id_'] = val['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    test['id_'] = test['movieid'].replace(list(item_support.keys()), range(len(item_support)))
    user_unknown_train = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_'])
        user_unknown_train[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_val = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_'])
        user_unknown_val[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_test = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_']) + list(test[test['id']==i]['id_'])
        user_unknown_test[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    train, val, test = \
        torch.tensor(train[['id', 'id_', 'rating']].values), \
        torch.tensor(val[['id', 'id_', 'rating']].values), \
        torch.tensor(test[['id', 'id_', 'rating']].values)
        
    ## save
    torch.save(train, 'ml-1m/train.pt')
    torch.save(val, 'ml-1m/val.pt')
    torch.save(test, 'ml-1m/test.pt')
    torch.save(user_gender, 'ml-1m/user_gender.pt')
    with open('ml-1m/user_unknown_train.pickle', 'wb') as handle:
        pickle.dump(user_unknown_train, handle)
    with open('ml-1m/user_unknown_val.pickle', 'wb') as handle:
        pickle.dump(user_unknown_val, handle)
    with open('ml-1m/user_unknown_test.pickle', 'wb') as handle:
        pickle.dump(user_unknown_test, handle)
    with open('ml-1m/item_tag.pickle', 'wb') as handle:
        pickle.dump(item_tag, handle)
    
    # %%% preprocess lastfm1k: interaction
    ## load the lastfm data
    lastfm = pd.read_csv("lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv",
                         sep = '\t', header = None, usecols = [0,1,2,3,4,5])
    lastfm.columns = ['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname']
    artid_name = lastfm[['artid', 'artname']].drop_duplicates().reset_index(drop=True)
    
    ## prune to leave 3 columns
    lastfm = lastfm[['userid', 'artid', 'timestamp']]
    
    ## convert timestamp
    lastfm['timestamp'] = pd.to_datetime(lastfm['timestamp'])
    
    ## drop duplicates
    lastfm_ = lastfm.groupby(by = ['userid', 'artid']).min().reset_index(drop = False)
    lastfm = lastfm_.sort_values(by = 'timestamp')
    
    ## perform refinement: user interaction >= thres, item interaction >= thres
    thres = 10
    dense_item0 = list(lastfm['artid'].unique())
    dense_user0 = list(lastfm['userid'].unique())
    
    while True:    
    
        ### perform refinement by user
        count_user =  ((lastfm[lastfm['artid'].isin(dense_item0)]).groupby('userid').count())['artid']
        dense_user = list((count_user[count_user > thres]).index)
        dense_user0 = dense_user
        
        ### perform refinement by item
        count_item =  ((lastfm[lastfm['userid'].isin(dense_user0)]).groupby('artid').count())['userid']
        dense_item = list((count_item[count_item > thres]).index)
        if len(dense_item) == len(dense_item0):
            break
        else:
            dense_item0 = dense_item
            
    lastfm = lastfm[(lastfm['userid'].isin(dense_user))&(lastfm['artid'].isin(dense_item))]
    artid_name = artid_name[artid_name['artid'].isin(dense_item)].reset_index(drop=True)
    
    # %%% preprocess lastfm1k: user support
    user_support = pd.read_csv('lastfm-dataset-1K/userid-profile.tsv', sep = '\t')
    user_support = user_support[user_support['#id'].isin(list(lastfm['userid'].unique()))][['#id', 'gender']]
    user_support.columns = ['userid', 'gender']
    user_support = user_support.dropna()
    
    # %%% preprocess lastfm1k: item support
    item_support_artist = []
    item_support_tags = []
    
    archive = zipfile.ZipFile('lastfm-dataset-1K/lastfm_train.zip', 'r')
    files = archive.infolist()
    name_exist = list(artid_name['artname'])
    for file in tqdm.tqdm(files):
        if not file.is_dir():
            content = archive.read(file.filename)
            try:
                dic = json.loads(content.decode('utf-8'))
                if dic['artist'] in name_exist:
                    if len(dic['tags']) > 0:
                        item_support_artist.append(dic['artist'])
                        item_support_tags.append(dic['tags'])
            except:
                continue
            
    archive = zipfile.ZipFile('lastfm-dataset-1K/lastfm_test.zip', 'r')
    files = archive.infolist()
    name_exist = list(artid_name['artname'])
    for file in tqdm.tqdm(files):
        if not file.is_dir():
            content = archive.read(file.filename)
            try:
                dic = json.loads(content.decode('utf-8'))
                if dic['artist'] in name_exist:
                    if len(dic['tags']) > 0:
                        item_support_artist.append(dic['artist'])
                        item_support_tags.append(dic['tags'])
            except:
                continue
            
    item_support_artist_unique = []
    item_support_tags_unique = []
    item_support_artist_index = [i for i in range(len(item_support_artist))]
    panel = pd.DataFrame({'name': item_support_artist,
                          'id': item_support_artist_index})
    panel = panel.drop_duplicates(subset = 'name')
    item_support_artist_unique = list(panel['name'])
    item_support_tags_unique = [item_support_tags[i] for i in list(panel['id'])]
    item_support_artid_unique = [list(artid_name['artid'])[list(artid_name['artname']).index(name)] \
                                 for name in item_support_artist_unique]
    
    # %%% preprocess lastfm1k: interaction final refinement
    ## initialize
    lastfm = lastfm[(lastfm['artid'].isin(item_support_artid_unique))&\
                    (lastfm['userid'].isin(list(user_support['userid'])))]
    dense_item0 = list(lastfm['artid'].unique())
    dense_user0 = list(lastfm['userid'].unique())
    
    while True:    
    
        ### perform refinement by user
        count_user =  ((lastfm[lastfm['artid'].isin(dense_item0)]).groupby('userid').count())['artid']
        dense_user = list((count_user[count_user > thres]).index)
        dense_user0 = dense_user
        
        ### perform refinement by item
        count_item =  ((lastfm[lastfm['userid'].isin(dense_user0)]).groupby('artid').count())['userid']
        dense_item = list((count_item[count_item > thres]).index)
        if len(dense_item) == len(dense_item0):
            break
        else:
            dense_item0 = dense_item
            
    lastfm = lastfm[(lastfm['userid'].isin(dense_user))&(lastfm['artid'].isin(dense_item))]
    artid_name = artid_name[artid_name['artid'].isin(dense_item)].reset_index(drop=True)
    
    ## finalize user support, item support
    user_support = user_support[user_support['userid'].isin(list(lastfm['userid'].unique()))]
    item_support = {}
    for i in range(len(item_support_artid_unique)):
        artid, tags = item_support_artid_unique[i], item_support_tags_unique[i]
        item_support[artid] = tags
    diff = list(set(item_support.keys()).difference(set(list(lastfm['artid'].unique()))))
    for artid in diff:
        del item_support[artid]
    item_support_copy = item_support.copy()
    item_support = {}
    for key in item_support_copy.keys():
        item_support[key] = [item[0] for item in item_support_copy[key] if int(item[1]) > 50]
    
    # %%% preprocess lastfm1k: split & save
    ## perform the raw split by timestamp
    user_list = list(user_support['userid'])
    lastfm_train, lastfm_val, lastfm_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for user in tqdm.tqdm(user_list):
        sub_lastfm = lastfm.loc[lastfm['userid']==user, :]
        train_cut, val_cut = int(sub_lastfm.shape[0]*0.8), int(sub_lastfm.shape[0]*0.9)
        sub_lastfm_train, sub_lastfm_val, sub_lastfm_test = \
            sub_lastfm.iloc[:train_cut, :], \
            sub_lastfm.iloc[train_cut:val_cut, :], \
            sub_lastfm.iloc[val_cut:, :]
        lastfm_train, lastfm_val, lastfm_test = \
            pd.concat((lastfm_train, sub_lastfm_train), axis = 0), \
            pd.concat((lastfm_val, sub_lastfm_val), axis = 0), \
            pd.concat((lastfm_test, sub_lastfm_test), axis = 0)
    
    ## drop timestamp
    train, val, test = \
        lastfm_train[['userid', 'artid']], \
        lastfm_val[['userid', 'artid']], \
        lastfm_test[['userid', 'artid']]
        
    ## normalize to tensors
    user_support['id'] = range(user_support.shape[0])
    user_support['if_male'] = np.where(user_support['gender']=='m', 1, 0)
    train, val, test = \
        train.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        val.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid'), \
        test.merge(user_support[['userid', 'id']], how = 'left', left_on = 'userid', right_on = 'userid')
    user_gender = torch.tensor(user_support[['id', 'if_male']].values)
    item_tag = {}
    for i, key in enumerate(item_support.keys()):
        item_tag[i] = item_support[key]
    train['id_'] = train['artid'].replace(list(item_support.keys()), range(len(item_support)))
    val['id_'] = val['artid'].replace(list(item_support.keys()), range(len(item_support)))
    test['id_'] = test['artid'].replace(list(item_support.keys()), range(len(item_support)))
    user_unknown_train = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_'])
        user_unknown_train[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_val = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_'])
        user_unknown_val[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    user_unknown_test = {}
    for i in user_gender[:, 0].tolist():
        sub = list(train[train['id']==i]['id_']) + list(val[val['id']==i]['id_']) + list(test[test['id']==i]['id_'])
        user_unknown_test[i] = [j for j in range(len(item_tag.keys())) if j not in sub]
    train, val, test = \
        torch.tensor(train[['id', 'id_']].values), \
        torch.tensor(val[['id', 'id_']].values), \
        torch.tensor(test[['id', 'id_']].values)
        
    ## save
    torch.save(train, 'lastfm-dataset-1K//train.pt')
    torch.save(val, 'lastfm-dataset-1K//val.pt')
    torch.save(test, 'lastfm-dataset-1K//test.pt')
    torch.save(user_gender, 'lastfm-dataset-1K//user_gender.pt')
    with open('lastfm-dataset-1K//user_unknown_train.pickle', 'wb') as handle:
        pickle.dump(user_unknown_train, handle)
    with open('lastfm-dataset-1K//user_unknown_val.pickle', 'wb') as handle:
        pickle.dump(user_unknown_val, handle)
    with open('lastfm-dataset-1K//user_unknown_test.pickle', 'wb') as handle:
        pickle.dump(user_unknown_test, handle)
    with open('lastfm-dataset-1K//item_tag.pickle', 'wb') as handle:
        pickle.dump(item_tag, handle)
        
# %% summary statistics
panel = pd.DataFrame(data=None, index = ['movielens', 'lastfm'],
                     columns = ['# Interactions', '# Users', 
                                '# Items', '% Male', 'Mean Tags', 'Std Tags'])

train, val, test = \
    torch.load('ml-1m/train.pt'), \
    torch.load('ml-1m/val.pt'), \
    torch.load('ml-1m/test.pt')
user_gender = torch.load('ml-1m/user_gender.pt')
with open('ml-1m/item_tag.pickle', 'rb') as f:
    item_tag = pickle.load(f)
data = torch.cat((train, val, test), dim = 0)
panel.loc['movielens', '# Interactions'] = data.shape[0]
panel.loc['movielens', '# Users'] = user_gender.shape[0]
panel.loc['movielens', '# Items'] = torch.unique(data[:, 1]).shape[0]
panel.loc['movielens', '% Male'] = user_gender[data[:, 0], 1].sum().item()/data.shape[0]*100
tags = torch.zeros((torch.unique(data[:, 1]).shape[0]))
for i in range(torch.unique(data[:, 1]).shape[0]):
    tags[i] = len(item_tag[i])
panel.loc['movielens', 'Mean Tags'] = tags.mean().item()
panel.loc['movielens', 'Std Tags'] = tags.std().item()

train, val, test = \
    torch.load('lastfm-dataset-1K/train.pt'), \
    torch.load('lastfm-dataset-1K/val.pt'), \
    torch.load('lastfm-dataset-1K/test.pt')
user_gender = torch.load('lastfm-dataset-1K/user_gender.pt')
with open('lastfm-dataset-1K/item_tag.pickle', 'rb') as f:
    item_tag = pickle.load(f)
data = torch.cat((train, val, test), dim = 0)
panel.loc['lastfm', '# Interactions'] = data.shape[0]
panel.loc['lastfm', '# Users'] = user_gender.shape[0]
panel.loc['lastfm', '# Items'] = torch.unique(data[:, 1]).shape[0]
panel.loc['lastfm', '% Male'] = user_gender[data[:, 0], 1].sum().item()/data.shape[0]*100
tags = torch.zeros((torch.unique(data[:, 1]).shape[0]))
for i in range(torch.unique(data[:, 1]).shape[0]):
    tags[i] = len(item_tag[i])
panel.loc['lastfm', 'Mean Tags'] = tags.mean().item()
panel.loc['lastfm', 'Std Tags'] = tags.std().item()
    
print(panel)
        
# %% create unknown sample for movielens
total_eval = 1000
user_num, item_num, style = \
    panel.loc['movielens', '# Users'], \
    panel.loc['movielens', '# Items'], 'explicit'
train, val, test = \
    torch.load('ml-1m/train.pt'), \
    torch.load('ml-1m/val.pt'), \
    torch.load('ml-1m/test.pt')
with open('ml-1m/user_unknown_train.pickle', 'rb') as f:
    user_unknown_train = pickle.load(f)
with open('ml-1m/user_unknown_val.pickle', 'rb') as f:
    user_unknown_val = pickle.load(f)
with open('ml-1m/user_unknown_test.pickle', 'rb') as f:
    user_unknown_test = pickle.load(f)
user_unknown_val_mat, user_unknown_test_mat = \
    torch.zeros((user_num, total_eval)), torch.zeros((user_num, total_eval))
user_unknown_val_cut, user_unknown_test_cut = \
    torch.zeros((user_num)), torch.zeros((user_num))
for user in range(user_num):
    user_known = val[val[:, 0] == user, 1]
    user_unknown = torch.tensor(user_unknown_val[user])
    user_cut = user_known.shape[0]
    user_unknown_val_mat[user, 0:user_cut] = user_known
    user_unknown_val_mat[user, user_cut:] = user_unknown[torch.randperm(user_unknown.shape[0])[0:(total_eval - user_cut)]]
    user_unknown_val_cut[user] = user_cut
    
    user_known = test[test[:, 0] == user, 1]
    user_unknown = torch.tensor(user_unknown_test[user])
    user_cut = user_known.shape[0]
    user_unknown_test_mat[user, 0:user_cut] = user_known
    user_unknown_test_mat[user, user_cut:] = user_unknown[torch.randperm(user_unknown.shape[0])[0:(total_eval - user_cut)]]
    user_unknown_test_cut[user] = user_cut
    
torch.save(user_unknown_val_mat.long(), 'ml-1m//user_unknown_val_mat.pt')
torch.save(user_unknown_test_mat.long(), 'ml-1m//user_unknown_test_mat.pt')
torch.save(user_unknown_val_cut.long(), 'ml-1m//user_unknown_val_cut.pt')
torch.save(user_unknown_test_cut.long(), 'ml-1m//user_unknown_test_cut.pt')
        
# %% create unknown sample for lastfm
total_eval = 1000
user_num, item_num, style = \
    panel.loc['lastfm', '# Users'], \
    panel.loc['lastfm', '# Items'], 'implicit'
train, val, test = \
    torch.load('lastfm-dataset-1K/train.pt'), \
    torch.load('lastfm-dataset-1K/val.pt'), \
    torch.load('lastfm-dataset-1K/test.pt')
with open('lastfm-dataset-1K/user_unknown_train.pickle', 'rb') as f:
    user_unknown_train = pickle.load(f)
with open('lastfm-dataset-1K/user_unknown_val.pickle', 'rb') as f:
    user_unknown_val = pickle.load(f)
with open('lastfm-dataset-1K/user_unknown_test.pickle', 'rb') as f:
    user_unknown_test = pickle.load(f)
user_unknown_val_mat, user_unknown_test_mat = \
    torch.zeros((user_num, total_eval)), torch.zeros((user_num, total_eval))
user_unknown_val_cut, user_unknown_test_cut = \
    torch.zeros((user_num)), torch.zeros((user_num))
for user in range(user_num):
    user_known = val[val[:, 0] == user, 1]
    user_unknown = torch.tensor(user_unknown_val[user])
    user_cut = user_known.shape[0]
    user_unknown_val_mat[user, 0:user_cut] = user_known
    user_unknown_val_mat[user, user_cut:] = user_unknown[torch.randperm(user_unknown.shape[0])[0:(total_eval - user_cut)]]
    user_unknown_val_cut[user] = user_cut
    
    user_known = test[test[:, 0] == user, 1]
    user_unknown = torch.tensor(user_unknown_test[user])
    user_cut = user_known.shape[0]
    user_unknown_test_mat[user, 0:user_cut] = user_known
    user_unknown_test_mat[user, user_cut:] = user_unknown[torch.randperm(user_unknown.shape[0])[0:(total_eval - user_cut)]]
    user_unknown_test_cut[user] = user_cut
    
torch.save(user_unknown_val_mat.long(), 'lastfm-dataset-1K//user_unknown_val_mat.pt')
torch.save(user_unknown_test_mat.long(), 'lastfm-dataset-1K//user_unknown_test_mat.pt')
torch.save(user_unknown_val_cut.long(), 'lastfm-dataset-1K//user_unknown_val_cut.pt')
torch.save(user_unknown_test_cut.long(), 'lastfm-dataset-1K//user_unknown_test_cut.pt')
    

    
