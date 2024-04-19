import json
import tensorflow as tf
import numpy as np

def preprocess(data,reladict={},shuffle_size=0):# -> dict[str, Any]:
    sent=[]
    relation=[]
    for d in data:
        token=d['token']
        token[d['h']['pos'][0]]='['+token[d['h']['pos'][0]]
        token[d['h']['pos'][1]]=token[d['h']['pos'][1]]+']'
        token[d['t']['pos'][0]]='['+token[d['t']['pos'][0]]
        token[d['t']['pos'][1]]=token[d['t']['pos'][1]]+']'
        
        sent.append([' '.join(token)])
        relation.append(reladict[d['relation']])
    temp=np.zeros(shape=(len(relation),len(reladict)))
    for i in range(len(relation)):
        temp[i][relation[i]]=1
    if shuffle_size==0:
        data=tf.data.Dataset.from_tensor_slices((sent,temp))#.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        data=tf.data.Dataset.from_tensor_slices((sent,temp)).shuffle(shuffle_size)#.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return {'data':data,'X':sent,'Y':relation}

def load_dataset(path,conj_method=None):# -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict]:
    with open(path['train'],'r',encoding='UTF-8') as f:
        trdata=[json.loads(line) for line in f]
    with open(path['test'],'r',encoding='UTF-8') as f:
        tsdata=[json.loads(line) for line in f]
    with open(path['valid'],'r',encoding='UTF-8') as f:
        valdata=[json.loads(line) for line in f]
    reladict={}
    for sets in [trdata,tsdata,valdata]:
        for row in sets:
            if reladict.get(row['relation'],False)==False:
                reladict.setdefault(row['relation'],len(reladict))
    
    test_ds=preprocess(tsdata,reladict)
    valid_ds=preprocess(valdata,reladict,shuffle_size=5000)
    train_ds=preprocess(trdata,reladict,shuffle_size=5000)
    return train_ds,test_ds,valid_ds,reladict
