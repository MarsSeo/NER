import json
import tensorflow as tf
import numpy as np

def preprocessM(data,reladict={},shuffle_size=0):# -> dict[str, Any]:
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

def preprocessMdiff(data,reladict={},shuffle_size=0):# -> dict[str, Any]:
    sent=[]
    relation=[]
    for d in data:
        token=d['token']
        
        token[d['h']['pos'][0]]='['+token[d['h']['pos'][0]]
        token[d['h']['pos'][1]]=token[d['h']['pos'][1]]+']'
        token[d['t']['pos'][0]]='<'+token[d['t']['pos'][0]]
        token[d['t']['pos'][1]]=token[d['t']['pos'][1]]+'>'
        
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

def preprocess(data,reladict={},shuffle_size=0):
    sent=[]
    relation=[]
    for d in data:
        token=d['token']

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

def preprocessR(data,reladict={},shuffle_size=0):
    sent=[]
    relation=[]
    for d in data:
        token=d['token']
        
        for i in range(d['h']['pos'][0],d['h']['pos'][1]):
            token[i]='[MASK]'
        for i in range(d['t']['pos'][0],d['t']['pos'][1]):
            token[i]='[MASK]'
        
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

def preprocessE(data,reladict={},shuffle_size=0):
    sent=[]
    relation=[]
    for d in data:
        token=d['token']
        
        for i in range(d['h']['pos'][0],d['h']['pos'][1]):
            token[i]='*'+token[i]
        for i in range(d['t']['pos'][0],d['t']['pos'][1]):
            token[i]='*'+token[i]
        
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
    
    if conj_method=='None':
        test_ds=preprocess(tsdata,reladict)
        valid_ds=preprocess(valdata,reladict,shuffle_size=5000)
        train_ds=preprocess(trdata,reladict,shuffle_size=5000)
    elif conj_method=='Replace':
        test_ds=preprocessR(tsdata,reladict)
        valid_ds=preprocessR(valdata,reladict,shuffle_size=5000)
        train_ds=preprocessR(trdata,reladict,shuffle_size=5000)
    elif conj_method=='Marking':
        test_ds=preprocessM(tsdata,reladict)
        valid_ds=preprocessM(valdata,reladict,shuffle_size=5000)
        train_ds=preprocessM(trdata,reladict,shuffle_size=5000)
    elif conj_method=='MarkingEvery':
        test_ds=preprocessE(tsdata,reladict)
        valid_ds=preprocessE(valdata,reladict,shuffle_size=5000)
        train_ds=preprocessE(trdata,reladict,shuffle_size=5000)
    elif conj_method=='MarkingDiff':
        test_ds=preprocessMdiff(tsdata,reladict)
        valid_ds=preprocessMdiff(valdata,reladict,shuffle_size=5000)
        train_ds=preprocessMdiff(trdata,reladict,shuffle_size=5000)
    else:
        raise Exception('Method "{}" not found.'.format(conj_method))
    return train_ds,test_ds,valid_ds,reladict
