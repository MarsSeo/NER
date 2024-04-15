import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
#import tensorflow_models as tfm
import tensorflow_hub as hub
import matplotlib.pyplot as plt
#import seaborn as sbn
from sklearn import metrics as met
from official.nlp import optimization  # to create AdamW optimizer
#import graphviz
#import pydot
#import shutil
#import os
import json
import GPUtil
import time

class MaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)
    def call(self, inputs):
        #return inputs[0]
        return {'input_word_ids':inputs[0]['input_word_ids'],'input_type_ids':tf.math.add(inputs[0]['input_type_ids'],inputs[1]),'input_mask':tf.math.add(inputs[0]['input_mask'],tf.math.multiply(inputs[1],-1))}

def build_classifier_model(output_shape,encoder,preprocessor):
    #position_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='mask')
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = preprocessor
    #masklayer = MaskingLayer()
    encoder_inputs = preprocessing_layer(text_input)
    #encoder_inputs = masklayer([encoder_inputs,position_input])
    encoder = encoder
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    #net = tf.keras.layers.Dropout(0.1)(net)
    #net = tf.keras.layers.Dense(128, activation=None, name='shuffler')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(output_shape, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

def evaluation(y_true,y_pred):
    acc=met.accuracy_score(y_true,y_pred)
    p=met.precision_score(y_true,y_pred,average=None,zero_division=True).tolist()
    r=met.recall_score(y_true,y_pred,average=None).tolist()
    f1=met.f1_score(y_true,y_pred,average=None).tolist()
    weighted_p=met.precision_score(y_true,y_pred,average='weighted',zero_division=True)
    weighted_r=met.recall_score(y_true,y_pred,average='weighted')
    weighted_f1=met.f1_score(y_true,y_pred,average='weighted')
    macro_p=met.precision_score(y_true,y_pred,average='macro',zero_division=True)
    macro_r=met.recall_score(y_true,y_pred,average='macro')
    macro_f1=met.f1_score(y_true,y_pred,average='macro')
    micro_p=met.precision_score(y_true,y_pred,average='micro',zero_division=True)
    micro_r=met.recall_score(y_true,y_pred,average='micro')
    micro_f1=met.f1_score(y_true,y_pred,average='micro')
    cmatrix=met.confusion_matrix(y_true,y_pred).tolist()
    return {'accuracy':acc,'confusion_matrix':cmatrix,
            'Categorical':{'precision':p,'recall':r,'f1':f1},'Weighted':{'precision':weighted_p,'recall':weighted_r,'f1':weighted_f1},
            'Macro':{'precision':macro_p,'recall':macro_r,'f1':macro_f1},'Micro':{'precision':micro_p,'recall':micro_r,'f1':micro_f1},}

def modeling(config,datapack):
    tf.keras.backend.clear_session()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.metrics.CategoricalAccuracy()]
    steps_per_epoch = tf.data.experimental.cardinality(datapack.train_ds).numpy()
    num_train_steps = steps_per_epoch * config.EPOCHS
    num_warmup_steps = int(0.1*num_train_steps)

    preprocessors={'bert-base':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                'bert-large':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                'roberta-base':'https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1',
                'roberta-large':'https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1'}
    layers={'bert-base':'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert-large':'https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-l-24-h-1024-a-16/versions/4',
            'roberta-base':'https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1',
            'roberta-large':'https://www.kaggle.com/models/kaggle/roberta/frameworks/tensorFlow2/variations/en-cased-l-24-h-1024-a-16/versions/1'}
    preprocessor = hub.KerasLayer(preprocessors[config.ENCODER_TYPE], name='Preprocessor/Tokenizer')
    bert_layer = hub.KerasLayer(layers[config.ENCODER_TYPE], name='Encoder',trainable=True)
        
    optimizer = optimization.create_optimizer(init_lr=config.LEARNING_RATE,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')
    model = build_classifier_model(len(datapack.reladict),encoder=bert_layer,preprocessor=preprocessor)
    tf.keras.utils.plot_model(model)
    #print(model.summary())
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
    trs_time=time.process_time()
    
    history=model.fit(datapack.train_ds,validation_data=datapack.valid_ds,verbose=1,epochs=config.EPOCHS)
    tre_time=time.process_time()
    train_time=tre_time-trs_time
    
    model_name="{}_EP{}_BS{}_LR{}_ML{}".format(config.ENCODER_TYPE,config.EPOCHS,config.BATCH_SIZE,config.LEARNING_RATE,config.MAX_LENGTH)

    history_dict = history.history
    print(history_dict.keys())
    metrics_name=[m.name for m in metrics]
    fig = plt.figure(figsize=(10, 3*(len(metrics_name)+1)))
    fig.tight_layout()
    epochs = range(1, config.EPOCHS + 1)
    plt.subplot(len(metrics_name)+1, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, history_dict['loss'], 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    for i in range(0,len(metrics_name)):
        plt.subplot(len(metrics_name)+1, 1, i+2)
        plt.plot(epochs, history_dict[metrics_name[i]], 'r', label='Training {}'.format(metrics_name[i]))
        plt.plot(epochs, history_dict['val_{}'.format(metrics_name[i])], 'b', label='Validation {}'.format(metrics_name[i]))
        plt.title('Training and validation {}'.format(metrics_name[i]))
        plt.ylabel(metrics_name[i])
        plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.savefig('results/img/RE/{}_history.png'.format(model_name))
    plt.close()
    

    tss_time= time.process_time()
    y_pred=model.predict(datapack.testx,batch_size=config.BATCH_SIZE).argmax(axis=1)
    tse_time= time.process_time()
    test_time=tse_time-tss_time
    y_true=datapack.testy
    test_result=evaluation(y_true,y_pred)
    test_result.setdefault('History',history.history)
    test_result.setdefault('Time_cost',{'train':train_time,'test':test_time})
    test_result.setdefault('Original',{'y_true':y_true,'y_pred':[int(x) for x in y_pred]})
    #matrix=tf.math.confusion_matrix(y_true,y_pred).numpy()
    #sbn.heatmap(test_result['confusion_matrix'],xticklabels=invdict,yticklabels=invdict,annot=False)
        
    metadata={'model_type':config.ENCODER_TYPE,
            'task_type':config.TASK,
            'epochs':config.EPOCHS,
            'learning_rate':config.LEARNING_RATE,
            'batch_size':config.BATCH_SIZE,
            'max_length':config.MAX_LENGTH,
            'model_name':model_name,
            'optimizer':optimizer._name,
            'metrics':[m.name for m in metrics],
            'loss':loss.name,
            'class_name':datapack.invdict,
            'is_saved':config.SAVE_MODEL,
            'devices':[gpu.name for gpu in GPUtil.getAvailable()],
            'structure':model.summary()
            }
    
    if config.SAVE_MODEL:
        model.save('models/{}'.format(model_name))
    saved_data={'Metadata':metadata,
                'Result':test_result}
    with open('results/NRE_test_result.json','r',encoding='UTF-8')as f:
        s=json.load(f)
    s.setdefault('task',config.TASK)
    s.setdefault('Results',[])
    s['Results'].append(saved_data)
    with open('results/NRE_test_result.json','w',encoding='UTF-8')as f:
        json.dump(s,f,indent=2)
    
    return model,saved_data