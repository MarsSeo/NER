import numpy as np
#import pandas as pd
import tensorflow as tf
import tensorflow_text as text  # this module is a must when using tfhub's resources
#import tensorflow_models as tfm
#import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import metrics as met
from official.nlp import optimization  # to create AdamW optimizer
#import graphviz
#import pydot
import json
#import GPUtil
import time


from Dataset import load_dataset
from Models import ModelBuilder

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

class Classifier:
    def build_model(this):
        this.train_ds,this.test_ds,this.valid_ds,classdict=load_dataset(this.data_path,this.config.PREPROCESS_METHOD)
        this.train_ds['data']=this.train_ds['data'].shuffle(5000).batch(this.config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        this.valid_ds['data']=this.valid_ds['data'].shuffle(5000).batch(this.config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        this.test_ds['data']=this.test_ds['data'].batch(this.config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        this.invdict=np.zeros(len(classdict)).tolist()
        for k,v in classdict.items():
            this.invdict[v]=k

        tf.keras.backend.clear_session()

        this.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        this.metrics = [tf.metrics.CategoricalAccuracy()]

        steps_per_epoch = tf.data.experimental.cardinality(this.train_ds['data']).numpy()
        num_train_steps = steps_per_epoch * this.config.EPOCHS
        num_warmup_steps = int(0.1*num_train_steps)

        this.optimizer = optimization.create_optimizer(init_lr=this.config.LEARNING_RATE,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')


        this.model = ModelBuilder().build_classifier_model(len(classdict),this.config.ENCODER_TYPE,this.config.EXTRA_LAYERS)
        tf.keras.utils.plot_model(this.model,'results/img/REmodel.png')
        this.model.compile(optimizer=this.optimizer,
              loss=this.loss,
              metrics=this.metrics)
        this.model_name="{}-{}_EP{}_BS{}_LR{}_ML{}".format(this.config.ENCODER_TYPE,this.config.EXTRA_LAYERS,this.config.EPOCHS,this.config.BATCH_SIZE,this.config.LEARNING_RATE,this.config.MAX_LENGTH)
    
    def train_model(this):
        trs_time=time.process_time()
        history=this.model.fit(this.train_ds['data'],validation_data=this.valid_ds['data'],verbose=1,epochs=this.config.EPOCHS)
        tre_time=time.process_time()
        train_time=tre_time-trs_time
        this.history=history
        this.train_time=train_time
        return history,train_time
    
    def plot_history(this,history,save_fig=False,folder_path="results/img/"):
        history_dict = history.history
        print(history_dict.keys())

        metrics_name=[m.name for m in this.metrics]

        fig = plt.figure(figsize=(10, 3*(len(metrics_name)+1)))
        fig.tight_layout()
        epochs = range(1, this.config.EPOCHS + 1)

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
        plt.show()
        if save_fig:
            plt.savefig('{}{}_history.png'.format(folder_path,this.model_name))
        plt.close()
    
    def test_model(this):
        tss_time= time.process_time()
        y_pred=this.model.predict(this.test_ds['X'],batch_size=this.config.BATCH_SIZE).argmax(axis=1)
        tse_time= time.process_time()
        test_time=tse_time-tss_time
        y_true=this.test_ds['Y']
        test_result=evaluation(y_true,y_pred)
        #test_result.setdefault('History',history.history)
        #test_result.setdefault('Time_cost',{'train':train_time,'test':test_time})
        sbn.heatmap(test_result['confusion_matrix'],xticklabels=this.invdict,yticklabels=this.invdict,annot=False)

        print('accuracy: '+str(test_result['accuracy']))
        print('macro: '+str(test_result['Macro']))
        print('micro: '+str(test_result['Micro']))
        print('weighted: '+str(test_result['Weighted']))
        this.test_result=test_result
        this.test_time=test_time
        return test_result,test_time
    
    def save_test_result(this,path):
        config=this.config
        metadata={'model_type':config.ENCODER_TYPE,
                'task_type':config.TASK,
                'epochs':config.EPOCHS,
                'learning_rate':config.LEARNING_RATE,
                'batch_size':config.BATCH_SIZE,
                'max_length':config.MAX_LENGTH,
                'model_name':this.model_name,
                'optimizer':this.optimizer._name,
                'metrics':[m.name for m in this.metrics],
                'loss':this.loss.name,
                'class_name':this.invdict,
                #'is_saved':config.SAVE_MODEL,
                'devices':[gpu.name for gpu in this.GPUtil.getAvailable()],
                #'structure':model.summary(),
                'preprocess_method':config.PREPROCESS_METHOD,
                }
        test_result=this.test_result
        test_result.setdefault('History',this.history.history)
        test_result.setdefault('Time_cost',{'train':this.train_time,'test':this.test_time})

        saved_data={'Metadata':metadata,
            'Result':test_result}
        with open(path+this.model_name+'_test_result.json','w',encoding='UTF-8')as f:
            json.dump(saved_data,f,indent=2)
    
    def save_model_weight(this,path):
        this.model.save('{}{}'.format(path,this.model_name))

    def predict(this,data):
        pred=this.model.predict(data,batch_size=this.config.BATCH_SIZE).argmax(axis=1)
        res=[this.invdict[x] for x in pred]
        return res

    def run_task(this,save_result=False,save_weight=False,result_folder="results/",model_folder="models/"):
        this.build_model()
        this.train_model()
        this.plot_history(this.history)
        this.test_model()
        if save_result:
            this.save_test_result(result_folder)
        if save_weight:
            this.save_model_weight(model_folder)
    
    def __init__(this,data_path,config):
        this.data_path=data_path
        this.config=config
