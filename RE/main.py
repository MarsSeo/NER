#if this python file encountered an error, plz run Relation_Analyze.ipynb instead. notice for some reason the output format may be different.
#plz use tensorflow<=2.10 environment for windows, since module later version tensorflow_text is not avaliable on pip. 
import tensorflow as tf
from sources.Relation_classify import Classifier

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 6GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



class CONFIG: 
    #EPS = 1e-8 
    EPOCHS = 8 # 3~8
    BATCH_SIZE = 16 # 8, 32 try lower if OOM
    LEARNING_RATE = 3e-5 # 1e-5
    MAX_LENGTH = 128 # 256
    ENCODER_TYPE = 'bert-large' # bert-base, bert-large, roberta-base, roberta-large 
    EXTRA_LAYERS = 'BiLSTM' # None, BiLSTM
    OPTIMIZER ='adamw'
    PREPROCESS_METHOD='Marking' # None, Replace, Marking, MarkingDiff, Marking Every   
    DEVICE_TYPE = "cuda" # Cuda or alternative
    #SAVE_MODEL=True
    TASK='RE-classification'
config=CONFIG()

# file path of dataset.
data_path={'train':'data/semeval_train.txt',
           'test':'data/semeval_test.txt',
           'valid':'data/semeval_val.txt'}

classifier=Classifier(data_path,config)

# this is for whole training and testing process

#classifier.run_task()


# if there's already trained model, run this.
# note loaded model does not contain optimizer. use .create_optimizer() method to create one.
classifier.load_and_test()


# this is an interface for predicting.

print(classifier.predict(['the most common [ audits ] were about [ waste ] and recycling.']))

