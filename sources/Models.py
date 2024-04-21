import tensorflow as tf
import tensorflow_hub as hub

class MaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)
    def call(self, inputs):
        #return inputs[0]
        return {'input_word_ids':inputs[0]['input_word_ids'],'input_type_ids':tf.math.add(inputs[0]['input_type_ids'],inputs[1]),'input_mask':tf.math.add(inputs[0]['input_mask'],tf.math.multiply(inputs[1],-1))}


class ModelBuilder:
    preprocessors={'bert-base':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
               'bert-large':'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
               'roberta-base':'https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1',
               'roberta-large':'https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1'}
    encoders={'bert-base':'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert-large':'https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-l-24-h-1024-a-16/versions/4',
            'roberta-base':'https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1',
            'roberta-large':'https://www.kaggle.com/models/kaggle/roberta/frameworks/tensorFlow2/variations/en-cased-l-24-h-1024-a-16/versions/1'}

    def load_tokenizer(this,type):
        return hub.KerasLayer(this.preprocessors[type], name='Preprocessor/Tokenizer')
    
    def load_encoder(this,type):
        return hub.KerasLayer(this.encoders[type], name='Encoder',trainable=True)
    
    def load_extra_layers(this,type):
        if type=='BiLSTM':
            #outputs=tf.keras.layers.LSTM(128,input_shape=inputs.shape,return_sequences=True)(inputs,)
            #outputs=tf.keras.layers.LSTM(units=128,go_backwards=True)(outputs)
            return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True))
        else:
            raise Exception('Extra layers type: "{}" does not exists.'.format(type))

    def build_classifier_model(this,output_shape,encoder_type,extra_layers=''):
        #position_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='mask')
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = this.load_tokenizer(encoder_type)
        #masklayer = MaskingLayer()
        encoder_inputs = preprocessing_layer(text_input)
        #encoder_inputs = masklayer([encoder_inputs,position_input])
        encoder = this.load_encoder(encoder_type)
        outputs = encoder(encoder_inputs)
        if extra_layers == '' or extra_layers == 'None':
            net = outputs['pooled_output']
        else:
            net=this.load_extra_layers(extra_layers)(outputs['sequence_output'])
            net=tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(output_shape, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)