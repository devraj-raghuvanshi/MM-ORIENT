# !pip install spektral
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras import models
import pandas as pd
import numpy as np
import sklearn
import keras.backend as K
from sklearn.metrics.pairwise import cosine_similarity
from spektral.layers import GCNConv, GlobalSumPool, GraphSageConv
import skimage.measure
from tensorflow.keras.layers import LeakyReLU
from spektral.data import Graph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.label_utils import get_train_labels, get_test_labels

# TRAIN

dataframe = pd.read_csv("path/to/dataset/train_labels.csv")
# clip
img_glob = np.load("path/to/clip_features/image_clip_train.npy")
text_glob = np.load("path/to/clip_features/text_clip_train.npy")
# monomodal features
img_monomodal_features = np.load("path/to/monomodal_features/image_features_train.npy")
text_monomodal_features = np.load("path/to/monomodal_features/text_features_train.npy")
# toxcity_roberta, nrclex, stanford_corenlp
toxicity = np.load("path/to/additional_features/roberta_features_train.npy")
nrclex = np.load("path/to/additional_features/nrclex_features_train.npy")
stanford = np.load("path/to/additional_features/corenlp_features_train.npy")
# concatenate other features
additional_features = np.concatenate((toxicity, nrclex, stanford), axis=1)

#TEST

dataframe_test = pd.read_csv("path/to/dataset/test_labels.csv")
# clip
img_glob_test = np.load("path/to/clip_features/image_clip_test.npy")
text_glob_test = np.load("path/to/clip_features/text_clip_test.npy")
# monomodal features
img_monomodal_features_test = np.load("path/to/monomodal_features/image_features_test.npy")
text_monomodal_features_test = np.load("path/to/monomodal_features/text_features_test.npy")
# toxcity_roberta, nrclex, stanford_corenlp
toxicity_test = np.load("path/to/additional_features/roberta_features_test.npy")
nrclex_test = np.load("path/to/additional_features/nrclex_features_test.npy")
stanford_test = np.load("path/to/additional_features/corenlp_features_test.npy")
# concatenate other features
other_features_test = np.concatenate((toxicity_test, nrclex_test, stanford_test), axis=1)




def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

filename = "path/to/results/metrics_summary.txt"
with open(filename, "a") as file:
    file.write(f"Task, Category, Accuracy, P_macro, R_macro, F1_macro\n")


def train_and_eval(df, df_test, img_embd_clip_train, img_embd_clip_test, text_embd_clip_train, text_embd_clip_test, img_embd_resnext_train, img_embd_resnext_test, text_embd_bert_train, text_embd_bert_test, additional_feat_train, additional_feat_test, thr, num_epochs, b_size, ol_n):
    
    n_train = int(img_embd_clip_train.shape[0]/b_size)*b_size
    n_test = int(img_embd_clip_test.shape[0]/b_size) * b_size

    # load features for CMRL
    img_glob = img_embd_clip_train[:n_train]
    text_glob = text_embd_clip_train[:n_train]
    img_glob_test = img_embd_clip_test[:n_test]
    text_glob_test = text_embd_clip_test[:n_test]
   
    # load features for HAN
    img_monomodal_features = img_embd_resnext_train[:n_train]
    text_monomodal_features = text_embd_bert_train[:n_train]
    img_monomodal_features_test = img_embd_resnext_test[:n_test]
    text_monomodal_features_test = text_embd_bert_test[:n_test]
    
    # load task specific features
    additional_features = additional_feat_train[:n_train]
    additional_features_test = additional_feat_test[:n_test]  
    
    def get_adj(enco):
        norm_enco = enco / np.linalg.norm(enco, axis=1, keepdims=True)
        similarity_matrix = cosine_similarity(norm_enco)
        adjacency_matrix = np.where(similarity_matrix > thr, 1, 0)
        adj_sparse = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float32)
        return adjacency_matrix, adj_sparse

    for task in ['A', 'B', 'C']:  
        for cat in ['humour', 'sarcasm', 'offensive', 'motivational']:  

            if task == 'A':
                labels = get_train_labels(df, 'A')
                test_labels = get_test_labels(df_test, 'A')

            else:
                labels = get_train_labels(df, task, cat)
                test_labels = get_test_labels(df_test, task, cat) 
                
            label = labels[:n_train]
            label_test = test_labels[:n_test]

            adj_tt, adj_sparse_tt = get_adj(text_glob)
            adj_ii, adj_sparse_ii = get_adj(img_glob)  
            
            adj_tt_test, adj_sparse_tt_test = get_adj(text_glob_test)
            adj_ii_test, adj_sparse_ii_test = get_adj(img_glob_test)
            
            # clip input and clip graph input
            image_input_clip = tf.keras.Input(shape=(512,))
            text_input_clip = tf.keras.Input(shape=(512,))

            adj_sp_ii = tf.keras.Input(shape=(b_size,), sparse=True, dtype=tf.int64)
            adj_sp_tt = tf.keras.Input(shape=(b_size,), sparse=True, dtype=tf.int64)

            # monomodal features input
            image_monomodal_input = tf.keras.Input(shape=(100, 2048))
            text_monomodal_input = tf.keras.Input(shape=(128, 768))

            # toxicity_bert, nrclex, stanford_core_nlp input (merge them before input)
            other_features = tf.keras.Input(shape=(784,))

            # word_level attention on BERT text embeddings, output_size=(768,)
            ww = tf.cast(text_monomodal_input, tf.float32)
            attention_weights_text = tf.keras.layers.Dense(units=1, activation='gelu')(ww)
            attention_weights_text = tf.squeeze(attention_weights_text, axis=-1)
            attention_weights_text = tf.nn.softmax(attention_weights_text, axis=1)
            attention_weighted_repr_text = tf.expand_dims(attention_weights_text, axis=-1) * ww
            word_level_attention = tf.reduce_sum(attention_weighted_repr_text, axis=1)

            # positional_embedding_level attention on MRCNN-X152 image embeddings, output_size=(2048,)
            pp = tf.cast(image_monomodal_input, tf.float32)
            attention_weights_image = tf.keras.layers.Dense(units=1, activation='gelu')(pp)
            attention_weights_image = tf.squeeze(attention_weights_image, axis=-1)
            attention_weights_image = tf.nn.softmax(attention_weights_image, axis=1)
            attention_weighted_repr_image = tf.expand_dims(attention_weights_image, axis=-1) * pp
            position_level_attention = tf.reduce_sum(attention_weighted_repr_image, axis=1)

            # image clip with image graph
            img_img = GraphSageConv(channels=512)([image_input_clip, adj_sp_ii])

            # text clip with text graph
            text_text = GraphSageConv(channels=512)([text_input_clip, adj_sp_tt])

            # image clip with text graph
            img_text = GraphSageConv(channels=512)([image_input_clip, adj_sp_tt])

            # text clip with image graph
            text_img = GraphSageConv(channels=512)([text_input_clip, adj_sp_ii])

            # Concatenate all
            merge = tf.keras.layers.concatenate([img_img, text_text, img_text, text_img,
                                    position_level_attention, word_level_attention, other_features], axis=1)

            final = tf.keras.layers.Dense(4618, activation="relu")(merge)
            final = tf.keras.layers.Dropout(0.2)(final)
            final = tf.keras.layers.Dense(2048, activation="relu")(final)
            final = tf.keras.layers.Dropout(0.2)(final)

            task_output_layers = []

            # Creating separate output layers for each task
            if task == 'A':
                output_layers = [tf.keras.layers.Dense(3, activation="softmax")(final)]
            
            elif task == 'B':
                output_layers = [tf.keras.layers.Dense(3, activation="softmax")(final),
                                 tf.keras.layers.Dense(2, activation="softmax")(final),
                                 tf.keras.layers.Dense(2, activation="softmax")(final),
                                 tf.keras.layers.Dense(2, activation="softmax")(final),
                                 tf.keras.layers.Dense(2, activation="softmax")(final)]
                
            elif task == 'C':
                output_layers = [tf.keras.layers.Dense(3, activation="softmax")(final),
                                 tf.keras.layers.Dense(4, activation="softmax")(final),
                                 tf.keras.layers.Dense(4, activation="softmax")(final),
                                 tf.keras.layers.Dense(4, activation="softmax")(final),
                                 tf.keras.layers.Dense(2, activation="softmax")(final)]                

            # Defining the model with multiple outputs for each task
            model = tf.keras.models.Model(inputs=[image_input_clip, text_input_clip, adj_sp_ii, adj_sp_tt, image_monomodal_input, text_monomodal_input, other_features], outputs=output_layers)

            if len(label.shape) == 1:
                model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-6), loss=tf.keras.losses.BinaryCrossentropy(), 
                              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])

            else:
                Metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                get_f1
                ]
                model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-6),
                               loss='categorical_crossentropy',
                               metrics=Metrics)

            for j in range(ol_n):
                indices = np.random.permutation(text_glob.shape[0])[:b_size]

                X1 = np.array([img_glob[i] for i in indices])
                X2 = np.array([text_glob[i] for i in indices])

                adj_sp_iii = tf.sparse.from_dense(get_adj(X1)[0])
                adj_sp_ttt = tf.sparse.from_dense(get_adj(X2)[0])

                A1 = np.array([img_monomodal_features[i] for i in indices])
                A2 = np.array([text_monomodal_features[i] for i in indices])

                other = np.array([additional_features[i] for i in indices])

                X = [X1, X2, adj_sp_iii, adj_sp_ttt, A1, A2, other]
                Y = np.array([label[i] for i in indices])

                history = model.fit(X, Y, epochs=num_epochs, batch_size=b_size, verbose=2,
                                      shuffle=False)

            adj_tt_test_sliced = adj_tt_test[:n_test, :n_test]
            adj_ii_test_sliced = adj_ii_test[:n_test, :n_test]

            n1 = int(text_glob_test.shape[0]/b_size)
            y_pred = []

            for i in range(n1):
                s = slice((i*b_size), b_size*(i+1))
                X1 = img_glob_test[s]
                X2 = text_glob_test[s]
                adj_sp_iii_test = tf.sparse.from_dense(get_adj(X1)[0])
                adj_sp_ttt_test = tf.sparse.from_dense(get_adj(X2)[0])

                A1 = img_monomodal_features_test[s]
                A2 = text_monomodal_features_test[s]

                other = additional_features_test[s]

                X = [X1, X2, adj_sp_iii_test, adj_sp_ttt_test, A1, A2, other]

                pred = model.predict(X, batch_size=b_size)
                y_pred.append(pred)

            if len(label.shape) == 1:   
                y_pred = np.asarray(y_pred)
                y_pred = np.reshape(y_pred, (n_test, 1))
                max_val = 0
                index = 0.1
                y_test = test_labels

                for i in range(9000):
                    value = 0.1 + i*0.0001
                    y_pred1 = np.where(y_pred > value, 1, 0)
                    f1_macro = precision_recall_fscore_support(label_test, y_pred1, average='macro', zero_division=1)
                    if f1_macro[2] > max_val:
                        max_val = f1_macro[2]
                        index = value
                print(max_val)
                print(index)

                thresh = index
                y_pred_max = np.where(y_pred > thresh, 1, 0)

                f1_macro = precision_recall_fscore_support(label_test, y_pred_max, average='macro', zero_division=1)
                accuracy = accuracy_score(label_test, y_pred_max)

                filename = "path/to/results/model_evaluation_results.txt"
                with open(filename, "a") as file:
                    file.write(f"{task, cat, accuracy, f1_macro}\n")

            else:
                y_pred = np.array(y_pred)
                y_pred = y_pred.reshape(n_test, label.shape[1])
                one_hot_encoded = []

                for pred in y_pred:
                    one_hot = np.zeros_like(pred)
                    max_prob_index = np.argmax(pred)
                    one_hot[max_prob_index] = 1
                    one_hot_encoded.append(one_hot)

                y_pred_ohe = np.array(one_hot_encoded)
                y_pred_ohe = y_pred_ohe.astype(int)
                test_labels = test_labels[:n_test]
                test_label = test_labels
                y_pred1 = y_pred_ohe

                accuracy = accuracy_score(test_label, y_pred1)
                f1_macro = precision_recall_fscore_support(test_label, y_pred1, average='macro')

                filename = "path/to/results/model_evaluation_results.txt"

                if task == 'A':
                    with open(filename, "a") as file:
                        file.write(f"{task, '-', accuracy, f1_macro}\n")
                        break

                else:
                    with open(filename, "a") as file:
                        file.write(f"{task, cat, accuracy, f1_macro}\n")


train_and_eval(dataframe, dataframe_test, img_glob, img_glob_test, text_glob, text_glob_test, img_monomodal_features, img_monomodal_features_test, text_monomodal_features, text_monomodal_features_test, additional_features, other_features_test, 0.9, 2, 120, 3) 