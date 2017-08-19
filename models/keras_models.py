from funcy import pluck
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import (Activation, Conv1D, Dense, Dropout, Embedding, LSTM, Bidirectional,
                          Flatten, Input, MaxPooling1D, concatenate)
from keras.regularizers import L1L2
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model

from feature_engineering import DocToWordIndices


class _KerasModel(object):

    def __init__(self, train, test):
        self.x_train, self.x_test = pluck('content', train), pluck('content', test)
        self.y_train, self.y_test = pluck('label', train), pluck('label', test)

        self.train_ids = pluck('id', train)
        self.test_ids = pluck('id', test)

        self.transform = DocToWordIndices().fit(self.x_train)

    def fit(self, batch_size, epochs, save_best_model_to_filepath=None):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=[self.x_test, self.y_test])


class BidirectionalLSTM(object):

    def __init__(self, train, test, embedding_size=128, lstm_cell_size=64, dropout_prob=.5, kernel_l2=0):

        self.x_train, self.x_test = pluck('content', train), pluck('content', test)
        self.y_train, self.y_test = pluck('label', train), pluck('label', test)

        self.train_ids = pluck('id', train)
        self.test_ids = pluck('id', test)

        self.transform = DocToWordIndices().fit(self.x_train)
        self.x_train = self.transform.transform(self.x_train)
        self.x_test = self.transform.transform(self.x_test)

        vocab_size = np.max(self.x_train) + 1  # vocab and classes are 0 indexed
        n_labels = int(np.max(self.y_train)) + 1
        self.y_train, self.y_test = to_categorical(self.y_train), to_categorical(self.y_test)

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_size, input_length=self.x_train.shape[1]))
        self.model.add(Bidirectional(LSTM(lstm_cell_size, kernel_regularizer=L1L2(l1=0.0, l2=kernel_l2))))
        self.model.add(Dropout(dropout_prob))
        self.model.add(Dense(n_labels, activation='softmax'))
        self.model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batch_size, epochs, save_best_model_to_filepath=None):
        checkpoint = ModelCheckpoint(save_best_model_to_filepath,
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        # Fit the model
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[checkpoint] if save_best_model_to_filepath is not None else [],
                       validation_data=[self.x_test, self.y_test])
        return self.model

    def get_predict_proba(self):
        """Return a function that goes from docs to class probabilities"""
        def predict_proba(examples):
            x = self.transform.transform(examples)
            return self.model.predict(x)
        return predict_proba

class ConvNet(object):

    def __init__(self, train, test, **model_options):
        embedding_size = model_options.get('embedding_size', 128)
        filter_sizes = model_options.get('filter_sizes', [3, 4, 5])
        n_filters = model_options.get('n_filters', 50)

        hidden_dims = model_options.get('hidden_dims', 128)

        dropout_prob = model_options.get('dropout_prob', .5)

        conv_l2 = model_options.get('conv_l2', .05)
        fc_l2 = model_options.get('fc_l2', .05)



        self.x_train, self.x_test = pluck('content', train), pluck('content', test)
        self.y_train, self.y_test = pluck('label', train), pluck('label', test)

        self.train_ids = pluck('id', train)
        self.test_ids = pluck('id', test)

        self.transform = DocToWordIndices().fit(self.x_train)
        self.x_train = self.transform.transform(self.x_train)
        self.x_test = self.transform.transform(self.x_test)

        self.vocab_size = np.max(self.x_train) + 1  # vocab and classes are 0 indexed
        self.n_labels = int(np.max(self.y_train)) + 1
        self.y_train, self.y_test = to_categorical(self.y_train), to_categorical(self.y_test)

        self.sequence_length = self.x_train.shape[1]
        self.n_labels = self.y_train.shape[1]

        graph_in = Input(shape=(self.sequence_length, embedding_size))
        convs = []
        for filter_size in filter_sizes:
            conv = Conv1D(activation="relu", padding="valid",
                          strides=1, filters=n_filters, kernel_size=filter_size,
                          kernel_regularizer=L1L2(l1=0.0, l2=conv_l2))(graph_in)
            pool = MaxPooling1D(pool_size=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)

        if len(filter_sizes) > 1:
            out = concatenate(convs)
        else:
            out = convs[0]

        graph = Model(inputs=graph_in, outputs=out)

        # main sequential model
        model = Sequential()
        model.add(Embedding(self.vocab_size, embedding_size, input_length=self.sequence_length,
                            weights=None))

        # model.add(Dropout(dropout_prob, input_shape=(self.sequence_length, embedding_size)))
        model.add(graph)
        model.add(Dense(hidden_dims, kernel_regularizer=L1L2(l1=0.0, l2=fc_l2)))
        model.add(Dropout(dropout_prob))
        model.add(Activation('relu'))
        model.add(Dense(self.n_labels, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])