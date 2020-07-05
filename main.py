##################
# Name: Gal Ziv
# ID:   205564198
##################

import nltk
import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from normalise import normalise
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm, preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize.punkt import PunktSentenceTokenizer as PST
from nltk import FreqDist
from keras import Sequential
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from textblob import TextBlob
from imblearn.over_sampling import SMOTE


# TARGET = 'Category'
# BODY = 'Message'
TARGET = 'target'
BODY = 'text'
USE_FEATURES = False
HANDLE_IMBALANCE = False


def get_all_words(words_col_df):
    for words in words_col_df:
        for word in words:
            yield word


#########################################################################
### Preprocessing functions
def to_lowercase(row):
    if type(row) is str:
        return row.lower()
    else:
        row[BODY] = row[BODY].lower()
        row['Words'] = [word.lower() for word in row['Words']]
        return row


def remove_punctuation(row):
    if type(row) is str:
        return row.translate(str.maketrans('', '', string.punctuation))
    else:
        row[BODY] = row[BODY].translate(str.maketrans('', '', string.punctuation))
        row['Words'] = [word.translate(str.maketrans('', '', string.punctuation)) for word in row['Words']]
        return row


def remove_numbers(row):
    if type(row) is str:
        return ''.join(i for i in row if not i.isdigit())
    else:
        row[BODY] = ''.join(i for i in row[BODY] if not i.isdigit())
        row['Words'] = [''.join(i for i in word if not i.isdigit()) for word in row['Words']]
        return row


def remove_special_characters(row):
    if type(row) is str:
        return ''.join(c for c in row if (c.isalnum() or c == ' '))
    else:
        row[BODY] = ''.join(c for c in row[BODY] if (c.isalnum() or c == ' '))
        row['Words'] = [''.join(c for c in word if (c.isalnum() or c == ' ')) for word in row['Words']]
        return row


# As of now this is NOT WORKING WELL
def stopwords_cleaner(row):
    stopwords = nltk.corpus.stopwords.words('english')

    if type(row) is str:
        words = nltk.word_tokenize(row)
        return [word for word in words if word not in stopwords]
    else:
        row['Words'] = [word for word in row['Words'] if word not in stopwords]
        row[BODY] = ' '.join(row['Words'])
        return row


# def stem_text(row):
#     row['Words'] = [ps.stem(word) for word in row['Words']]
#     row[BODY] = ' '.join(row['Words'])
#     return row


def stem_text(row):
    stem_words = np.vectorize(ps.stem)
    row['Words'] = stem_words(row['Words'])
    row[BODY] = ' '.join(row['Words'])
    return row


def lemm_text(row):
    lemm_words = np.vectorize(wnl.lemmatize)
    row['Words'] = lemm_words(row['Words'])
    row[BODY] = ' '.join(row['Words'])
    return row


def normalise_text(row):
    # try:
    row['Words'] = normalise(row['Words'], verbose=False)
    row[BODY] = ' '.join(row['Words'])
    # except IndexError as e:
    #     print(e)

    return row


def normalize_columns(df_cols):
    names = df_cols.columns
    x = df_cols.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    df_normalized.columns = names
    return df_normalized

def standardize_columns(df_cols):
    names = df_cols.columns
    x = df_cols.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    df_standardized = pd.DataFrame(x_scaled)
    df_standardized.columns = names
    return df_standardized


def preprocess(df):
    df = df.apply(remove_punctuation, axis=1)
    df = df.apply(remove_special_characters, axis=1)
    # df = df.apply(normalise_text, axis=1)
    df = df.apply(remove_numbers, axis=1)
    df = df.apply(to_lowercase, axis=1)
    df = df.apply(stopwords_cleaner, axis=1)
    df = df.apply(stem_text, axis=1)
    # df = df.apply(lemm_text, axis=1)
    return df

#######################################################################
### Feature Creation functions
def words_count(row):
    words_no_punct = [word for word in row['Words'] if not any(char in set(string.punctuation) for char in word)]
    return len(words_no_punct)


def characters_count(row):
    return len(row[BODY])


def sentences_count(row):
    sentences = [sentence for sentence in pst.sentences_from_text(row[BODY], False) if sentence not in string.punctuation]
    return len(sentences)


def punct_count(row):
    puncts = [c for c in row[BODY] if c in string.punctuation]
    return len(puncts)


def all_caps_count(row):
    all_caps_words = [word for word in row['Words'] if (
        word.isupper() and len(remove_punctuation(word)) > 1 and not bool(
            re.search('(24:00|2[0-3]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9])', word)))]
    return len(all_caps_words)


def stopwords_count(row):
    stopwords = nltk.corpus.stopwords.words('english')
    stops = [word for word in row['Words'] if word in stopwords]
    return len(stops)


def mean_length(row):
    return sum(map(len, row['Words'])) / len(row['Words'])


def comma_count(row):
    return row[BODY].count(",")


def qmark_count(row):
    return row[BODY].count("?")


def excmark_count(row):
    return row[BODY].count("!")


def quotes_count(row):
    return row[BODY].count('\"')


def currency_count(row):
    curr = [c for c in row[BODY] if c in "$£€₦₨￥"]
    return len(curr)


def get_sentiment(row):
    return TextBlob(row[BODY]).sentiment


# tODO: remove spaces throughout, both in body and in words
def create_features(df):
    df['Words_Count'] = df.apply(words_count, axis=1)
    df['Chars_Count'] = df.apply(characters_count, axis=1)
    df['Sents_Count'] = df.apply(sentences_count, axis=1)
    df['Punct_Count'] = df.apply(punct_count, axis=1)
    df['All_Caps_Count'] = df.apply(all_caps_count, axis=1)
    df['StopWords_Count'] = df.apply(stopwords_count, axis=1)
    df['MeanLength'] = df.apply(mean_length, axis=1)
    df['Comma_Count'] = df.apply(comma_count, axis=1)
    df['QMark_Count'] = df.apply(qmark_count, axis=1)
    df['ExcMark_Count'] = df.apply(excmark_count, axis=1)
    df['Quotes_Count'] = df.apply(quotes_count, axis=1)
    df['Currency_Count'] = df.apply(currency_count, axis=1)

    df['Sentiment_Score'] = df.apply(get_sentiment, axis=1)
    sentiment_series = df['Sentiment_Score'].tolist()
    columns = ['Polarity', 'Subjectivity']
    temp_df = pd.DataFrame(sentiment_series, columns=columns, index=df.index)
    df['Polarity'] = temp_df['Polarity']
    df['Subjectivity'] = temp_df['Subjectivity']
    df = df.drop(['Sentiment_Score'], axis=1)
    return df


def select_features(X_train, y_train):
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestClassifier(n_estimators=1000))
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    # X_train_fs_df = pd.DataFrame(fs.transform(X_train))
    cols = fs.get_support(indices=True)
    features_df_fs = X_train.iloc[:, cols]
    # transform test input data
    # X_test_fs = fs.transform(X_test)
    return features_df_fs


def handle_imbalance(x_train, y_train):
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

    return x_train_res, y_train_res


# Training models
def RNN(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        Bidirectional(LSTM(embedding_dim)),
        # layer = LSTM(64)(layer)
        Dense(embedding_dim, name='FC1'),
        # layer = Dense(256, name='FC1')(layer)
        Activation('relu'),
        Dropout(0.5),
        Dense(1, name='out_layer'),
        Activation('sigmoid')
    ])
    # inputs = Input(name='inputs', shape=[maxlen])
    model.summary()
    # model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



ps = nltk.PorterStemmer()
wnl = WordNetLemmatizer()
pst = PST()

# data = pd.read_csv("spam_text.csv")
data = pd.read_csv("train.csv")
data = data[[BODY, TARGET]]
data[BODY] = data[BODY].str.strip()
data['Words'] = data[BODY].apply(nltk.word_tokenize)
data['tempMessage'] = data[BODY]
data['tempWords'] = data['Words']
data = create_features(data)
data = preprocess(data)
data = data.rename(columns={BODY: "PP_Message", "tempMessage": BODY, "Words": "PP_Words", "tempWords": "Words"})

# Encode target variable
# data[TARGET] = data[TARGET].replace({'ham': 0}, {'spam': 1})

### Visualization and exploration
# sns.countplot(data[TARGET])
# plt.xlabel('Label')
# plt.title('Label categories')
# plt.show()

# all_words = get_all_words(data['PP_Words'])
# freq_dist_pos = FreqDist(all_words)
# print(freq_dist_pos.most_common(10))
###

data.to_csv("temp.csv")
data = data.drop([BODY, 'Words'], axis=1)
data_custom_feats = data.drop(['PP_Message', 'PP_Words', TARGET], axis=1)
data_custom_feats_norm = normalize_columns(data_custom_feats)
data_custom_feats_fs = select_features(data_custom_feats_norm, data[TARGET])
data_text = data[['PP_Message', 'PP_Words', TARGET]]
data = data_text.join(data_custom_feats_fs)
# data = data_text.join(standardize_columns(data_custom_feats))

le = LabelEncoder()
Y = data[TARGET]
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

###### IMPORTANT TODO: Use stack ensemble to incorporate created features with classifications of binary-input BoW etc. models
X = data.drop(['PP_Words', TARGET], axis=1)
# X = data['PP_Message']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train_text = X_train['PP_Message']
X_train_feats = X_train.drop('PP_Message', axis=1)
X_test_text = X_test['PP_Message']
X_test_feats = X_test.drop('PP_Message', axis=1)

max_words = 5000
max_len = 150

Tfidf_vect = TfidfVectorizer(max_features=max_words)
Tfidf_vect.fit(data['PP_Message'])
if USE_FEATURES:
    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)
else:
    Train_X_Tfidf = Tfidf_vect.transform(X_train_text)
    Test_X_Tfidf = Tfidf_vect.transform(X_test_text)

# Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Y_train)
predictions_NB = Naive.predict(Test_X_Tfidf)
nb_accr = accuracy_score(predictions_NB, Y_test)

svm_param_grid = dict(C=[0.01, 0.1, 1, 10],
                  kernel=['linear', 'rbf', 'sigmoid'],
                  degree=[2, 3, 5, 10],
                  gamma=['auto', 'scale', 0.1, 1, 10])
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf, Y_train)
svm_grid = RandomizedSearchCV(estimator=svm.SVC(), param_distributions=svm_param_grid, cv=5, verbose=1, n_iter=5, refit=True)
svm_grid_result = svm_grid.fit(Train_X_Tfidf, Y_train)
# predict the labels on validation dataset
predictions_SVM = svm_grid.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
svm_accr = accuracy_score(predictions_SVM, Y_test)

# Neural network
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train_text)
sequences = tok.texts_to_sequences(X_train_text)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

if USE_FEATURES:
    feats_matrix = np.float_(X_train_feats.to_numpy())
    full_matrix = np.concatenate((sequences_matrix, feats_matrix), axis=1)
else:
    full_matrix = sequences_matrix

if HANDLE_IMBALANCE:
    full_matrix, Y_train = handle_imbalance(full_matrix, Y_train)

param_grid = dict(vocab_size=[max_words],
                  embedding_dim=[50],
                  maxlen=[np.size(full_matrix, 1)])

model = KerasClassifier(build_fn=RNN, epochs=10, batch_size=128, verbose=True)
# model = RNN()
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
# model.fit(full_matrix, Y_train, batch_size=128, epochs=10,
#           validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

nn_grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
nn_grid_result = nn_grid.fit(full_matrix, Y_train)

X_test_text = X_test['PP_Message']
X_test_feats = X_test.drop('PP_Message', axis=1)
test_sequences = tok.texts_to_sequences(X_test_text)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
if USE_FEATURES:
    test_feats_matrix = np.float_(X_test_feats.to_numpy())
    test_full_matrix = np.concatenate((test_sequences_matrix, test_feats_matrix), axis=1)
else:
    test_full_matrix = test_sequences_matrix


test_pred = nn_grid.predict(test_full_matrix)
# accr = model.evaluate(test_full_matrix, Y_test)
nn_accr = accuracy_score(Y_test, test_pred)

print(nn_accr)
