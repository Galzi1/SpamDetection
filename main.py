import nltk
import pandas as pd
import re
import string
from normalise import normalise


TARGET = 'Category'
BODY = 'Message'


# Preprocessing functions
def to_lowercase(row):
    row[BODY] = row[BODY].lower()
    return row


def remove_punctuation(row):
    row[BODY] = row[BODY].translate(str.maketrans('', '', string.punctuation))
    return row


def remove_numbers(row):
    row[BODY] = ''.join(i for i in row[BODY] if not i.isdigit())
    return row


# As of now this is NOT WORKING WELL
def stopwords_cleaner(row):
    stopwords = nltk.corpus.stopwords.words('english')
    row['Words'] = [word for word in row['Words'] if word not in stopwords]
    row[BODY] = ' '.join(row['Words'])
    return row


def stem_text(row):
    row['Words'] = [ps.stem(word) for word in row['Words']]
    row[BODY] = ' '.join(row['Words'])
    return row


def normalise_text(row):
    # try:
    norm_words = normalise(row['Words'], verbose=False)
    row[BODY] = ' '.join(norm_words)
    # except IndexError as e:
    #     print(e)

    return row


def preprocess(df):
    df = df.apply(remove_punctuation, axis=1)
    # df = df.apply(normalise_text, axis=1)
    df = df.apply(remove_numbers, axis=1)
    df = df.apply(to_lowercase, axis=1)
    df = df.apply(stopwords_cleaner, axis=1)
    df = df.apply(stem_text, axis=1)
    return df


# Feature Creation functions
def words_counter(row):
    return len(row['Words'])


def characters_counter(row):
    return len(row[BODY])


def sentences_counter(row, pst):
    sentences = [sentence for sentence in pst.sentences_from_text(row[BODY], False) if not sentence in string.punctuation]
    return len(sentences)


def punct_counter(row):
    puncts = [c for c in row[BODY] if c in string.punctuation]
    return len(puncts)


def all_caps_counter(row):
    all_caps_words = [word for word in row['Words'] if (
                word.isupper() and len(remove_punctuation(word)) > 1 and not bool(
            re.search('(24:00|2[0-3]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9])', word)))]
    return len(all_caps_words)


def stopwords_counter(row):
    stopwords = nltk.corpus.stopwords.words('english')
    stops = [word for word in row['Words'] if word in stopwords]
    return len(stops)


# As of now this is NOT WORKING WELL
# def mean_length(text):
#     text = nltk.word_tokenize(text)
#     return (sum(map(len, text)) / len(text))

def mean_length(row):
    return sum(map(len, row[BODY])) / len(row[BODY])


def comma_counter(row):
    return row[BODY].count(",")


def qmark_counter(row):
    return row[BODY].count("?")


def excmark_counter(row):
    return row[BODY].count("!")


ps = nltk.PorterStemmer()

data = pd.read_csv("spam_text.csv")
data['Message'] = data['Message'].str.strip()
data['Words'] = data['Message'].apply(nltk.word_tokenize)
data['tempMessage'] = data['Message']
data['tempWords'] = data['Words']
data = preprocess(data)
data = data.rename(columns={"Message": "PP_Message", "tempMessage": "Message", "Words": "PP_Words", "tempWords": "Words"})
print(data.head())
