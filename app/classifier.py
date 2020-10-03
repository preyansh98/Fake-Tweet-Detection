import argparse
import os

import pandas as pd
import numpy as np

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words('english')

import pickle

class ModelWrapper(object):

    def __init__(self, model, count_vectorizer):
        self.model = model
        self.cv = count_vectorizer

    def transform_predict(self, value):
        tokens = word_tokenize(value)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(token) for token in tokens]

        cleaned_tweet = ' '.join(words)

        X = self.cv.transform([cleaned_tweet])
        result = self.model.predict(X)
        return result

def load_dataset(datapath):
    '''
    load and merge the two .csvs
    '''

    fake = pd.read_csv(os.path.join(datapath, 'Fake.csv'))
    true = pd.read_csv(os.path.join(datapath, 'True.csv'))

    fake['fake'] = 1
    true['fake'] = 0

    data = pd.concat([fake, true], ignore_index=True)

    headlines = data['title']
    targets = data['fake']
    
    return headlines, targets

def clean_and_annotate(sents, 
                       targets, 
                       normalization_mode = 'stem',
                       remove_stopwords = False,
                       infreq_word_thresh = 1,
                       count_vectorizer = None):
    '''
    Clean data according to parameters, return 
    
    positional arguments:
    sents: list of sentences
    targets: respective list of targets (0 for negative, 1 for positive)

    Keyword arguments:
    normalization_mode: 'stem' or 'lemma'. Default value: 'stem'
    remove_stopwords: boolean, remove stopwords or not. Default Value: False
    infreq_word_thresh: int, count threshold at which to remove words from corpus (strictly lower). Default value: 1
    '''

    #if normalization_mode != ('stem' or 'lemma'):
    #    raise ValueError("normalization_mode must equal 'stem' or 'lemma'.")

    if normalization_mode == 'stem':
        stemmer = PorterStemmer()
    elif normalization_mode == 'lemma':
        lemmatizer = WordNetLemmatizer()

    if remove_stopwords:
        stop_word_list = stop_words
    else:
        stop_word_list = []

    # transform sentences according to parameters
    cleaned_sents = []
    for sent in sents:
        tokens = word_tokenize(sent)
        if normalization_mode == 'stem':
            words = [stemmer.stem(token) for token in tokens]
        elif normalization_mode == 'lemma':
            words = [lemmatizer.lemmatize(token) for token in tokens]

        cleaned_sents.append(' '.join(words))


    # Create feature vectors, and make targets a numpy array
    if count_vectorizer == None:
        count_vectorizer = CountVectorizer(stop_words = stop_word_list, min_df = infreq_word_thresh)
    X = count_vectorizer.fit_transform(cleaned_sents)
    y = np.array(targets)

    return X,y, count_vectorizer

def train_val_test(X,y):
    '''
    Split up data into train, validation, and test sets.
    The data is shuffled in a pre-determined way so that sets are always the same.

    Sets are determined by order of .pos and .neg files so those should remain completely
    untouched.

    Using this set-up, I can do all my work in train/val and then just switch on test
    once I'm ready to finish the experiment. Only really possible to use test data
    if you intend to use it.

    Positional Arguments:
    X: document-term matrix
    y: array of target values
    '''

    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    # Random state as magic number to avoid screw-ups
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=1 - train_ratio, 
                                                        random_state=42, 
                                                        shuffle=True)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
                                                    test_size=test_ratio/(test_ratio + val_ratio), 
                                                    random_state=42, 
                                                    shuffle=True) 

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', help="path to a folder containing rt-polarity.neg and rt-polarity.pos")
    parser.add_argument('-t', '--test', help="if included, then will run final test set code")

    args = parser.parse_args()


    train_df = pd.read_csv(os.path.join(args.datapath, 'fnn_train.csv'))
    val_df = pd.read_csv(os.path.join(args.datapath, 'fnn_dev.csv'))

    X_train, y_train, cv_train = clean_and_annotate(train_df['statement'], train_df['label_fnn'], normalization_mode = 'lemma', remove_stopwords=True, infreq_word_thresh=5)
    X_val, y_val, cv_val = clean_and_annotate(train_df['statement'], train_df['label_fnn'], normalization_mode = 'lemma', remove_stopwords=True, infreq_word_thresh=5, count_vectorizer = cv_train)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    score = classifier.score(X_val, y_val)

    print("accuracy: " + str(score))

    test = ModelWrapper(classifier, cv_train)
    print(test.transform_predict("Who would have ever thought that @realDonaldTrump MelaniaTrump testing positive for #COVID19 would be the October Surprise for #Election2020 I’ve been covering politics a long time, but I’ve never seen anything like this! #TrumpHasCovid"))

    with open('model.pkl', 'wb') as fp:
        pickle.dump(test, fp)

if __name__ == "__main__":
    main()