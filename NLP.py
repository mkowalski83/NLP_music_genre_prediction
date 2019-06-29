#!/usr/bin/env python
# coding: utf-8

# In[195]:


import pandas as pd
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np


#nltk.download('stopwords')

class Lyrics_Data:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)#nrows=100000
        self.data = self.data[['genre','lyrics']].dropna()
        
    def get_data(self):
        return self.data
    
    def lyrics_preprocessing(self):
        
        #remove new line
        self.data['lyrics'] = self.data['lyrics'].map(lambda lyrics: re.sub(r'\s+', ' ', lyrics))
        
        #remove numbers
        self.data['lyrics'] = self.data['lyrics'].map(lambda lyrics: re.sub(r'\d+', ' ', lyrics))
        
        #lowercase
        self.data['lyrics'] = self.data['lyrics'].map(lambda lyrics: lyrics.lower())
        
        #remove punctuation
        self.data['lyrics'] = self.data['lyrics'].map(lambda lyrics: lyrics.translate(str.maketrans('', '', string.punctuation)))
        
        #remove whitespace
        self.data['lyrics'] = self.data['lyrics'].map(lambda lyrics: lyrics.strip())
        
        #Tokenize
        self.tokenize()
        
        #remove stop words
        self.remove_stop_words()
        
        #stemming
        self.stemmer()
        
        #Lemmatization
        self.lemmatizer()
        
        #join
        for i in self.data.index:
            self.data.at[i, 'lyrics'] = ' '.join(self.data.at[i,'lyrics'])
        
    
    def lemmatizer(self):
        lemmatizer=WordNetLemmatizer()
        words = self.data.at[i, 'lyrics']
        lemmatized = []
        for word in words:
            lemmatized.append(lemmatizer.lemmatize(word))
        
    def stemmer(self):
        stemmer= PorterStemmer()
        words = self.data.at[i, 'lyrics']
        stemmed = []
        for word in words:
            stemmed.append(stemmer.stem(word))
        
    def tokenize(self):
        for i in self.data.index:
            word_tokens = word_tokenize(self.data.at[i, 'lyrics'])
            self.data.at[i, 'lyrics'] = word_tokens
            
    def remove_stop_words(self):
        stop_words = set(stopwords.words('english')) 
        stopWords = []
        for word in stop_words:
            w = word.translate(str.maketrans('', '', string.punctuation))
            stopWords.append(w)
        stopWords=set(stopWords)
            
        for i in self.data.index:
            word_tokens = self.data.at[i, 'lyrics']
            filtered_sentence = [w for w in word_tokens if not w in stop_words]  
            filtered_sentence = [] 
            
            for w in word_tokens: 
                if w not in stop_words: 
                    filtered_sentence.append(w) 
            self.data.at[i, 'lyrics'] = filtered_sentence
        
   

    
data = Lyrics_Data('lyrics.csv')


# In[196]:


#data.lyrics_preprocessing()
d = data.get_data()
d


# In[197]:




train_x, valid_x, train_y, valid_y = model_selection.train_test_split(d['lyrics'], d['genre'])


# In[199]:



text_clf = Pipeline(
    [('vect', CountVectorizer()),
     ('clf', MultinomialNB(alpha=0.1))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[200]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', MultinomialNB(alpha=0.1))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[224]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', MultinomialNB(alpha=0.2))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[225]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', MultinomialNB(alpha=0.08))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[226]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer()),
     ('clf', MultinomialNB(alpha=0.15))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[227]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer(max_df=0.4,min_df=4)),
     ('clf', MultinomialNB(alpha=0.15))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[230]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer(max_df=0.4,min_df=4)),
     ('clf', MultinomialNB(alpha=0.05))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[229]:


text_clf = Pipeline(
    [('vect', TfidfVectorizer(max_df=0.4,min_df=4)),
     ('clf', MultinomialNB(alpha=0.08))])

# train our model on training data
text_clf.fit(train_x, train_y)  

# score our model on testing data
predicted = text_clf.predict(valid_x)
np.mean(predicted == valid_y)


# In[231]:


from sklearn.metrics import precision_recall_fscore_support
genres = list(d['genre'].unique())

precision, recall, fscore, support = precision_recall_fscore_support(valid_y, predicted)
for n,genre in enumerate(genres):
    genre = genre.upper()
    print(genre+'_precision: {}'.format(precision[n]))
    print(genre+'_recall: {}'.format(recall[n]))
    print(genre+'_fscore: {}'.format(fscore[n]))
    print(genre+'_support: {}'.format(support[n]))
    print()


# In[233]:


from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', TfidfVectorizer(max_df=0.4,min_df=4)),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
])

text_clf_svm.fit(train_x, train_y)
predicted_svm = text_clf_svm.predict(valid_x)
print(np.mean(predicted_svm == valid_y))


# In[234]:


precision, recall, fscore, support = precision_recall_fscore_support(valid_y, predicted)
for n,genre in enumerate(genres):
    genre = genre.upper()
    print(genre+'_precision: {}'.format(precision[n]))
    print(genre+'_recall: {}'.format(recall[n]))
    print(genre+'_fscore: {}'.format(fscore[n]))
    print(genre+'_support: {}'.format(support[n]))
    print()


# In[236]:


from sklearn.ensemble import RandomForestClassifier
text_clf = Pipeline([('vect', TfidfVectorizer(max_df=0.4,min_df=4)),
                         ('clf', RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
        ])

text_clf.fit(train_x, train_y)
predicted_svm = text_clf.predict(valid_x)
print(np.mean(predicted_svm == valid_y))

