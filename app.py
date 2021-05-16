import sklearn
import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
nltk.download('stopwords')
df = pd.read_csv('spam.csv')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.rename(columns = {'v1':'labels','v2':'message'},inplace=True)
df.drop_duplicates(inplace=True) 
df['labels'] = df['labels'].map({'ham':0,'spam':1})

def clean_data(message):
    message_without_punc = [char for char in message if char not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    seperator = ' '
    return seperator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = MultinomialNB().fit(x_train,y_train)
prediction = model.predict(x_test)

#print(accuracy_score(y_test,prediction))

def predict(text):
    labels = ['Not Spam','Spam']
    x = cv.transform(text).toarray()
    result = model.predict(x)
    if (1 in result):
        return str('This message is '+labels[1])
    else:
        return str('This message is '+labels[0])
    

print(predict(['hey how are you']))

st.title('Spam Classifier')

st.image('spam_image.jpg')

user_input = st.text_input('Write the message')
submit = st.button('Predict')

if submit:
    answer = predict([user_input])
    st.text(answer)
