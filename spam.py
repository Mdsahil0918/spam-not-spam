#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from collections import Counter
import seaborn as sns


# In[2]:


df = pd.read_excel("C://Users//moham//OneDrive//Documents//Desktop//spam.xlsx")


# In[3]:


df.sample(5)


# In[4]:


df.shape



# In[5]:


#Data cleaning 
df.info()


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


#renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['target'] = encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


# missing values
df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# # EDA

# In[17]:


#EDA
df.head()


# In[18]:


df['target'].value_counts()


# In[19]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[20]:


df.head()


# In[21]:


df['num_characters'] = df['text'].apply(lambda x: len(str(x)) if isinstance(x, (str, int)) else None)


# In[22]:


df.head()


# In[23]:


df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(str(x))) if isinstance(x, (str, bytes)) else 0)


# In[24]:


df.head()


# In[25]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(str(x))))


# In[26]:


df.head()


# In[27]:


df[['num_characters','num_words','num_sentences']].describe()


# In[28]:


#ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[29]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[30]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[31]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[32]:


sns.pairplot(df,hue='target')


# In[33]:


sns.heatmap(df.corr(),annot=True)


# In[34]:


import nltk
from nltk.corpus import stopwords

# Download the stopwords data (if not already downloaded)
nltk.download('stopwords')


# In[35]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    # Check if the input is not a string
    if not isinstance(text, str):
        text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Stemming
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]

    # Join the words back into a string
    return " ".join(text)

# Example usage
input_text = "This is an example sentence with stopwords, and it has punctuation!"
transformed_text = transform_text(input_text)
print("Original text:", input_text)
print("Transformed text:", transformed_text)


# In[36]:


transform_text(str("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."))


# In[37]:


df['text'][10]


# In[38]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[39]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[40]:


df.head()


# In[41]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[42]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[43]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[44]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15,6))
plt.imshow(ham_wc)



# In[45]:


df.head()


# In[46]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[47]:


len(spam_corpus)


# In[48]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[49]:


df.sample(4)


# In[50]:


word_counts = Counter(spam_corpus)

# Create a DataFrame from the Counter
df_word_counts = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Count'])

# Plot using Seaborn
sns.barplot(x='Word', y='Count', data=df_word_counts)
plt.xticks(rotation='vertical')
plt.show()


# In[51]:


word_counts = Counter(ham_corpus)

# Create a DataFrame from the Counter
df_word_counts = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Count'])

# Plot using Seaborn
sns.barplot(x='Word', y='Count', data=df_word_counts)
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


df.head()


# In[53]:


#MODEL BUILDING


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[55]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[56]:


X.shape


# In[57]:


y= df['target'].values


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[60]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[61]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[62]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[63]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[64]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[66]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[67]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}



# In[68]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[69]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[70]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[71]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[72]:


performance_df


# In[73]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[74]:


performance_df1


# In[75]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[76]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

new_df = performance_df.merge(temp_df,on='Algorithm')

new_df_scaled = new_df.merge(temp_df,on='Algorithm')

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

new_df_scaled.merge(temp_df,on='Algorithm')



# In[77]:


svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')

voting.fit(X_train,y_train)



# In[78]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[79]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[80]:


from sklearn.ensemble import StackingClassifier


# In[81]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[ ]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




