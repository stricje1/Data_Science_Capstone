
# coding: utf-8

# # Complaint Routing Model

# #### by Jeffrey Strickland, Ph.D.
# #### 12 August 2019

# ## Introduction

# ### Goal: route complaints to the proper researched team based on product groupings. 
# 
# The business unit receives complaintsin the form of free text and wants to route the complaints to the wants to route the complaints to one of seven different depaertments (product_group name in the daata parenthesis:.
# 	1. Bank account or service (bank_servie)
# 	2. Credit card (credit_card)
# 	3. Credit reporting (credict_reporting)
# 	4. Debt collection (debt_collection)
# 	5. Lines of loans (loan)
# 	6. Money Transfers (money_transfers)
# 	7. Mortgage (mortgange)
# 
# We have obtained a data set wiith 286,362 records that contains complaint text (text), a message identifier (complaint_id) and a verified correct complaint department product_group).

# ### Methodology
# 
# - We created three differnet classification models for comparison:
# 
#     - Naive Bayes
#     - Classification Tree
#     - Random Forest
# 
# - For of natural language processing (NLP) we chose the bag-of-words method.

# ## Load Required Packages

# In[227]:


#Standard packages
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Scikit Learn
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

#Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()



# Allow plots in Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Optional Packages

# In[ ]:


#String
import string

#Plotly
import plotly
import plotly.graph_objects as go
import plotly.express as px
import chart_studio.plotly as py
import re
from sodapy import Socrata
from pprint import pprint

# Spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# import seaborn as sns
#nltk.download('punkt')
#nltk.download('wordnet')


# ### Loading (reading) the Dataset using Pandas
# Our first step in the modeling process is to load the data. In this instance the data is contained in a CVS file that we will read into the Jupyter Notebook.

# In[348]:


df = pd.read_csv("C:/Users/jeff/Documents/Data/case_study_data_copy.csv")


# The next several step we take are for exploring the complaint data, including listing the headings, viewing some records, and assessing the shape of the data frame.

# In[236]:


list(df)


# In[237]:


df.shape


# In[347]:


df.head(5) # for showing a snapshot of the dataset


# ## Store Product Group Data

# The next step, and optional one, is to store the data for each product group into its own object for later use (potentially)

# In[254]:


bks = train[train.product_group=="bank_service"]["text"].values
mtg = train[train.product_group=="mortgage"]["text"].values
crp = train[train.product_group=="credit_reporting"]["text"].values
ccd = train[train.product_group=="credit_card"]["text"].values
lon = train[train.product_group=="loan"]["text"].values
dct = train[train.product_group=="debt_collection"]["text"].values
mts = train[train.product_group=="money_transfers"]["text"].values


# ## Prepare a Frequency Distribution
# 
# The next step consist of plotting the frequency of complaints for each product group and plotting the word counts

# In[256]:


# Provides calculation of the shape of the data for each product group
bank_service_len = train[train['product_group'] == 'bank_service'].shape[0]
mortgage_len = train[train['product_group'] == 'mortgage'].shape[0]
credit_reporting_len = train[train['product_group'] == 'credit_reporting'].shape[0]
loan_len = train[train['product_group'] == 'loan'].shape[0]
credit_card_len = train[train['product_group'] == 'credit_card'].shape[0]
debt_collection_len = train[train['product_group'] == 'debt_collection'].shape[0]
money_transfers_len = train[train['product_group'] == 'money_transfers'].shape[0]


# ## Plotting the Data

# In[257]:


# Sets up the data for construct the frequency distribution (bar chart)
plt.bar(10,bank_service_len,3, label="bank_service")
plt.bar(15,mortgage_len,3, label="mortgage")
plt.bar(20,credit_reporting_len,3, label="credit_reporting")
plt.bar(25,credit_card_len,3, label="credit_card")
plt.bar(30,loan_len,3, label="loan")
plt.bar(35,debt_collection_len,3, label="debt_collection")
plt.bar(40,money_transfers_len,3, label="money_transfers")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propoertion of examples')
plt.show()


# In[258]:


# Returns that complaint frequencies for each product group
credit_reporting_len, mortgage_len, bank_service_len, loan_len, credit_card_len, money_transfers_len


# ## Text Processing Steps:
# 
# 1. Removal of Punctuation → All the punctuation marks are removed from all the text-snippets (instances or documents) from the dataset (corpus).
# 1. Lemmatisation → Inflected forms of a word are known as lemma. For example, (studying, studied) are inflected forms or lemma of the word study which is the root word. So, the lemma of a word are grouped under the single root word. This is done to make the vocabulary of words in the corpus contain distinct words only.
# 1. Removal of Stopwords → Stop-words are usually articles (a, an, the), prepositions (in, on, under, …) and other frequently occurring words that do not provide any key or necessary information. They are removed from all the text-snippets present in the dataset (corpus).

# In[79]:


# Defining a module for Text Processing
def text_process(tex):
    
# 1. Removal of Punctuation Marks 
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)

# 2. Lemmatisation 
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '

# 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not 
            in stopwords.words('english')]


# ## Label Encoding of Classes:
# 
# - As this is a classification problem, here classes are the 7 product groups as mentioned. In our dataset, the labels are non-numeric (bank_services, credit_card, credit_reporting, debt_collection, loan, money_transfers, and mortgage). 
# - These are label encoded to make them numeric, starting from 0 depicting each label in the alphabetic order i.e., (0 → bank_services, 1 → credit_card, 2 → credit_reporting, 3 → debt_collection, 4 → loan, 5 → money_transfers, and 5 → mortgage).

# In[303]:


# Importing necessary libraries
from sklearn.preprocessing import LabelEncoder

y = df['product_group']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# ## Word Cloud Visualization:
# 
# We developed the word clouds as follows:
# 
# - Product groups have their own unique words and phrases in as well as some common ones
# - visualization of the mostly-used words to the least-used words for the product groups can be done
# - Seven text snippets each belonging to the 7 product groups respectively can render a Word Cloud

# In[357]:


# Importing necessary libraries
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

X = df['text']

wordcloud0 = WordCloud().generate(X[0]) # for bank_service
wordcloud1 = WordCloud().generate(X[1]) # for credit_card
wordcloud2 = WordCloud().generate(X[2]) # for credit_reporting 
wordcloud3 = WordCloud().generate(X[3]) # for debt_collection 
wordcloud4 = WordCloud().generate(X[4]) # for loan
wordcloud5 = WordCloud().generate(X[5]) # for money_transfers
wordcloud6 = WordCloud().generate(X[6]) # for mortgage


# In[350]:


#print(X[0])
print(df['product_group'][0])
plt.imshow(wordcloud0, interpolation='bilinear')
plt.show()


# In[351]:


#print(X[22100])
print(df['product_group'][22100])
plt.imshow(wordcloud1, interpolation='bilinear')
plt.show()


# In[352]:


#print(X[42000])
print(df['product_group'][42000])
plt.imshow(wordcloud2, interpolation='bilinear')
plt.show()


# In[353]:


#print(X[152100])
print(df['product_group'][152100])
plt.imshow(wordcloud3, interpolation='bilinear')
plt.show()


# In[354]:


#print(X[199100])
print(df['product_group'][199100])
plt.imshow(wordcloud4, interpolation='bilinear')
plt.show()


# In[355]:


#print(X[227100])
print(df['product_group'][227100])
plt.imshow(wordcloud5, interpolation='bilinear')
plt.show()


# In[356]:


#print(X[237100])
print(df['product_group'][237100])
plt.imshow(wordcloud6, interpolation='bilinear')
plt.show()


# ## Feature Engineering using Bag-of-Words:
# 
# - Machine Learning Algorithms work only on numeric data. 
# - But here, data is present in the form of text only. 
# - For that, by some means, textual data needs to be transformed into numeric form. 
# - One such approach of doing this, is Feature Engineering. 
# - In this approach, numeric features are extracted or engineered from textual data. 
# - There are many Feature Engineering Techniques in existence. 
# - In this problem, Bag-of-Words Technique of Feature Engineering has been used.

# ### Bag-of-Words:
# 
# - With Bag-of-Words a vocabulary of words present in the corpus is maintained. 
# - These words serve as features for each instance or document (each complaint). 
# - Against each word as feature, its frequency in the current document (complaint) is considered. 
# - In this way word features are engineered or extracted from the textual data or corpus.

# In[84]:


# Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# 80-20 splitting the dataset (80%->Training and 20%->Validation)
X_train, X_test, y_train, y_test = train_test_split(X, y
                                  ,test_size=0.2, random_state=1234)

# defining the bag-of-words transformer on the text-processed corpus 
# i.e., text_process() declared in II is executed...
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_train)

# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA

# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_test=bow_transformer.transform(X_test)#TEST DATA


# ## Training the Multinomial Naive Bayes Model:
# 
# Multinomial Naive Bayes Algorithm (Classifier) has been used as the Classification Machine Learning Algorithm [1].

# In[85]:


# Importing necessary libraries
from sklearn.naive_bayes import MultinomialNB

# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

# training the model...
model = model.fit(text_bow_train, y_train)


# ### Model Performance Analysis:
# 
# #### Training Accuracy

# In[213]:


bayes_score = model.score(text_bow_train, y_train)
bayes_score


# #### Validation Accuracy

# In[214]:


bayes_val_score = model.score(text_bow_test, y_test)
bayes_val_score


# The cross-validation score is very close to the model score, 80.99 versus 82.44. The naive Bayes model is a good fit of the data and will produce good predictions.
# 

# ### Precision, Recall and F1–Score

# In[205]:


# Importing necessary libraries
from sklearn.metrics import classification_report
 
# getting the predictions of the Validation Set...
bayes_preds = model.predict(text_bow_test)

# getting the Precision, Recall, F1-Score
print(classification_report(y_test,predictions))


# ### Naive Bayes Predictions

# In[206]:


bayes_preds[0:10]


# In[345]:


model.predict_proba(text_bow_test)


# ### Confusion Matrix

# In[360]:


from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[358]:


# Compute confusion matrix
cm = confusion_matrix(y_test,predictions)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, classes=[0,1,2,3,4,5,6])


# In[359]:


# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, classes=[0,1,2,3,4,5,6],title='Normalized confusion matrix')
plt.show()


# # Training the Classification Tree Model

# In[130]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(text_bow_train, y_train)


# In[153]:


clf


# ### Classification Tree Scores

# In[194]:


from sklearn import tree


# In[221]:


tree_score = clf.score(text_bow_train, y_train)
tree_score


# In[217]:


tree_val_score = clf.score(text_bow_test, y_test)
tree_val_score


# The cross-validation score is much lower than the model score, 75.83 versus 99.91. The classificaton tree model is over-fitting the data and will produce numerous false positives.
# 

# ### Classification Tree Predictions

# In[201]:


tree_preds = clf.predict(text_bow_test)
tree_preds[0:10]


# In[200]:


clf.predict_proba(text_bow_test)


# According to the Performance Analysis, it can be concluded that the NLP Powered Machine Learning Model has been successful in effectively classifying 82.44% unknown (Validation Set) examples correctly. In other words, 82.44% of complaints are identified correctly that it belongs to which product group.

# # Training the Random Forest Model

# In[154]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In[155]:


rf = RandomForestClassifier(n_estimators=100) # initialize
rf.fit(text_bow_train, y_train) # fit the data to the algorithm


# ### Random Forest Model Scores

# In[219]:


forest_score = rf.score(text_bow_train, y_train)
forest_score


# In[218]:


forest_val_score = rf.score(text_bow_test, y_test)
forest_val_score


# In[174]:


rf.get_params(rf.fit)


# ### Random Forest Predictions

# In[204]:


forest_preds = rf.predict(text_bow_test)
forest_preds[0:10]


# In[176]:


rf.predict_proba(text_bow_test)[0:10]


# The cross-validation score is much lower than the model score, 82.45 versus 99.91. The random forest model is over-fitting the data and will produce numerous false positives.

# # Model Comparisons

# In[208]:


bayes_preds[0:20], tree_preds[0:20], forest_preds[0:20]


# In[224]:


bayes_score, bayes_val_score, tree_score, tree_val_score, forest_score, forest_val_score


# # Conclusion
# 
# - The Classification and Rando Forest models will have a tendency to over-predict. 
# - While the Naive Bayes was trained having a lower performnce score (82,44),it will have more accurate predictions. 
# - If we had an out-of-time sample, we could perform better validations on the three models. 
# - In the mean time, we recommend the Naive Bayes Classification MOdel. 
