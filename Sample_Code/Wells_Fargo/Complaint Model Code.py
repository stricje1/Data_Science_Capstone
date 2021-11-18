#Complaint Model

##Goal: route complaints to the proper researched team based on product groupings. 

### The business unit receives complaintsin the form of free text and wants to route the complaints to the wants to route the complaints to one of seven different depaertments (product_group name in the daata parenthesis.
#	1. Bank account or service (bank_servie)
#	2. Credit card (credit_card)
#	3. Credit reporting (credict_reporting)
#	4. Debt collection (debt_collection)
#	5. Lines of loans (loan)
#	6. Money Transfers (money_transfers)
#	7. Mortgage (mortgange)
### We have obtained a data set wiith 286,362 records that contains complaint text (text), a message identifier (complaint_id) and a verified correct complaint department product_group).

## Load required packages

import pandas as pd
from sodapy import Socrata
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import matplotlib
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import chart_studio.plotly as py
import re
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# import seaborn as sns
%matplotlib inline
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

## Load data and required pacages
import pandas as pd
from sodapy import Socrata

dta = pd.read_csv("C:\Users\jeff\Documents\Data\case_study_data.csv")
list(dta)
dta.head()

## Analyze distribution of Product Gouop:
bank_service_len = train[train['product_group'] == 'bank_service'].shape[0]
mortgage_len = train[train['product_group'] == 'mortgage'].shape[0]
credit_reporting_len = train[train['product_group'] == 'credit_reporting'].shape[0]
loan_len = train[train['product_group'] == 'loan'].shape[0]
credit_card_len = train[train['product_group'] == 'credit_card'].shape[0]
debt_collection_len = train[train['product_group'] == 'debt_collection'].shape[0]
money_transfers_len = train[train['product_group'] == 'money_transfers'].shape[0]

## Get frequencies
credit_reporting_len, mortgage_len, bank_service_len, loan_len, credit_card_len, money_transfers_len

## Plot Product Group Frequencies
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

## Store frequencies for future access:
bks = train[train.product_group=="bank_service"]["text"].values
mtg = train[train.product_group=="mortgage"]["text"].values
crp = train[train.product_group=="credit_reporting"]["text"].values
ccd = train[train.product_group=="credit_card"]["text"].values
lon = train[train.product_group=="loan"]["text"].values
dct = train[train.product_group=="debt_collection"]["text"].values
mts = train[train.product_group=="money_transfers"]["text"].values


## Importing necessary libraries
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()

##Text Processing Steps:

### Removal of Punctuation → All the punctuation marks are removed from all the text-snippets (instances or documents) from the dataset (corpus).
###    Lemmatisation → Inflected forms of a word are known as lemma. For example, (studying, studied) are inflected forms or lemma of the word study which is the root word. So, the lemma of a word are grouped under the single root word. This is done to make the vocabulary of words in the corpus contain distinct words only.
###    Removal of Stopwords → Stop-words are usually articles (a, an, the), prepositions (in, on, under, …) and other frequently occurring words that do not provide any key or necessary information. They are removed from all the text-snippets present in the dataset (corpus).

### Defining a module for Text Processing
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
			
## Label Encoding of Classes:

### As this is a classification problem, here classes are the 3 authors as mentioned. But in the dataset, it can be seen that labels are non-numeric (bank_services, credit_card, credit_reporting, debt_collection, loan, money_transfers, and mortgage). These are label encoded to make them numeric, starting from 0 depicting each label in the alphabetic order i.e., (0 → bank_services, 1 → credit_card, 2 → credit_reporting, 3 → debt_collection, 4 → loan, 5 → money_transfers, and 5 → mortgage).

## Importing necessary libraries
from sklearn.preprocessing import LabelEncoder

y = df['product_group']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

## Word Cloud Visualization:

###As the Machine Learning Model is being developed, banking on the fact that the authors have their own unique styles of using particular words in the text, a visualization of the mostly-used words to the least-used words by the 3 authors is done, taking 3 text snippets each belonging to the 3 authors respectively with the help of a Word Cloud.

## Importing necessary libraries
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

print(X[0])
print(df['product_group'][0])
plt.imshow(wordcloud0, interpolation='bilinear')
plt.show()

print(X[1])
print(df['product_group'][1])
plt.imshow(wordcloud1, interpolation='bilinear')
plt.show()

print(X[2])
print(df['product_group'][2])
plt.imshow(wordcloud2, interpolation='bilinear')
plt.show()

print(X[3])
print(df['product_group'][3])
plt.imshow(wordcloud3, interpolation='bilinear')
plt.show()

print(X[4])
print(df['product_group'][4])
plt.imshow(wordcloud4, interpolation='bilinear')
plt.show()

print(X[5])
print(df['product_group'][5])
plt.imshow(wordcloud5, interpolation='bilinear')
plt.show()

print(X[6])
print(df['product_group'][6])
plt.imshow(wordcloud6, interpolation='bilinear')
plt.show()

## Feature Engineering using Bag-of-Words:

### Machine Learning Algorithms work only on numeric data. But here, data is present in the form of text only. For that, by some means, textual data needs to be transformed into numeric form. One such approach of doing this, is Feature Engineering. In this approach, numeric features are extracted or engineered from textual data. There are many Feature Engineering Techniques in existence. In this problem, Bag-of-Words Technique of Feature Engineering has been used.

## Bag-of-Words:

### Here, a vocabulary of words present in the corpus is maintained. These words serve as features for each instance or document (here text snippet). Against each word as feature, its frequency in the current document (text snippet) is considered. Hence, in this way word features are engineered or extracted from the textual data or corpus.

## Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

## 80-20 splitting the dataset (80%->Training and 20%->Validation)
X_train, X_test, y_train, y_test = train_test_split(X, y
                                  ,test_size=0.2, random_state=1234)

## defining the bag-of-words transformer on the text-processed corpus 
##3 i.e., text_process() declared in II is executed...
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_train)

## transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA

## transforming into Bag-of-Words and hence textual data to numeric..
text_bow_test=bow_transformer.transform(X_test)#TEST DATA

## Training the Model:

## Multinomial Naive Bayes Algorithm (Classifier) has been used as the Classification Machine Learning Algorithm [1].

## Training the Model:

# Multinomial Naive Bayes Algorithm (Classifier) has been used as the Classification Machine Learning Algorithm [1].

## Importing necessary libraries
from sklearn.naive_bayes import MultinomialNB

## instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

## training the model...
model = model.fit(text_bow_train, y_train)

## Model Performance Analysis:

### Training Accuracy

model.score(text_bow_train, y_train)

### Validation Accuracy											

model.score(text_bow_test, y_test)

### Precision, Recall and F1–Score

## Importing necessary libraries
from sklearn.metrics import classification_report
 
## getting the predictions of the Validation Set...
predictions = model.predict(text_bow_test)

## getting the Precision, Recall, F1-Score
print(classification_report(y_test,predictions))

## Confusion Matrix

### Importing necessary libraries
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

### Compute confusion matrix
cm = confusion_matrix(y_test,predictions)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, classes=[0,1,2,3,4,5,6])

### Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, classes=[0,1,2,3,4,5,6],title='Normalized confusion matrix')

plt.show()

## According to the Performance Analysis, it can be concluded that the NLP Powered Machine Learning Model has been successful in effectively classifying 82.44% unknown (Validation Set) examples correctly. In other words, 82.44% of complaints are identified correctly that it belongs to which product group.

## DecisionTreeClassifier is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, …, K-1]) classification.

## Using the Complaints dataset, we can construct a tree as follows:

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

tree.plot_tree(clf.fit(X_train, y_train)) 

clf.predict(X_test, y_test)
clf.predict_proba(X_test, y_test)

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("df") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=X_train.feature_names,  
                      class_names=y_train.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_train, y_train)
r = export_text(decision_tree, feature_names=X_train['feature_names'])
print(r)

jupyter nbconvert Jupyter\ Complaints_Model.ipynb --to slides --post serve

 jupyter nbconvert Complaint_Routing_Model.ipynb --to slides --SlidesExporter.reveal_scroll=True --output-dir=C:\Users\jeff\Documents\Wells_Fargo\
 
 jupyter nbconvert Complaint_Routing_Model.ipynb --to slides --reveal-prefix=reveal.js --SlidesExporter.reveal_scroll=True --output-dir=C:\Users\jeff\Documents\Wells_Fargo\