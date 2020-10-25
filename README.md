# FAKE NEWS

## Importing libraries


```
import numpy as np
import pandas as pd
import nltk
import spacy 
```

## Import the training data 


```
news = pd.read_csv("fake-news/train.csv")

# ignore the unicode characters also hidden in the text
for columnName in ['title','author','text']:
    news[columnName] = news[columnName].str.encode('ascii', 'ignore').str.decode('ascii')
news
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>House Dem Aide: We Didnt Even See Comeys Lette...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didnt Even See Comeys Lette...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20795</th>
      <td>20795</td>
      <td>Rapper T.I.: Trump a Poster Child For White Su...</td>
      <td>Jerome Hudson</td>
      <td>Rapper T. I. unloaded on black celebrities who...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20796</th>
      <td>20796</td>
      <td>N.F.L. Playoffs: Schedule, Matchups and Odds -...</td>
      <td>Benjamin Hoffman</td>
      <td>When the Green Bay Packers lost to the Washing...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20797</th>
      <td>20797</td>
      <td>Macys Is Said to Receive Takeover Approach by ...</td>
      <td>Michael J. de la Merced and Rachel Abrams</td>
      <td>The Macys of today grew from the union of seve...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20798</th>
      <td>20798</td>
      <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>
      <td>Alex Ansary</td>
      <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20799</th>
      <td>20799</td>
      <td>What Keeps the F-35 Alive</td>
      <td>David Swanson</td>
      <td>David Swanson is an author, activist, journa...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>20800 rows × 5 columns</p>
</div>



## Cleaning the data and creating a new file


```

# Cleaning the data

nltk.download('stopwords')
nltk.download('punkt')

import string
from nltk.corpus import stopwords
from unicodedata import normalize

stop = stopwords.words('english')
stop.append('didn’t')

new_news = news[['title', 'text', 'label']].dropna()

##string mutation
new_news["text"] = new_news["text"].str.lower()
new_news["title"] = new_news["title"].str.lower()
new_news["text"]  = new_news["text"].str.translate(str.maketrans('', '', string.punctuation))
new_news["title"]  = new_news["title"].str.translate(str.maketrans('', '', string.punctuation))

## getting the stem words
ps = nltk.PorterStemmer()

##making an array (for some reason idk why i did this but fine )
new_news['title'] = new_news['title'].apply(lambda row: ' '.join(ps.stem(w) for w in row.split() if not w in stop and len(w) >= 3 and not any(i.isdigit() for i in w)))
new_news['text'] = new_news['text'].apply(lambda row: ' '.join(ps.stem(w) for w in row.split() if not w in stop and len(w) >= 3 and not any(i.isdigit() for i in w)))
new_news.to_csv("cleaned_data.csv", index=False)

```


## Analysis on Cleaned Data


```
import ast 

# normal cleaned data 

cleaned = pd.read_csv("cleaned.csv")
cleaned.title = cleaned.title.apply(ast.literal_eval)
cleaned.text = cleaned.text.apply(ast.literal_eval)
```


```
cleaned
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[house, dem, aide, didnt, even, see, comeys, l...</td>
      <td>[house, dem, aide, didnt, even, see, comeys, l...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[flynn, hillary, clinton, big, woman, campus, ...</td>
      <td>[ever, get, feeling, life, circles, roundabout...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[truth, might, get, fired]</td>
      <td>[truth, might, get, fired, october, tension, i...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[civilians, killed, single, airstrike, identif...</td>
      <td>[videos, civilians, killed, single, airstrike,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[iranian, woman, jailed, fictional, unpublishe...</td>
      <td>[print, iranian, woman, sentenced, six, years,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20198</th>
      <td>[rapper, trump, poster, child, white, supremacy]</td>
      <td>[rapper, unloaded, black, celebrities, met, do...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20199</th>
      <td>[nfl, playoffs, schedule, matchups, odds, new,...</td>
      <td>[green, bay, packers, lost, washington, redski...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20200</th>
      <td>[macys, said, receive, takeover, approach, hud...</td>
      <td>[macys, today, grew, union, several, great, na...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20201</th>
      <td>[nato, russia, hold, parallel, exercises, balk...</td>
      <td>[nato, russia, hold, parallel, exercises, balk...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20202</th>
      <td>[keeps, alive]</td>
      <td>[david, swanson, author, activist, journalist,...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>20203 rows × 3 columns</p>
</div>




```
cleaned_new = pd.read_csv("cleaned_new.csv")
cleaned_new
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hous dem aid didnt even see comey letter jason...</td>
      <td>hous dem aid didnt even see comey letter jason...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>flynn hillari clinton big woman campu breitbart</td>
      <td>ever get feel life circl roundabout rather hea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>truth might get fire</td>
      <td>truth might get fire octob tension intellig an...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>civilian kill singl airstrik identifi</td>
      <td>video civilian kill singl airstrik identifi ra...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>iranian woman jail fiction unpublish stori wom...</td>
      <td>print iranian woman sentenc six year prison ir...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19954</th>
      <td>rapper trump poster child white supremaci</td>
      <td>rapper unload black celebr met donald trump el...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19955</th>
      <td>nfl playoff schedul matchup odd new york time</td>
      <td>green bay packer lost washington redskin week ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19956</th>
      <td>maci said receiv takeov approach hudson bay ne...</td>
      <td>maci today grew union sever great name america...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19957</th>
      <td>nato russia hold parallel exercis balkan</td>
      <td>nato russia hold parallel exercis balkan press...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19958</th>
      <td>keep aliv</td>
      <td>david swanson author activist journalist radio...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>19959 rows × 3 columns</p>
</div>



## Frequency graphs 


```
# Confusing stuff - this is "get the most common words in title"
#    - cleaned[cleaned['label']==1] - get the rows that have label = 1
#    - then get the title column only, .str.split( ) then applies string functions to the valuess in the whole row
#.   - so this creates a list for every word in title (i.e "hi there", "[hi, there] but for every row)
#.   - stack then puts everything into a giant list and value counts then counts unique words
fake_text = cleaned[cleaned['label']==1].title.explode().value_counts().nlargest(25)
real_text = cleaned[cleaned['label']==0].title.explode().value_counts().nlargest(25)

import matplotlib.pyplot as plt

# most common words in fake text
plt.figure()
plt.xticks(rotation='vertical')
plt.title("most common words in fake text")
plt.bar(fake_text.index, fake_text.values )

# most common words in real text
plt.figure()
plt.xticks(rotation='vertical')
plt.title("most common words in real text")
plt.bar(real_text.index, real_text.values)
```




    <BarContainer object of 25 artists>




    
![png](output_files/output_12_1.png)
    



    
![png](output_files/output_12_2.png)
    


# Word Cloud of Titles


```
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#this might not work since Ronnie might have the entuire notebook without the commented part
# wordcloud of the fake text
# we can try and fix this 
fake_words = ''
for i in cleaned[cleaned['label']==1].title:
    fake_words += " ".join(i)+" "
wordcloud_fake = WordCloud(width = 800, height = 800, background_color ='black', min_font_size = 10).generate(fake_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_fake) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
```


    
![png](output_files/output_14_0.png)
    



```
# wordcloud of the titles in the real articles

real_words = ''
for i in cleaned[cleaned['label']==0].title:
    real_words += " ".join(i)+" "
wordcloud_real = WordCloud(width = 800, height = 800, background_color ='black', min_font_size = 10).generate(real_words) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_real) 
plt.axis("off") 
plt.tight_layout(pad = 0)
 
```


    
![png](output_files/output_15_0.png)
    


## Featurization


```
from nltk.tokenize import word_tokenize

all_words = []
for row in cleaned_new.title:
    words = word_tokenize(row)
    for w in words:
        all_words.append(w)

# storing the frequencies of all words 
all_words_freq = nltk.FreqDist(all_words)
all_words_freq.most_common(20)
```




    [('new', 7111),
     ('time', 6515),
     ('york', 6370),
     ('trump', 3573),
     ('breitbart', 2402),
     ('clinton', 1240),
     ('hillari', 1232),
     ('donald', 882),
     ('elect', 710),
     ('say', 622),
     ('obama', 564),
     ('news', 544),
     ('russia', 502),
     ('video', 465),
     ('war', 452),
     ('state', 446),
     ('presid', 437),
     ('vote', 420),
     ('america', 418),
     ('report', 412)]




```
word_features = list(all_words_freq)[:2000]
word_features[:10]
```




    ['new',
     'time',
     'york',
     'trump',
     'breitbart',
     'clinton',
     'hillari',
     'donald',
     'elect',
     'say']




```
#define a find_feature func to featurize the entire data

def find_features(data):
    words = word_tokenize(data)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    
    return features

#Example
features = find_features(cleaned_new.title[0])
for key, value in features.items():
    if value == True:
        print(key)

print("\nA view into the features\n")
dict(list(features.items())[0:10])
```

    hous
    even
    comey
    see
    dem
    aid
    tweet
    letter
    didnt
    jason
    
    A view into the features
    





    {'new': False,
     'time': False,
     'york': False,
     'trump': False,
     'breitbart': False,
     'clinton': False,
     'hillari': False,
     'donald': False,
     'elect': False,
     'say': False}




```
titles = list(zip(cleaned_new.title, cleaned_new.label))

# define a seed for reproducibilaty
seed = 1
np.random.seed = seed
np.random.shuffle(titles)

featuresets = [(find_features(text), label) for (text, label) in titles]
```

## Here comes the NLP and scikit-learn


```
# Split into training and testing for now 
# (we can maybe combine the two test and train.csv together later)
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

```

### Important imports to train our model


```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
```


```
#Define models to to train
names = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression', 'SGDClassifier', 'MultinomialNB', 'SVM Linear']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))
models
```




    [('KNeighborsClassifier', KNeighborsClassifier()),
     ('DecisionTreeClassifier', DecisionTreeClassifier()),
     ('RandomForestClassifier', RandomForestClassifier()),
     ('LogisticRegression', LogisticRegression()),
     ('SGDClassifier', SGDClassifier()),
     ('MultinomialNB', MultinomialNB()),
     ('SVM Linear', SVC(kernel='linear'))]




```
#wrap models in NLTK
from nltk.classify.scikitlearn import SklearnClassifier

for name, model in models:
    nltk.model = SklearnClassifier(model)
    nltk.model.train(training)
    accuracy = nltk.classify.accuracy(nltk.model, testing)*100
    print('{0}: Acurracy: {1}'.format(name, accuracy))

```

    KNeighborsClassifier: Acurracy: 84.60921843687375
    DecisionTreeClassifier: Acurracy: 91.62324649298597
    RandomForestClassifier: Acurracy: 93.80761523046091
    LogisticRegression: Acurracy: 93.56713426853707
    SGDClassifier: Acurracy: 93.48697394789579
    MultinomialNB: Acurracy: 88.6372745490982
    SVM Linear: Acurracy: 93.24649298597194



```
texts = list(zip(cleaned_new.text, cleaned_new.label))

# define a seed for reproducibilaty
seed = 1
np.random.seed = seed
np.random.shuffle(texts)

featuresets2 = [(find_features(text), label) for (text, label) in texts]
```


```
#Split into training and testing for now 
# (we can maybe combine the two test and train.csv together later)
from sklearn import model_selection

training2, testing2 = model_selection.train_test_split(featuresets2, test_size = 0.25, random_state = seed)

```


```
from nltk.classify.scikitlearn import SklearnClassifier

for name, model in models:
    nltk.model = SklearnClassifier(model)
    nltk.model.train(training2)
    accuracy = nltk.classify.accuracy(nltk.model, testing2)*100
    print('{0}: Acurracy: {1}'.format(name, accuracy))
```

    KNeighborsClassifier: Acurracy: 56.6933867735471
    DecisionTreeClassifier: Acurracy: 80.92184368737475
    RandomForestClassifier: Acurracy: 89.93987975951903
    LogisticRegression: Acurracy: 92.14428857715431
    SGDClassifier: Acurracy: 90.92184368737475
    MultinomialNB: Acurracy: 86.27254509018036
    SVM Linear: Acurracy: 91.04208416833669



```

```
