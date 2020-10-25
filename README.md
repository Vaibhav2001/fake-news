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

# Not useful now

"""
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
new_news.to_csv("cleaned_new.csv", index=False)

"""

```




    '\nnltk.download(\'stopwords\')\nnltk.download(\'punkt\')\n\nimport string\nfrom nltk.corpus import stopwords\nfrom unicodedata import normalize\n\nstop = stopwords.words(\'english\')\nstop.append(\'didn’t\')\n\nnew_news = news[[\'title\', \'text\', \'label\']].dropna()\n\n##string mutation\nnew_news["text"] = new_news["text"].str.lower()\nnew_news["title"] = new_news["title"].str.lower()\nnew_news["text"]  = new_news["text"].str.translate(str.maketrans(\'\', \'\', string.punctuation))\nnew_news["title"]  = new_news["title"].str.translate(str.maketrans(\'\', \'\', string.punctuation))\n\n## getting the stem words\nps = nltk.PorterStemmer()\n\n##making an array (for some reason idk why i did this but fine )\nnew_news[\'title\'] = new_news[\'title\'].apply(lambda row: \' \'.join(ps.stem(w) for w in row.split() if not w in stop and len(w) >= 3 and not any(i.isdigit() for i in w)))\nnew_news[\'text\'] = new_news[\'text\'].apply(lambda row: \' \'.join(ps.stem(w) for w in row.split() if not w in stop and len(w) >= 3 and not any(i.isdigit() for i in w)))\nnew_news.to_csv("cleaned_new.csv", index=False)\n\n'



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
import ast 

# cleaned data with only the stem for faster NLP 
cleaned_stem = pd.read_csv("cleaned_stemmed.csv")
cleaned_stem.title = cleaned_stem.title.apply(ast.literal_eval)
cleaned_stem.text = cleaned_stem.text.apply(ast.literal_eval)
```


```
cleaned_stem
```


```
cleaned_new = pd.read_csv("cleaned_new.csv")
cleaned_new
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

- Ask: Do some quotes here and there matter?
    - i removed em (it was b/c unicode that your code didnt catch it)

- What about featurization and which library would be the best for us to train out model?
    - http://uc-r.github.io/creating-text-features

- Do names/ locations influence the model (ie should we remove them)?
    - Try removing them vs not removing them and see what happens ;)
    - But for v1 probably no; you can use a part-of-speech (e.g. is this word a noun/verb/etc to remove later)

## Using NLTK to get the frequency efficiently and other stuff (Maybe Featurization)


```
from nltk.tokenize import word_tokenize

nltk.download('punkt')

#stemmed words
all_words = []
for row in cleaned_new.title:
    words = word_tokenize(row)
    for w in words:
        all_words.append(w)

all_words_freq = nltk.FreqDist(all_words)
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.



```
word_features = list(all_words_freq)[:2000]
```


```
word_features
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
     'say',
     'obama',
     'news',
     'russia',
     'video',
     'war',
     'state',
     'presid',
     'vote',
     'america',
     'report',
     'fbi',
     'world',
     'email',
     'attack',
     'american',
     'day',
     'comment',
     'call',
     'get',
     'hous',
     'kill',
     'media',
     'polic',
     'white',
     'year',
     'campaign',
     'plan',
     'protest',
     'support',
     'democrat',
     'bill',
     'syria',
     'make',
     'russian',
     'show',
     'first',
     'black',
     'win',
     'one',
     'brief',
     'take',
     'break',
     'peopl',
     'even',
     'man',
     'watch',
     'nation',
     'use',
     'investig',
     'like',
     'may',
     'million',
     'die',
     'right',
     'back',
     'polit',
     'putin',
     'wikileak',
     'could',
     'voter',
     'want',
     'muslim',
     'find',
     'immigr',
     'end',
     'ban',
     'life',
     'case',
     'fight',
     'offic',
     'fake',
     'come',
     'border',
     'china',
     'isi',
     'republican',
     'women',
     'court',
     'gop',
     'illeg',
     'chang',
     'comey',
     'fire',
     'offici',
     'govern',
     'warn',
     'face',
     'order',
     'health',
     'arrest',
     'live',
     'leader',
     'obamacar',
     'top',
     'forc',
     'big',
     'claim',
     'deal',
     'rule',
     'help',
     'law',
     'work',
     'power',
     'reveal',
     'citi',
     'tri',
     'way',
     'parti',
     'open',
     'wall',
     'death',
     'look',
     'north',
     'paul',
     'poll',
     'need',
     'terror',
     'woman',
     'dont',
     'california',
     'know',
     'home',
     'famili',
     'stand',
     'migrant',
     'stop',
     'meet',
     'charg',
     'turn',
     'islam',
     'leak',
     'run',
     'tell',
     'see',
     'john',
     'move',
     'found',
     'speech',
     'israel',
     'give',
     'senat',
     'leav',
     'student',
     'school',
     'shoot',
     'ask',
     'syrian',
     'real',
     'texa',
     'polici',
     'global',
     'expos',
     'two',
     'review',
     'aleppo',
     'secur',
     'gun',
     'race',
     'presidenti',
     'job',
     'would',
     'talk',
     'control',
     'iran',
     'releas',
     'thing',
     'lose',
     'public',
     'saudi',
     'mosul',
     'accus',
     'still',
     'lead',
     'human',
     'cnn',
     'terrorist',
     'sourc',
     'strike',
     'today',
     'secret',
     'brexit',
     'daili',
     'happen',
     'dead',
     'fund',
     'ryan',
     'nuclear',
     'refuge',
     'sex',
     'keep',
     'hit',
     'question',
     'crime',
     'game',
     'congress',
     'think',
     'foreign',
     'rig',
     'bomb',
     'lie',
     'last',
     'water',
     'celebr',
     'militari',
     'anoth',
     'care',
     'former',
     'put',
     'chief',
     'set',
     'gener',
     'money',
     'facebook',
     'ralli',
     'feder',
     'victori',
     'christian',
     'rise',
     'gold',
     'star',
     'fear',
     'execut',
     'children',
     'push',
     'dem',
     'rock',
     'stori',
     'director',
     'fail',
     'south',
     'aid',
     'scandal',
     'test',
     'part',
     'head',
     'pay',
     'tax',
     'wont',
     'threat',
     'group',
     'post',
     'debat',
     'judg',
     'confirm',
     'alien',
     'insid',
     'hack',
     'close',
     'go',
     'histori',
     'next',
     'final',
     'love',
     'photo',
     'pipelin',
     'child',
     'veteran',
     'liber',
     'fraud',
     'press',
     'washington',
     'countri',
     'food',
     'sign',
     'shock',
     'milo',
     'suspect',
     'podesta',
     'foundat',
     'cut',
     'return',
     'jame',
     'suprem',
     'threaten',
     'critic',
     'drug',
     'korea',
     'becom',
     'behind',
     'play',
     'rape',
     'twitter',
     'hope',
     'violenc',
     'team',
     'street',
     'record',
     'justic',
     'start',
     'week',
     'great',
     'free',
     'market',
     'unit',
     'must',
     'best',
     'turkey',
     'reason',
     'interview',
     'seek',
     'good',
     'demand',
     'pick',
     'michael',
     'hate',
     'left',
     'prepar',
     'high',
     'onion',
     'act',
     'climat',
     'conserv',
     'machin',
     'cant',
     'destroy',
     'mani',
     'announc',
     'let',
     'finest',
     'said',
     'step',
     'car',
     'got',
     'girl',
     'name',
     'murder',
     'tie',
     'truth',
     'dakota',
     'cop',
     'caught',
     'antitrump',
     'europ',
     'cancer',
     'prison',
     'system',
     'hollywood',
     'exclus',
     'blame',
     'crash',
     'candid',
     'crisi',
     'futur',
     'bank',
     'goe',
     'launch',
     'save',
     'full',
     'inform',
     'mike',
     'here',
     'billion',
     'target',
     'block',
     'major',
     'fox',
     'truthfe',
     'colleg',
     'battl',
     'peac',
     'troop',
     'univers',
     'continu',
     'link',
     'access',
     'rais',
     'session',
     'book',
     'build',
     'shot',
     'replac',
     'men',
     'agent',
     'administr',
     'legal',
     'french',
     'caus',
     'near',
     'bannon',
     'never',
     'admit',
     'long',
     'west',
     'weiner',
     'soro',
     'follow',
     'googl',
     'inaugur',
     'novemb',
     'georg',
     'issu',
     'armi',
     'mexico',
     'social',
     'kelli',
     'line',
     'protect',
     'refus',
     'franc',
     'corrupt',
     'evid',
     'travel',
     'weapon',
     'offer',
     'mark',
     'air',
     'london',
     'pari',
     'tweet',
     'bad',
     'civil',
     'nato',
     'night',
     'trade',
     'worker',
     'point',
     'minist',
     'studi',
     'econom',
     'send',
     'secretari',
     'ever',
     'possibl',
     'miss',
     'begin',
     'list',
     'penc',
     'readi',
     'assang',
     'made',
     'grow',
     'danger',
     'gorsuch',
     'wrong',
     'realli',
     'assault',
     'member',
     'promis',
     'usa',
     'without',
     'bring',
     'old',
     'month',
     'hold',
     'nfl',
     'scientist',
     'victim',
     'network',
     'hand',
     'doesnt',
     'slam',
     'reopen',
     'mean',
     'mother',
     'he',
     'visit',
     'palestinian',
     'robert',
     'defeat',
     'friday',
     'zika',
     'read',
     'everi',
     'latest',
     'dog',
     'surpris',
     'declar',
     'young',
     'iraq',
     'fall',
     'letter',
     'avoid',
     'hear',
     'rate',
     'amid',
     'sinc',
     'project',
     'person',
     'deep',
     'massiv',
     'camp',
     'event',
     'megyn',
     'surviv',
     'problem',
     'economi',
     'creat',
     'british',
     'civilian',
     'urg',
     'defend',
     'deport',
     'allow',
     'expect',
     'david',
     'interest',
     'chines',
     'steal',
     'rep',
     'bodi',
     'freedom',
     'town',
     'feel',
     'sander',
     'land',
     'fan',
     'cancel',
     'risk',
     'remov',
     'three',
     'drop',
     'berni',
     'crimin',
     'messag',
     'itali',
     'missil',
     'word',
     'sexual',
     'buy',
     'space',
     'india',
     'florida',
     'carri',
     'matter',
     'intellig',
     'mexican',
     'much',
     'wednesday',
     'didnt',
     'speak',
     'mass',
     'huma',
     'movi',
     'eye',
     'told',
     'challeng',
     'march',
     'red',
     'commun',
     'oper',
     'pass',
     'olymp',
     'spi',
     'activist',
     'action',
     'price',
     'chicago',
     'lost',
     'boy',
     'explain',
     'journalist',
     'gain',
     'art',
     'germani',
     'britain',
     'join',
     'hour',
     'second',
     'resist',
     'host',
     'away',
     'oil',
     'respons',
     'four',
     'arm',
     'walk',
     'central',
     'effort',
     'privat',
     'around',
     'busi',
     'jail',
     'doj',
     'thousand',
     'soon',
     'steve',
     'predict',
     'key',
     'cia',
     'reach',
     'blast',
     'gay',
     'endors',
     'prove',
     'well',
     'biggest',
     'percent',
     'resign',
     'concern',
     'octob',
     'fed',
     'sport',
     'includ',
     'doctor',
     'realiti',
     'berkeley',
     'pope',
     'compani',
     'elit',
     'littl',
     'morn',
     'iraqi',
     'special',
     'depart',
     'vow',
     'voic',
     'revolut',
     'mind',
     'guilti',
     'probe',
     'place',
     'japan',
     'past',
     'transgend',
     'fals',
     'budget',
     'arabia',
     'attempt',
     'role',
     'earli',
     'plane',
     'church',
     'cover',
     'internet',
     'financi',
     'share',
     'learn',
     'airport',
     'patriot',
     'east',
     'earthquak',
     'repeal',
     'blue',
     'attorney',
     'view',
     'appear',
     'bombshel',
     'beat',
     'noth',
     'son',
     'jewish',
     'expert',
     'babi',
     'tip',
     'discuss',
     'riot',
     'arriv',
     'your',
     'german',
     'train',
     'honor',
     'eastern',
     'paper',
     'ahead',
     'abus',
     'isnt',
     'age',
     'donor',
     'emerg',
     'award',
     'femal',
     'ignor',
     'indian',
     'cost',
     'abedin',
     'shut',
     'defens',
     'isra',
     'racist',
     'differ',
     'halloween',
     'flag',
     'outlook',
     'light',
     'better',
     'chao',
     'dump',
     'sen',
     'film',
     'believ',
     'alleg',
     'dollar',
     'discov',
     'johnson',
     'zone',
     'reject',
     'fact',
     'spicer',
     'western',
     'carolina',
     'thank',
     'middl',
     'damag',
     'chri',
     'result',
     'paid',
     'worri',
     'chemic',
     'collaps',
     'propos',
     'earth',
     'settlement',
     'sell',
     'silver',
     'sue',
     'choic',
     'mayor',
     'stage',
     'alli',
     'governor',
     'joe',
     'least',
     'updat',
     'rio',
     'abort',
     'alert',
     'intern',
     'inquiri',
     'collect',
     'tuesday',
     'jerusalem',
     'alreadi',
     'document',
     'heart',
     'activ',
     'total',
     'vladimir',
     'monday',
     'search',
     'violent',
     'sanctuari',
     'advis',
     'add',
     'michel',
     'receiv',
     'lo',
     'storm',
     'sentenc',
     'consid',
     'industri',
     'plot',
     'less',
     'file',
     'increas',
     'cultur',
     'servic',
     'path',
     'wors',
     'elector',
     'era',
     'rebel',
     'trial',
     'enter',
     'ice',
     'brazil',
     'class',
     'hard',
     'sean',
     'effect',
     'author',
     'nyt',
     'struggl',
     'anthoni',
     'season',
     'despit',
     'welcom',
     'nbc',
     'friend',
     'soldier',
     'side',
     'far',
     'jeff',
     'coulter',
     'establish',
     'appl',
     'thought',
     'melania',
     'flight',
     'agenda',
     'base',
     'favor',
     'union',
     'spend',
     'speaker',
     'agenc',
     'mysteri',
     'mock',
     'trust',
     'island',
     'brother',
     'thursday',
     'philippin',
     'fatal',
     'apolog',
     'super',
     'safe',
     'toward',
     'nearli',
     'lawyer',
     'employe',
     'number',
     'program',
     'benefit',
     'scott',
     'sea',
     'check',
     'prosecutor',
     'airstrik',
     'ukrain',
     'everyon',
     'mental',
     'moor',
     'ga',
     'assassin',
     'agre',
     'deni',
     'drone',
     'data',
     'tank',
     'liberti',
     'shift',
     'suggest',
     'account',
     'cross',
     'yemen',
     'moment',
     'european',
     'kushner',
     'clear',
     'hacker',
     'chart',
     'lawsuit',
     'daughter',
     'dream',
     'cartel',
     'might',
     'organ',
     'wife',
     'bust',
     'true',
     'oregon',
     'suspend',
     'brain',
     'pull',
     'cash',
     'democraci',
     'design',
     'suit',
     'donat',
     'someth',
     'approv',
     'aim',
     'coup',
     'promot',
     'wire',
     'green',
     'decis',
     'wear',
     'relat',
     'crowd',
     'king',
     'site',
     'tom',
     'clash',
     'flynn',
     'venezuela',
     'retir',
     'loss',
     'teacher',
     'seri',
     'duke',
     'troubl',
     'parent',
     'failur',
     'five',
     'audio',
     'constitut',
     'ann',
     'stock',
     'chelsea',
     'hire',
     'dutert',
     'insur',
     'sweden',
     'delay',
     'navi',
     'altern',
     'import',
     'progress',
     'address',
     'answer',
     'dnc',
     'pentagon',
     'success',
     'roger',
     'unveil',
     'potenti',
     'across',
     'sale',
     'prime',
     'burn',
     'teen',
     'council',
     'suicid',
     'orlando',
     'propaganda',
     'seen',
     'store',
     'ring',
     'god',
     'hidden',
     'yet',
     'jew',
     'kid',
     'door',
     'posit',
     'reform',
     'tillerson',
     'six',
     'limit',
     'fix',
     'medic',
     'took',
     'ambassador',
     'longer',
     'bridg',
     'pictur',
     'pressur',
     'corpor',
     'track',
     'bid',
     'natur',
     'what',
     'father',
     'cold',
     'michigan',
     'franci',
     'cabinet',
     'quit',
     'africa',
     'common',
     'arab',
     'afghanistan',
     'julian',
     'entir',
     'korean',
     'kim',
     'area',
     'educ',
     'mosqu',
     'remain',
     'drive',
     'tim',
     'indict',
     'nomin',
     'anonym',
     'note',
     'conspiraci',
     'chair',
     'stay',
     'eric',
     'player',
     'ceo',
     'movement',
     'largest',
     'stephen',
     'para',
     'focu',
     'wait',
     'afghan',
     'lynch',
     'convict',
     'marijuana',
     'hell',
     'ufo',
     'station',
     'oscar',
     'ted',
     'blow',
     'brooklyn',
     'condemn',
     'trip',
     'merkel',
     'jone',
     'develop',
     'worst',
     'websit',
     'hide',
     'involv',
     'phone',
     ...]




```
#define a find_feature func

def find_features(data):
    words = word_tokenize(data)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    
    return features

#Example
features = find_features(cleaned_new.text[2])
for key, value in features.items():
    if value == True:
        print(key)
```

    obama
    war
    state
    presid
    world
    attack
    american
    get
    hous
    white
    year
    campaign
    plan
    support
    syria
    make
    first
    one
    brief
    break
    peopl
    even
    nation
    use
    like
    million
    right
    back
    polit
    could
    want
    find
    end
    case
    offic
    chang
    fire
    govern
    order
    leader
    forc
    claim
    help
    work
    power
    tri
    way
    look
    need
    know
    meet
    charg
    tell
    see
    john
    found
    israel
    give
    real
    polici
    secur
    would
    talk
    thing
    public
    saudi
    still
    lead
    sourc
    today
    happen
    nuclear
    keep
    hit
    congress
    think
    foreign
    lie
    anoth
    former
    put
    chief
    gener
    ralli
    rise
    star
    fear
    stori
    director
    aid
    test
    head
    group
    close
    go
    photo
    liber
    countri
    jame
    becom
    play
    hope
    free
    market
    unit
    must
    reason
    demand
    act
    mani
    got
    truth
    launch
    inform
    peac
    univers
    administr
    caus
    never
    long
    follow
    georg
    armi
    social
    line
    protect
    evid
    weapon
    bad
    worker
    studi
    send
    list
    made
    danger
    hand
    mean
    declar
    iraq
    avoid
    person
    problem
    expect
    intellig
    mass
    march
    commun
    spi
    action
    oil
    effort
    thousand
    soon
    key
    cia
    prove
    well
    octob
    includ
    elit
    iraqi
    place
    past
    arabia
    earli
    east
    your
    defens
    differ
    better
    believ
    dollar
    middl
    result
    chemic
    sell
    choic
    alli
    least
    alreadi
    search
    advis
    receiv
    class
    hard
    despit
    spend
    agenc
    program
    might
    organ
    true
    democraci
    someth
    answer
    council
    hidden
    yet
    afghanistan
    develop
    hide
    accept
    independ
    center
    later
    ground
    bush
    beyond
    scienc
    research
    histor
    lack
    card
    barack
    gave
    proof
    connect
    associ
    lesson
    short
    strong
    manag
    probabl
    improv
    listen
    alway
    credit
    serv
    someon
    write
    forget
    almost
    staff
    lower
    sure
    wit
    leadership
    taken
    engin
    invad
    seem
    invas
    anyth
    appoint
    board
    nationwid
    analyst
    feed
    fit
    regim
    tension
    wonder
    libya
    rescu
    affair
    medicin
    destruct
    approach
    went
    can
    tale



```
titles = list(zip(cleaned_new.title, cleaned_new.label))

# define a seed for reproducibilaty
seed = 1
np.random.seed = seed
np.random.shuffle(titles)

featuresets = [(find_features(text), label) for (text, label) in titles]
```


```
# A look into our featureset 
featuresets[0]
```




    ({'new': False,
      'time': False,
      'york': False,
      'trump': True,
      'breitbart': True,
      'clinton': False,
      'hillari': False,
      'donald': False,
      'elect': False,
      'say': False,
      'obama': False,
      'news': False,
      'russia': False,
      'video': False,
      'war': False,
      'state': False,
      'presid': True,
      'vote': False,
      'america': False,
      'report': False,
      'fbi': False,
      'world': False,
      'email': False,
      'attack': False,
      'american': False,
      'day': False,
      'comment': False,
      'call': False,
      'get': False,
      'hous': False,
      'kill': False,
      'media': False,
      'polic': False,
      'white': True,
      'year': False,
      'campaign': False,
      'plan': False,
      'protest': False,
      'support': False,
      'democrat': False,
      'bill': False,
      'syria': False,
      'make': False,
      'russian': False,
      'show': False,
      'first': False,
      'black': False,
      'win': False,
      'one': False,
      'brief': False,
      'take': False,
      'break': False,
      'peopl': False,
      'even': False,
      'man': False,
      'watch': False,
      'nation': False,
      'use': False,
      'investig': False,
      'like': False,
      'may': False,
      'million': False,
      'die': False,
      'right': False,
      'back': False,
      'polit': False,
      'putin': False,
      'wikileak': False,
      'could': False,
      'voter': False,
      'want': False,
      'muslim': False,
      'find': False,
      'immigr': False,
      'end': False,
      'ban': False,
      'life': False,
      'case': False,
      'fight': False,
      'offic': False,
      'fake': False,
      'come': False,
      'border': False,
      'china': False,
      'isi': False,
      'republican': False,
      'women': False,
      'court': False,
      'gop': False,
      'illeg': False,
      'chang': False,
      'comey': False,
      'fire': False,
      'offici': False,
      'govern': False,
      'warn': False,
      'face': False,
      'order': False,
      'health': False,
      'arrest': False,
      'live': False,
      'leader': False,
      'obamacar': False,
      'top': False,
      'forc': False,
      'big': False,
      'claim': False,
      'deal': False,
      'rule': False,
      'help': False,
      'law': False,
      'work': False,
      'power': False,
      'reveal': False,
      'citi': False,
      'tri': False,
      'way': False,
      'parti': False,
      'open': False,
      'wall': False,
      'death': False,
      'look': False,
      'north': False,
      'paul': False,
      'poll': False,
      'need': False,
      'terror': False,
      'woman': False,
      'dont': False,
      'california': False,
      'know': False,
      'home': False,
      'famili': False,
      'stand': False,
      'migrant': False,
      'stop': False,
      'meet': False,
      'charg': False,
      'turn': False,
      'islam': False,
      'leak': False,
      'run': False,
      'tell': False,
      'see': False,
      'john': False,
      'move': False,
      'found': False,
      'speech': False,
      'israel': False,
      'give': False,
      'senat': False,
      'leav': False,
      'student': False,
      'school': False,
      'shoot': False,
      'ask': False,
      'syrian': False,
      'real': False,
      'texa': False,
      'polici': False,
      'global': False,
      'expos': False,
      'two': False,
      'review': False,
      'aleppo': False,
      'secur': False,
      'gun': False,
      'race': False,
      'presidenti': False,
      'job': False,
      'would': False,
      'talk': False,
      'control': False,
      'iran': False,
      'releas': False,
      'thing': False,
      'lose': False,
      'public': False,
      'saudi': False,
      'mosul': False,
      'accus': False,
      'still': False,
      'lead': False,
      'human': False,
      'cnn': False,
      'terrorist': False,
      'sourc': False,
      'strike': False,
      'today': False,
      'secret': False,
      'brexit': False,
      'daili': False,
      'happen': False,
      'dead': False,
      'fund': False,
      'ryan': False,
      'nuclear': False,
      'refuge': False,
      'sex': False,
      'keep': False,
      'hit': False,
      'question': False,
      'crime': False,
      'game': False,
      'congress': False,
      'think': False,
      'foreign': False,
      'rig': False,
      'bomb': False,
      'lie': False,
      'last': False,
      'water': False,
      'celebr': False,
      'militari': False,
      'anoth': False,
      'care': False,
      'former': False,
      'put': False,
      'chief': False,
      'set': False,
      'gener': False,
      'money': False,
      'facebook': False,
      'ralli': False,
      'feder': False,
      'victori': False,
      'christian': False,
      'rise': False,
      'gold': False,
      'star': False,
      'fear': True,
      'execut': False,
      'children': False,
      'push': False,
      'dem': False,
      'rock': False,
      'stori': False,
      'director': False,
      'fail': False,
      'south': False,
      'aid': False,
      'scandal': False,
      'test': False,
      'part': False,
      'head': False,
      'pay': False,
      'tax': False,
      'wont': False,
      'threat': False,
      'group': False,
      'post': False,
      'debat': False,
      'judg': False,
      'confirm': False,
      'alien': False,
      'insid': False,
      'hack': False,
      'close': False,
      'go': False,
      'histori': False,
      'next': False,
      'final': False,
      'love': False,
      'photo': False,
      'pipelin': False,
      'child': False,
      'veteran': False,
      'liber': False,
      'fraud': False,
      'press': False,
      'washington': False,
      'countri': False,
      'food': False,
      'sign': False,
      'shock': False,
      'milo': False,
      'suspect': False,
      'podesta': False,
      'foundat': False,
      'cut': False,
      'return': False,
      'jame': False,
      'suprem': False,
      'threaten': False,
      'critic': False,
      'drug': False,
      'korea': False,
      'becom': False,
      'behind': False,
      'play': False,
      'rape': False,
      'twitter': False,
      'hope': False,
      'violenc': False,
      'team': False,
      'street': False,
      'record': False,
      'justic': False,
      'start': False,
      'week': False,
      'great': False,
      'free': False,
      'market': False,
      'unit': False,
      'must': False,
      'best': False,
      'turkey': False,
      'reason': False,
      'interview': False,
      'seek': False,
      'good': False,
      'demand': False,
      'pick': False,
      'michael': False,
      'hate': False,
      'left': False,
      'prepar': False,
      'high': False,
      'onion': False,
      'act': False,
      'climat': False,
      'conserv': False,
      'machin': False,
      'cant': False,
      'destroy': False,
      'mani': False,
      'announc': False,
      'let': False,
      'finest': False,
      'said': False,
      'step': False,
      'car': False,
      'got': False,
      'girl': False,
      'name': False,
      'murder': False,
      'tie': False,
      'truth': False,
      'dakota': False,
      'cop': False,
      'caught': False,
      'antitrump': False,
      'europ': False,
      'cancer': False,
      'prison': False,
      'system': False,
      'hollywood': False,
      'exclus': False,
      'blame': False,
      'crash': False,
      'candid': False,
      'crisi': False,
      'futur': False,
      'bank': False,
      'goe': False,
      'launch': False,
      'save': False,
      'full': False,
      'inform': False,
      'mike': False,
      'here': False,
      'billion': False,
      'target': False,
      'block': False,
      'major': False,
      'fox': False,
      'truthfe': False,
      'colleg': False,
      'battl': False,
      'peac': False,
      'troop': False,
      'univers': False,
      'continu': False,
      'link': False,
      'access': False,
      'rais': False,
      'session': False,
      'book': False,
      'build': False,
      'shot': False,
      'replac': False,
      'men': False,
      'agent': False,
      'administr': False,
      'legal': False,
      'french': False,
      'caus': False,
      'near': False,
      'bannon': False,
      'never': False,
      'admit': False,
      'long': False,
      'west': False,
      'weiner': False,
      'soro': False,
      'follow': False,
      'googl': False,
      'inaugur': False,
      'novemb': False,
      'georg': False,
      'issu': False,
      'armi': False,
      'mexico': False,
      'social': False,
      'kelli': False,
      'line': False,
      'protect': False,
      'refus': False,
      'franc': False,
      'corrupt': False,
      'evid': False,
      'travel': False,
      'weapon': False,
      'offer': False,
      'mark': False,
      'air': False,
      'london': False,
      'pari': False,
      'tweet': False,
      'bad': False,
      'civil': False,
      'nato': False,
      'night': False,
      'trade': False,
      'worker': False,
      'point': False,
      'minist': False,
      'studi': False,
      'econom': False,
      'send': False,
      'secretari': False,
      'ever': False,
      'possibl': False,
      'miss': False,
      'begin': False,
      'list': False,
      'penc': False,
      'readi': False,
      'assang': False,
      'made': False,
      'grow': False,
      'danger': False,
      'gorsuch': False,
      'wrong': False,
      'realli': False,
      'assault': False,
      'member': False,
      'promis': False,
      'usa': False,
      'without': False,
      'bring': False,
      'old': False,
      'month': False,
      'hold': False,
      'nfl': False,
      'scientist': False,
      'victim': False,
      'network': False,
      'hand': False,
      'doesnt': False,
      'slam': False,
      'reopen': False,
      'mean': False,
      'mother': False,
      'he': False,
      'visit': False,
      'palestinian': False,
      'robert': False,
      'defeat': False,
      'friday': False,
      'zika': False,
      'read': False,
      'everi': False,
      'latest': False,
      'dog': False,
      'surpris': False,
      'declar': False,
      'young': False,
      'iraq': False,
      'fall': False,
      'letter': False,
      'avoid': False,
      'hear': False,
      'rate': False,
      'amid': False,
      'sinc': False,
      'project': False,
      'person': False,
      'deep': False,
      'massiv': False,
      'camp': False,
      'event': False,
      'megyn': False,
      'surviv': False,
      'problem': False,
      'economi': False,
      'creat': False,
      'british': False,
      'civilian': False,
      'urg': False,
      'defend': False,
      'deport': False,
      'allow': False,
      'expect': False,
      'david': False,
      'interest': False,
      'chines': False,
      'steal': False,
      'rep': False,
      'bodi': False,
      'freedom': False,
      'town': False,
      'feel': False,
      'sander': False,
      'land': False,
      'fan': False,
      'cancel': False,
      'risk': False,
      'remov': False,
      'three': False,
      'drop': False,
      'berni': False,
      'crimin': False,
      'messag': False,
      'itali': False,
      'missil': False,
      'word': False,
      'sexual': False,
      'buy': False,
      'space': False,
      'india': False,
      'florida': False,
      'carri': False,
      'matter': False,
      'intellig': False,
      'mexican': False,
      'much': False,
      'wednesday': False,
      'didnt': False,
      'speak': False,
      'mass': False,
      'huma': False,
      'movi': False,
      'eye': False,
      'told': False,
      'challeng': False,
      'march': False,
      'red': False,
      'commun': False,
      'oper': False,
      'pass': False,
      'olymp': False,
      'spi': False,
      'activist': False,
      'action': False,
      'price': False,
      'chicago': False,
      'lost': False,
      'boy': False,
      'explain': False,
      'journalist': False,
      'gain': False,
      'art': False,
      'germani': False,
      'britain': False,
      'join': False,
      'hour': False,
      'second': False,
      'resist': False,
      'host': False,
      'away': False,
      'oil': False,
      'respons': False,
      'four': False,
      'arm': False,
      'walk': False,
      'central': False,
      'effort': False,
      'privat': False,
      'around': False,
      'busi': False,
      'jail': False,
      'doj': False,
      'thousand': False,
      'soon': False,
      'steve': False,
      'predict': False,
      'key': False,
      'cia': False,
      'reach': False,
      'blast': False,
      'gay': False,
      'endors': False,
      'prove': False,
      'well': False,
      'biggest': False,
      'percent': False,
      'resign': False,
      'concern': False,
      'octob': False,
      'fed': False,
      'sport': False,
      'includ': False,
      'doctor': False,
      'realiti': False,
      'berkeley': False,
      'pope': False,
      'compani': False,
      'elit': False,
      'littl': False,
      'morn': False,
      'iraqi': False,
      'special': False,
      'depart': False,
      'vow': False,
      'voic': False,
      'revolut': False,
      'mind': False,
      'guilti': False,
      'probe': False,
      'place': False,
      'japan': False,
      'past': False,
      'transgend': False,
      'fals': False,
      'budget': False,
      'arabia': False,
      'attempt': False,
      'role': False,
      'earli': False,
      'plane': False,
      'church': False,
      'cover': False,
      'internet': False,
      'financi': False,
      'share': False,
      'learn': False,
      'airport': False,
      'patriot': False,
      'east': False,
      'earthquak': False,
      'repeal': False,
      'blue': False,
      'attorney': False,
      'view': False,
      'appear': False,
      'bombshel': False,
      'beat': False,
      'noth': False,
      'son': False,
      'jewish': False,
      'expert': False,
      'babi': False,
      'tip': False,
      'discuss': False,
      'riot': False,
      'arriv': False,
      'your': True,
      'german': False,
      'train': False,
      'honor': False,
      'eastern': False,
      'paper': False,
      'ahead': False,
      'abus': False,
      'isnt': False,
      'age': False,
      'donor': False,
      'emerg': False,
      'award': False,
      'femal': False,
      'ignor': False,
      'indian': False,
      'cost': False,
      'abedin': False,
      'shut': False,
      'defens': False,
      'isra': False,
      'racist': False,
      'differ': False,
      'halloween': False,
      'flag': False,
      'outlook': False,
      'light': False,
      'better': False,
      'chao': False,
      'dump': False,
      'sen': False,
      'film': False,
      'believ': False,
      'alleg': False,
      'dollar': False,
      'discov': False,
      'johnson': False,
      'zone': False,
      'reject': False,
      'fact': False,
      'spicer': False,
      'western': False,
      'carolina': False,
      'thank': False,
      'middl': False,
      'damag': False,
      'chri': False,
      'result': False,
      'paid': False,
      'worri': False,
      'chemic': False,
      'collaps': False,
      'propos': False,
      'earth': False,
      'settlement': False,
      'sell': False,
      'silver': False,
      'sue': False,
      'choic': False,
      'mayor': False,
      'stage': False,
      'alli': False,
      'governor': False,
      'joe': False,
      'least': False,
      'updat': False,
      'rio': False,
      'abort': False,
      'alert': False,
      'intern': False,
      'inquiri': False,
      'collect': False,
      'tuesday': False,
      'jerusalem': False,
      'alreadi': False,
      'document': False,
      'heart': False,
      'activ': False,
      'total': False,
      'vladimir': False,
      'monday': False,
      'search': False,
      'violent': False,
      'sanctuari': False,
      'advis': False,
      'add': False,
      'michel': False,
      'receiv': False,
      'lo': False,
      'storm': False,
      'sentenc': False,
      'consid': False,
      'industri': False,
      'plot': False,
      'less': False,
      'file': False,
      'increas': False,
      'cultur': False,
      'servic': False,
      'path': False,
      'wors': False,
      'elector': False,
      'era': False,
      'rebel': False,
      'trial': False,
      'enter': False,
      'ice': False,
      'brazil': False,
      'class': False,
      'hard': False,
      'sean': False,
      'effect': False,
      'author': False,
      'nyt': False,
      'struggl': False,
      'anthoni': False,
      'season': False,
      'despit': False,
      'welcom': False,
      'nbc': False,
      'friend': False,
      'soldier': False,
      'side': False,
      'far': False,
      'jeff': False,
      'coulter': False,
      'establish': False,
      'appl': False,
      'thought': False,
      'melania': False,
      'flight': False,
      'agenda': False,
      'base': False,
      'favor': False,
      'union': False,
      'spend': False,
      'speaker': False,
      'agenc': False,
      'mysteri': False,
      'mock': False,
      'trust': False,
      'island': False,
      'brother': False,
      'thursday': False,
      'philippin': False,
      'fatal': False,
      'apolog': False,
      'super': False,
      'safe': False,
      'toward': False,
      'nearli': False,
      'lawyer': False,
      'employe': False,
      'number': False,
      'program': False,
      'benefit': False,
      'scott': False,
      'sea': False,
      'check': False,
      'prosecutor': False,
      'airstrik': False,
      'ukrain': False,
      'everyon': False,
      'mental': False,
      'moor': False,
      'ga': False,
      'assassin': False,
      'agre': False,
      'deni': False,
      'drone': False,
      'data': False,
      'tank': False,
      'liberti': False,
      'shift': False,
      'suggest': False,
      'account': False,
      'cross': False,
      'yemen': False,
      'moment': False,
      'european': False,
      'kushner': False,
      'clear': False,
      'hacker': False,
      'chart': False,
      'lawsuit': False,
      'daughter': False,
      'dream': False,
      'cartel': False,
      'might': False,
      'organ': False,
      'wife': False,
      'bust': False,
      'true': False,
      'oregon': False,
      'suspend': False,
      'brain': False,
      'pull': False,
      'cash': False,
      'democraci': False,
      'design': False,
      'suit': False,
      'donat': False,
      'someth': False,
      'approv': False,
      'aim': False,
      'coup': False,
      'promot': False,
      'wire': False,
      'green': False,
      'decis': False,
      'wear': False,
      'relat': False,
      'crowd': False,
      'king': False,
      'site': False,
      'tom': False,
      'clash': False,
      'flynn': False,
      'venezuela': False,
      'retir': False,
      'loss': False,
      'teacher': False,
      'seri': False,
      'duke': False,
      'troubl': False,
      'parent': False,
      'failur': False,
      'five': False,
      'audio': False,
      'constitut': False,
      'ann': False,
      'stock': False,
      'chelsea': False,
      'hire': False,
      'dutert': False,
      'insur': False,
      'sweden': False,
      'delay': False,
      'navi': False,
      'altern': False,
      'import': False,
      'progress': False,
      'address': False,
      'answer': False,
      'dnc': False,
      'pentagon': False,
      'success': False,
      'roger': False,
      'unveil': False,
      'potenti': False,
      'across': False,
      'sale': False,
      'prime': False,
      'burn': False,
      'teen': False,
      'council': False,
      'suicid': False,
      'orlando': False,
      'propaganda': False,
      'seen': False,
      'store': False,
      'ring': False,
      'god': False,
      'hidden': False,
      'yet': False,
      'jew': False,
      'kid': False,
      'door': False,
      'posit': False,
      'reform': False,
      'tillerson': False,
      'six': False,
      'limit': False,
      'fix': False,
      'medic': False,
      'took': False,
      'ambassador': False,
      'longer': False,
      'bridg': False,
      'pictur': False,
      'pressur': False,
      'corpor': False,
      'track': False,
      'bid': False,
      'natur': False,
      'what': False,
      'father': False,
      'cold': False,
      'michigan': False,
      'franci': False,
      'cabinet': False,
      'quit': False,
      'africa': False,
      'common': False,
      'arab': False,
      'afghanistan': False,
      'julian': False,
      'entir': False,
      'korean': False,
      'kim': False,
      'area': False,
      'educ': False,
      'mosqu': False,
      'remain': False,
      'drive': False,
      'tim': False,
      'indict': False,
      'nomin': False,
      'anonym': False,
      'note': False,
      'conspiraci': False,
      'chair': False,
      'stay': False,
      'eric': False,
      'player': False,
      'ceo': False,
      'movement': False,
      'largest': False,
      'stephen': False,
      'para': False,
      'focu': False,
      'wait': False,
      'afghan': False,
      'lynch': False,
      'convict': False,
      'marijuana': False,
      'hell': False,
      'ufo': False,
      'station': False,
      'oscar': False,
      'ted': False,
      'blow': False,
      'brooklyn': False,
      'condemn': False,
      'trip': False,
      'merkel': False,
      'jone': False,
      'develop': False,
      'worst': False,
      'websit': False,
      'hide': False,
      'involv': False,
      'phone': False,
      ...},
     0)



## Here comes the NLP and scikit-learn


```
#Split into training and testing for now 
# (we can maybe combine the two test and train.csv together later)
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

```

### Important imports 


```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
```

### So we can basically try all these to try to find the best one for us


```
#Define models to to train
names = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression', 'SGDClassifier', 'MultinomialNB', 'SVM Linear']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter= 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))
```


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
    /opt/venv/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    LogisticRegression: Acurracy: 92.14428857715431
    SGDClassifier: Acurracy: 90.92184368737475
    MultinomialNB: Acurracy: 86.27254509018036
    SVM Linear: Acurracy: 91.04208416833669



```

```
