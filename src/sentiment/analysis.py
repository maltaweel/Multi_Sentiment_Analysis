'''
This provides sentiment analyses using various libraries that include VADER, 
TextBlob, and Flair.

Created on Jan 28, 2021

@author: 
'''

#basic libraries for the system + flair
import re
import os
from os import listdir
import csv
import flair, torch

#nltk and pandas
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd

#textblob
from textblob import TextBlob

#more flair + nltk; spacy to load spacy cnn model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from segtok.segmenter import split_single
import spacy
nlp = spacy.load("en_core_web_sm")

#spacy.prefer_gpu()
#analyser = SentimentIntensityAnalyzer()

#default pathway for src and subfolders
pn=os.path.abspath(__file__)
pn=pn.split("src")[0] 

#model data folder (models)
#data=os.path.join(pn,'models','best-model.pt')

#text classification for sentiment
classifier = TextClassifier.load('sentiment')

#lexicon location
lexicon=os.path.join(pn,'lexicon','wn-asr-multicorpus.csv')

#English stop words
stop_words = set(stopwords.words('english')) 
#flair.device = torch.device('cuda')

'''
Method for reading text for lexicon list (see PlosOne article)
@param file_l- lexicon list to read from
@return terms- terms and scores for sentiment
'''
def readLexiconList(file_l):
    
    #read file
    df = pd.read_csv(file_l)
    words=df['Word']
    scores=df['Category']
    counts=df['Count']
    ratings=df['Rating']
    
    terms={}
    
    ii=0
    ranges=[]
    sets=[]
    
    #go over words in file and get scores and counts to add to lexicon
    for word in words:
        w=word.split('/')[0]
        score=scores[ii]
        count=counts[ii]
        rating=ratings[ii]
        
        #ratings adjust scores
        if rating==1:
            ranges.clear()
            sets.clear()
        
        ranges.append(count)
        sets.append(score)
        
        if rating==5:
            s=sum(ranges)
            ts=0.0
            for i in range(0,5):
                cnts=ranges[i]
                scs=sets[i]
                
                ts+=scs*cnts
            
            #assign term scores
            value=float(ts/s)
            terms[w]=value
            
        ii+=1  
        
    return terms

'''
This method takes a list of terms and adds them to a lexicon

@param terms- the terms to add to the lexicon
@return SIA- the lexicon to return from a sentiment intensity analyser
'''
def addTerms(terms):
    SIA = SentimentIntensityAnalyzer() 
    SIA.lexicon.update(terms)
    
    return SIA

'''
This method scores polarity of sentences using the sentiment intensity
analyser polarity in NLTK.
@param sentence- the sentence to score
@param score- the polarity score
'''
def sentiment_analyzer_scores(sentence):
    analyser=SentimentIntensityAnalyzer() 
    score = analyser.polarity_scores(sentence)
    return score

'''
This method tokenizes and cleans text for analysis.
@param text- the text to tokenize
@return text- the cleaned up text
'''
def tokenizer(text: str) -> str:
    "Tokenize input string using a spacy pipeline"
#    nlp = spacy.blank('en')
    #strip and clean from stopwords
    tokenized_text = word_tokenize(text)
    tokenized_text = [w for w in tokenized_text  if not w in stop_words]
    tokenized_text = [w.strip() for w in tokenized_text]
    tokenized_text = [w for w in tokenized_text if not w==''] 
    tokenized_text = ' '.join(token for token in tokenized_text)
    
    #replace various text punctuations and formats
    text=text.replace(',','')
    text=text.replace('"','')
    text=text.replace('``','')
    text=text.replace(';','')
    text=text.replace(':', '')
    text=text.replace('&amp;', '')
    text=text.replace('.', '')
    text=text.replace('/', '')
    text=text.replace('[', '')
    text=text.replace('!', '')
    text=text.replace(']', '')
    text = text.replace('(', '') 
    text = text.replace(')', '')
    text = text.replace('?', '')
    text = text.replace("'", '')
 
    
    #nlp.add_pipe('sentencizer')  # Very basic NLP pipeline in spaCy
    #doc = nlp(text)
    
    #tokenized_text = ' '.join(token.text for token in doc)
    #tokenized_text = [w for w in tokenized_text if len(w) > 2]
    
    return text

'''
This method applies flair scoring for sentiment.
@param text- the text to score
@param model- the flair model to use for scoring
@return sentence.labels- the sentence scores and values
'''
def scoreFlair(text, model):
    #prior to scroing the text, just some basic housekeeping and cleaning of text in case needed
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', text)
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    
    #evaluate sentence using the flair model
    sentence = Sentence(result)
    model.eval()
    model.predict(sentence)
    
    #return results
    return sentence.labels

'''
Method to perform aspect from TextBlob and Flair + TextBlob polarity and sentiment from flair.
@param aspects- the container for the text to do aspect
@return aspectOut- the results of the aspect analysis
'''
def textAspect(aspects):
    asps=[]
    
    #get text for aspect analaysis using Flair and TextBlob
    for aspect in aspects:
        text=aspect['description']
        if text=='':
            continue
        aspect['blob_sentiment'] = TextBlob(text).sentiment
        aspect['flair_sentiment']=scoreFlair(text, classifier)
        asps.append(aspect)
   
    #this adds polarity and sentiment using flair
    aspectOut=assess_from_aspect(asps)
    return aspectOut

'''
Method to output TextBlob polarity, subjective score, and sentiment from Flair
to container to be returned.
@param asps-the aspect data to evaluate
@return result- container that has Flair sentiment and TextBlob polarity/subjectivity
'''
def assess_from_aspect(asps):
    result={}
    text=''
    blob_subj=0.0
    blob_polar=0.0
    flair_sent=0.0
    
    #read in text, polarity and subjectivity
    n=0
    for i in asps:
        text+=i['description']+'/'
        sentiment=i['blob_sentiment']
        blob_polar+=sentiment.polarity
        blob_subj+=sentiment.subjectivity
        flair=i['flair_sentiment']
        
        #get flair score
        dd=flair[0]._value.split(',')[0]
        sd=flair[0].score
        if dd=='NEGATIVE':
            sd=-sd
        flair_sent+=sd
        n+=1
    
    #adjust values if n>0 (i.e., there is text)
    if n>0:
        blob_subj=blob_subj/float(n)
        blob_polar=blob_polar/float(n)
        flair_sent=flair_sent/float(n)
    
    #put results in result container 
    result['description']=text
    result['blob_subjective']=blob_subj
    result['blob_polarity']=blob_polar
    result['flair_sentiment']=flair_sent
    
    return result

'''
Method to do aspect mining of sentences.
@param sentence- the sentence to aspect mine
@return aspect- the aspect result for given terms
'''
def aspectMine(sentence):
    aspect = []
    
    #get info about sentences using nlp
    doc = nlp(sentence)
    descriptive_term = ''
    target = ''
    
    #get the tokens (adv, nouns, etc.) from sentences
    for token in doc:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            target = token.text
        if token.pos_ == 'ADJ':
            prepend = ''
            for child in token.children:
                if child.pos_ != 'ADV':
                    continue
                prepend += child.text + ' '
            descriptive_term = prepend + token.text
        if descriptive_term=='':
            continue
        
        #put aspect results to return
        aspect.append({'aspect': target,
            'description': descriptive_term})
    
    return aspect

'''
Method to parse into sentences from a given text.
@param text- the text to parse
@return sentences- the sentences from the text
'''
def make_sentences(text):
    """ Break apart text into a list of sentences """
    sentences = [sent for sent in split_single(text)]
    return sentences

'''
This method is launched by the main clause (below). It conducts all the steps of the anlaysis and results
in output that provide sentiment for the analyses conducted.
'''
def main() -> None:
    
    #path to the files to read (in the modified folder)    
    file_read=os.path.join(pn,'modified')
    
    #fieldnames to read from the files to read  
    fieldnames = ['created_time','_id','positive','negative','neutral',
                  'compound','subjectivity','flair','flair score','aspect terms','aspect tb polarity',
                  'aspect tb subjectivity','aspect flair','lexicon score','sentence']
    
    #read terms from the lexicon
    terms_lexicon=readLexiconList(lexicon)
    
    #add terms to the lexicon
    sia=addTerms(terms_lexicon)
    
    #create the output files in the sentiment folder based on input files read
    for f in listdir(file_read):
        name=f.split('.csv')[0]
        fileOutput=os.path.join(pn,"sentiment","sentiment_classification"
                                +'_'+name+'.csv')
        
        with open(fileOutput, 'w') as csvf:
            
            #write the output
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            writer.writeheader()  
            
            
        #open file to read to do analysis
            with open(os.path.join(file_read,f),'r',encoding="ISO-8859-1") as csvfile:
                reader = csv.DictReader(csvfile)
            
                
                #read the rows
                for row in reader:
                    
                    #output data
                    printData={}
                
                    #get the tweet text
                    text=row['sentence']
                    sent=make_sentences(text)
                    i=0
                    
                    #read data from sentences, tokenize and score for sentiment
                    for te in sent:
                        try:
                            te = tokenizer(te)  # Tokenize text using spaCy before explaining
                            if te=='':
                                continue
                            #print("Generating LIME explanation for example {}: `{}`".format(i+1, text))
                            
                            #score sentiment           
                            score = sentiment_analyzer_scores(te)
                            #polarity
                            try:
                                t_score=sia.polarity_scores(te)['compound']
                            
                            except ZeroDivisionError:
                                t_score=0.0
                            
                            #textblob score
                            scoreS=TextBlob(te).sentiment.subjectivity
                            
                            #flair score
                            flair=scoreFlair(te,classifier)
                            
                            #remove \n
                            te=te.replace('\n',' ')
                            
                            #aspect score
                            aspect=aspectMine(te)
                            asps=textAspect(aspect)
                            printData['aspect']=asps
                            
#                           exp = explainer.explainer('fasttext', data, text, 1000)
                            #p=exp.score
                            
                            #get sentiment scores for positive, negative, netural
                            printData['positive']=score['pos']
                            printData['negative']=score['neg']
                            printData['neutral']=score['neu']
                            
                            #subjectivity score
                            printData['subjectivity']=scoreS
                            
                            if flair is None or len(flair)<0:
                                continue
                            
                            #determine flair score
                            try:
                                dd=flair[0]._value.split(',')[0]
                                sd=flair[0].score
                                if dd=='NEGATIVE':
                                    sd=-sd
                                
                            
                            except IndexError:
                                print(flair)
                                continue
                            
                            #now get sentiment output data for printing results   
                            printData['flair']=dd
                            printData['lexicon score']=t_score
                            printData['flair_score']=sd
                            printData['compound']=score['compound']
                            printData['created_time']=row['created_time']
                            try:
                                printData['_id']=row['_id']
                            except:
                                printData['_id']=row['from_id']
                            
                            printData['sentence']=te
                            output(writer,printData)
                            i+=1
                        
                        except ValueError as err:
                            print(err)
                            continue
                                       
                print(f)    
            csvf.close()
            
'''
    Method to output and print sentiment for individual texts.
    @param data- the data for given texts
    @param fileOutput- the file to output data
'''                 
def output(writer,f):
    result=f['aspect']
    
    #get descriptions, subjectivity, polarity, flair sentiment, lexicon scores
    asptext=result['description']
    blob_subj=result['blob_subjective']
    blob_polar=result['blob_polarity']
    flair_sent=result['flair_sentiment']
    lexicon_score=f['lexicon score']
    
    #write out row data + add other sentiment scores to output        
    writer.writerow({'created_time': str(f['created_time']),
            '_id':str(f['_id']),'positive':str(f['positive']),
            'negative':str(f['negative']),'neutral':str(f['neutral']),'compound':str(f['compound']),
            'subjectivity':str(f['subjectivity']),'flair':str(f['flair']),'flair score':str(f['flair_score']),
            'aspect terms':asptext,'aspect tb polarity':str(blob_polar),'aspect tb subjectivity':str(blob_subj),
            'aspect flair':str(flair_sent),'lexicon score':str(lexicon_score),'sentence':str(f['sentence'])})

'''
The main launch
'''        
if __name__ == "__main__":
    # evaluate text and sentiment
    main()
    
    print("Finished") 
