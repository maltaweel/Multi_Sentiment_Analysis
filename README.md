# Multi_Sentiment_Analysis

This tool provides sentiment analysis using VADER, Flair, and TextBlob. In addition, a lexical list is incorporated as part of the 
multi-sentiment analysis approach. The lexical list incorporated is from a previous publication and use case:  https://doi.org/10.1371/journal.pone.0197816.

The code is provided in /src/sentiment/analysis.py. This module combines all the libraries above and provides output to facilitate assessment of sentiment data.
Polarity, subjectivity, and aspect mining are provided using TextBlobl and Flair. Aspect mining is available from NLTK. Flair applies the sentiment-en-mix-distillbert_4.pt model as default. The module can be adjusted to run on GPU. To run the analysis.py module, simpy execute via the 'python analysis.py' command. Required libraries should be installed; requirements.txt has libraries used and that can be installed using the 'pip -r' command. 

<b>Data Folders</b>

Input data should be placed in the /modified folder. 

<b>Output Results</b>

Output will be placed in the /sentiment folder with the first part of the output files starting with 'sentiment_classification'. These files, like the input,
are in .csv format. Output results can be found here:  https://drive.google.com/file/d/103CmqtUDwcaTDzPhTvCkupmzYkuXeU2x/view?usp=drive_link.

<b>Input Lexicon</b>

In addition to lexicons provided by applied libraries, we also incorporate a lexical list (see link above). The data were downloaded and placed in the /lexicon folder.

<b>Key Methods</b>

The following methods are in analysis.py.

def readLexiconList: reads in input lexical information.

def scoreFlair: Provides sentiment using the Flair library

def textAspect: provides TextBlob and Flair sentiment scores

def main:  Main method that conducts the analysis for different sentiment libraries and organises the results, enabling them to be printed out in the output results. 

