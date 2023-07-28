# Multi_Sentiment_Analysis

This tool provides sentiment analysis using VADER, Flair, and TextBlob. In addition, a lexical list is incorporated as part of the 
multi-sentiment analysis approach. The lexical list incorporated is from a previous publication and use case:  https://doi.org/10.1371/journal.pone.0197816.

The code is provided in /src/sentiment/analysis.py. This module combines all the libraries above and provides output to facilitate assessment of sentiment data.
Polarity, subjectivity, and aspect mining are provided using TextBlobl and Flair. Aspect mining is available from NLTK. Flair applies the sentiment-en-mix-distillbert_4.pt model as default. The module can be adjusted to run on GPU. To run the analysis.py module, simpy execute via the 'python analysis.py' command. Required libraries should be installed; requirements.txt has libraries used and that can be installed using the 'pip -r' command. 

