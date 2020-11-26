# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:17:16 2020

@author: jeong
"""

from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
import pandas as pd
import numpy as np
import gc, pickle

#%% tokenizer model 
# voc1.csv: 기준년월, 연체상담일자, 상담내용으로 구성된 data.frame
voc1 = pd.read_csv('d:/work/voc1.csv')
voctxt = '\n'.join([t for t in voc1.sample(frac=0.3).text.dropna()])
fl = open('d:/work/voctxt.txt','w',encoding='utf-8')
fl.write(voctxt)
fl.close()

sents = DoublespaceLineCorpus('d:/work/voctxt.txt',iter_sent=True)

word_extractor = WordExtractor()
word_extractor.train(sents)
word_score = word_extractor.extract()

scores = {word:score.cohesion_forward for word,score in word_score.items()}
l_tokenizer = LTokenizer(scores = scores)

with open('d:/work/l_tokenizer','wb') as tokp:
    pickle.dump(l_tokenizer,tokp)
    
docs = [' '.join(l_tokenizer.tokenize(t)) for t in voc1.text]
with open('d:/work/vocdocs.txt','wb') as fp:
    pickle.dump(docs, fp)

#%% doc2vec model 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
tagged = [TaggedDocument(words=l_tokenizer.tokenize(t),tags=[str(i)]) \
        for i,t in enumerate(voc1.sample(frac=0.3).text)]
model = Doc2Vec(vector_size=100, epochs=30,min_count=2)
model.build_vocab(tagged)
model.train(tagged.total_examples=model.corpus_count,epochs=model.epochs)
model.save('d:/work/vocd2v.model')