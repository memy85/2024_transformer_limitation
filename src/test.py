
import sys

import spacy
import medspacy
import nltk

from medspacy.util import DEFAULT_PIPENAMES
from medspacy.visualization import visualize_ent
from medspacy.section_detection import Sectionizer
medspacy_pipes = DEFAULT_PIPENAMES.copy()

if 'medspacy_quickumls' not in medspacy_pipes:
    medspacy_pipes.add('medspacy_quickumls')
   
print(medspacy_pipes)
   
nlp = medspacy.load(enable = medspacy_pipes, quickumls_path='/home/zhichaoyang/medspacy_test/')
print(nlp.pipe_names)
