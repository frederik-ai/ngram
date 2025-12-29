from collections import defaultdict
from pathlib import Path
from typing import Iterable, List
import string
import random
import re

from tqdm import tqdm

from settings import SENTENCE_END_CHARS, SENTENCE_START_TOKEN, SENTENCE_END_TOKEN

class Corpus():
   """
   Dataset class which loads the utf-8 encoded text files stored at data/raw. Can iterate over objects of this class
   to successively obtain word-level tokens from the dataset.
   """
   def __init__(self):
      self.raw_file_paths = list(Path('data/raw').glob('*.utf-8'))
      # Add them to the folder data/processed
      self.file_paths = [str(raw_file_path).replace('raw', 'processed') for raw_file_path in self.raw_file_paths]
      self.number_of_lines_in_corpus = 0
      
      # Create preprocessed corpus files
      for i, raw_file_path in enumerate(self.raw_file_paths):
         self.preprocess_book(raw_file_path, self.file_paths[i])
   
   def preprocess_book(self, input_file: str, output_file: str) -> None:
      """
      Takes a utf-8 encoded text file of a book as input and writes out a preprocessed version. The file is expected to
      stem from project gutenberg. The preprocessing omits the project gutenberg header text and tail text, removes  blank 
      lines and \n chars that occur in the middle of a sentence as well as applying additional processing to the text.
      
      Link to Project Gutenberg: https://www.gutenberg.org/
      
      Args:
         input_file (str): Full or relative path to the utf-8 encoded text file which contains a book loaded from \
            project gutenberg.
         output_file (str): Full or relative path to the output text file which contains the preprocessed text.   
      """
      
      with open(input_file, mode='r') as file_in:
         with open(output_file, mode='w') as file_out:
            reached_book_start = False
            reached_book_end = False
            for line in file_in:
               # Check for book start and book end
               if not reached_book_start and line.startswith('*** START OF THE PROJECT GUTENBERG EBOOK'):
                  reached_book_start = True
                  continue
               elif not reached_book_end and line.startswith('*** END OF THE PROJECT GUTENBERG EBOOK'):
                  reached_book_end = True

               # Preprocess each line
               if reached_book_start and not reached_book_end:
                  
                  line_stripped = line.strip()
                  if line_stripped == '':
                     continue
                  # Remove all lines that start with "_"
                  elif line_stripped.startswith('_'):
                     continue
                  # Remove all lines that start with "_"
                  elif line_stripped.startswith('_'):
                     continue
                  # Remove table of content caption
                  elif line_stripped == 'CONTENTS':
                     continue
                  # Remove all letter and chapter captions
                  elif re.match(r'^(chapter|letter)\s+\d+', line_stripped.lower()):
                     continue
                  
                  # Only keep \n char if the end of this line marks the end of a sentence.
                  if not line_stripped[-1] in SENTENCE_END_CHARS:
                     line = line_stripped + ' '
                  else:
                     self.number_of_lines_in_corpus += 1
                  
                  # Remove quotation marks
                  line = line.replace('“', '')
                  line = line.replace('”', '')
                  # Remove underscores
                  line = line.replace('_', '')
                  
                  line = line.replace("’", "'")
                  file_out.write(line)
         
   def tokenize(self, str) -> List[str]:
      """
      Tokenize an input string. If the end of a sentence is detected, an end of sentence token is inserted.
      
      Args:
         str (str): Input string
      Returns:
         A list of strings whereas each list element is a token.
      """
      tokens = str.split()
      tokens_processed = []
      for token in tokens:
         # We make each special character at the start and the end of the word their own token.
         special_chars_before_word = []
         special_chars_after_word = []
         
         while token and token[0] in string.punctuation:
            special_chars_before_word.append(token[0])
            token = token[1:]
         while token and token[-1] in string.punctuation:
            special_chars_after_word.append(token[-1])
            # Insert end of sentence tag
            if token[-1] in SENTENCE_END_CHARS:
               special_chars_after_word.append(SENTENCE_END_TOKEN)
            token = token[:-1]
            
         tokens_processed.extend(special_chars_before_word)
         tokens_processed.append(token)
         tokens_processed.extend(special_chars_after_word)
         
      return tokens_processed
   
   def __iter__(self) -> Iterable[List[str]]:
      """
      Iterate over the corpus. In each iteration, yields a list of tokens.
      """
      for file_path in self.file_paths:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
               line = line.lower()
               tokens = self.tokenize(line)
               yield tokens
   
   def __len__(self) -> int:
      """
      Returns the number of lines (lists of tokens) in the corpus.
      """
      return self.number_of_lines_in_corpus

class NGramModel():
   def __init__(self, corpus: Corpus, n=1):
      self.n = n
      self.ngram_dict = self.init_ngram_dict(corpus)
      
   def init_ngram_dict(self, corpus):
      ngram_dict = defaultdict(lambda: defaultdict(int))
      
      for tokens in tqdm(corpus, desc='Initializing NGram Model'):
         # Add padding
         tokens = [SENTENCE_START_TOKEN] * (self.n-1) + tokens
         
         for i in range(self.n-1, len(tokens)):
            context_start_idx = i - self.n + 1
            context = tokens[context_start_idx:i]
            context = tuple(context)
            word = tokens[i]
            
            ngram_dict[context][word] += 1
         
      # turn counts into probabilities
      for context, word_counts in ngram_dict.items():
         total = sum(word_counts.values())
         for word in word_counts:
            word_counts[word] /= total
      
      return ngram_dict
      
   def generate_token(self, context: tuple):
      probability = random.random()
      probability_sum = 0.0
      for next_word in self.ngram_dict[context].keys():
         probability_sum += self.ngram_dict[context][next_word]
         if probability_sum >= probability:
            return next_word
      raise Exception('No token was generated')
      
   def __call__(self):
      
      sentence = ''
      context = [SENTENCE_START_TOKEN] * (self.n-1)
      while True:
         token = self.generate_token(tuple(context))
         
         # update context (append generated token and remove first token)
         context.append(token)
         context = context[1:]
         
         if token == SENTENCE_END_TOKEN:
            break
         
         if token in set(string.punctuation):
            sentence = sentence + token
         elif sentence == '': # if beginning of the sentence
            sentence = token.title()
         else:
            sentence = sentence + ' ' + token
      
      return sentence
      
if __name__ == '__main__':
   corpus = Corpus()
   ngram = NGramModel(corpus=corpus, n=4)
   for i in range(20):
      print(ngram())