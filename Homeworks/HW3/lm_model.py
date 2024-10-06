from collections import Counter
import numpy as np
import math

"""
CS 4120, Fall 2024
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  ngrams = []
  if n > len(tokens):
      return ngrams
  for token in range(len(tokens)-n+1):
      temp = tuple(tokens[token:token+n])
      ngrams.append(temp)
  return ngrams

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    self.n_gram = n_gram
    self.training_n_grams = []
    self.training_n_gram_counts = {}
    self.vocab = {}

  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # STUDENTS IMPLEMENT'
    verbose = False
    
    
    token_counts = Counter(tokens)
    #Chat-gpt helped generate the following line (made my implementation concise)
    tokens = [token if token_counts[token] > 1 else "<UNK>" for token in tokens]

    
    training_n_grams = create_ngrams(tokens, self.n_gram)
    training_n_gram_counts = Counter(training_n_grams)
    
    self.training_n_grams = training_n_grams
    self.training_n_gram_counts = training_n_gram_counts
    self.vocab = set(tokens)
    
     
    if verbose:
        print("Received tokens:", tokens)
        print("Created n_grams:", training_n_grams)
        print("Training_n_gram_counts:", training_n_gram_counts)
        print("Vocab:", self.vocab)
        
        

  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    sentence_token_counts = Counter(sentence_tokens)
    
    #Chat-gpt helped generate the following line (made my implementation concise)
    sentence_tokens = [token if token in self.vocab or token in [SENTENCE_BEGIN, SENTENCE_END] else "<UNK>" for token in sentence_tokens]
    
    to_score_n_grams = create_ngrams(sentence_tokens, self.n_gram)
    total_prob = 1
    
    
    #Get P(wi | w0, w1, ..., wi-1)
    n_minus_1_gram_counts = {}
    for n_gram in self.training_n_gram_counts:
        if n_gram[:-1] in n_minus_1_gram_counts:
            n_minus_1_gram_counts[n_gram[:-1]] += self.training_n_gram_counts[n_gram]
        else:
            n_minus_1_gram_counts[n_gram[:-1]] = self.training_n_gram_counts[n_gram]
            
            
    # Equation: (count of n_gram + 1) / (number of tokens + vocab size)
    for n_gram in to_score_n_grams:
        if self.n_gram == 1:
            denom = sum(self.training_n_gram_counts.values())
        else:
            denom = n_minus_1_gram_counts.get(n_gram[:-1], 0)
            
        numer = self.training_n_gram_counts[n_gram]
       
        total_prob *= (numer + 1) / (denom + len(self.vocab))
    return total_prob


  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    """
    For bigrams:
    • Sample a random bigram according to the probability distribution with
    <s> as its first element
    • Sample a new random bigram according to its probability where is fixed
    and has not yet been chosen.
    • Continue sampling a new bigram until is </s>
    """
    import random
    #-------------------------------------------------------------
    #ChatGPT-4 helped me out a lot in this section.
    #I asked ChatGPT to provide a rough outline of how the Shannon technique was implemented,
    #and adapted the code provided to fit this program.
    #I used AI to generate/modify many lines of code (in-line comments will reflect which ones)
    #I also used ChatGPT to debug some misc. statements
    #-------------------------------------------------------------
    
    #Handle unigrams
    sentence = []
    if self.n_gram == 1:
        while True:
            #Get the possible n-gram and 
            candidates = self.training_n_gram_counts
            total = sum(candidates.values())
            
            
            # Probability of each word to appear (i.e. this word / count of all words)
            probabilities = [count / total for count in candidates.values()]
            
            #ChatGPT helped with this section (generated the second line (almost) entirely)
            n_grams = list(candidates.keys())
            next_token = random.choices(n_grams, weights=probabilities, k=1)[0]
            
            #Break if we get a </s>, ignore any <s>
            if next_token[0] == SENTENCE_END:
                break
            if next_token[0] != SENTENCE_BEGIN:
                sentence.append(next_token[0])
        return sentence

    
    #Handle n-grams
    else:
        #Create a starting n-gram that looks like (<s>, <s>, <s>, ..., word1)
        current_n_gram = (SENTENCE_BEGIN,) * (self.n_gram - 1)
        for i in range(self.n_gram - 1):
            sentence.append(SENTENCE_BEGIN)
            
        while True:
            candidates = {n_gram: count for n_gram, count in self.training_n_gram_counts.items() if n_gram[:-1] == current_n_gram} #I asked chatGPT to generate a dictionary of n_grams that have the same sequence of tokens as the current sequence minus the last one. 
            total = sum(candidates.values())
            
            #Safety net
            if total == 0:
                break
                
            #ChatGPT helped with this section (generated the third line (almost) entirely)    
            probabilities = [count / total for count in candidates.values()]
            n_grams = list(candidates.keys())
            next_n_gram = random.choices(n_grams, weights=probabilities, k=1)[0]
            
            
            next_token = next_n_gram[-1]

            if next_token == SENTENCE_END:
                break

            sentence.append(next_token)
            #Update the current n-gram to reflect the newly added word
            current_n_gram = tuple(sentence[-(self.n_gram - 1):])
            
        #Return the sentence minus the leading <s>
        return sentence[self.n_gram - 1:]

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    #From lecture 6
    #PP(W) = P(w1,w2... wN)^(-1/N)
    return pow(self.score(sequence), -1/len(sequence))
  
# not required
if __name__ == '__main__':
  print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")