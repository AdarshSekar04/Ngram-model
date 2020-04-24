#Ngram.py
#Mamon Alsalihy and Adarsh Sekar
#Mamon ID No. = 1545777
#Adarsh ID No. = 1619894

import fileinput,sys,math
# Counter class is a dict subclass for counting hashable objects
from collections import Counter

# The way each model will work is different, as the probabilities have different dependencies.
# So, to take care of such problems, we are going to create different classes for each model, as each model will work differently on the input


class UnigramModel:
    # Constructor
    def __init__(self, text):
        # Store the number of lines into the var num_of_lines
        num_of_lines = text.count('\n')
        # Creates a new Counter, with the words from text forming a dictionary
        self.dict_corpus = Counter(text.split())
        # Sum of the values of all the counts of words into num_of_words_in_corpus. Now we have the number of words in num_of_words_in_corpus.
        self.num_of_words_in_corpus = sum(self.dict_corpus.values())

        # Set a counter to 0
        counter = 0
        # For every key, value in the dictionary, we are gonna delete the keys with a value less than 3, and add their counts to our counter
        for key, value in list(self.dict_corpus.items()):
            if value < 3:
                counter += value
                del self.dict_corpus[key]
        # Create a key UNK, and set it's value to counter
        self.dict_corpus['<UNK>'] += counter
        # Storing the number of unique keys in a variable called length. Subtracting 1 for the <STOP>
        self.length = len(self.dict_corpus.keys()) - 1
        # Adding '<STOP>' to the dictionary, and setting its count to the number of lines
        self.dict_corpus['<START>'] = num_of_lines
        self.dict_corpus['<STOP>'] = num_of_lines
        # By the end of the constructor, we should have constructed the dictionary we need for a unigram model.
    # Function that will calculate the probability of a word given, using the dictionary created from training data.

    def wordProbability(self, word):
        # Check the count of the word we're looking for
        numerator = self.dict_corpus[word]
        # If the word is not there, set the numerator to the value of '<UNK>'
        if numerator == 0:
            numerator = self.dict_corpus['<UNK>']
        # Set the denominator to the length of the dictionary, i.e, the number of unique words including '<UNK>'
        denominator = self.num_of_words_in_corpus
        to_return = float(numerator)/float(denominator)
        return to_return

    # Finally, the probability of a sentence given the Unigram Model
    def sentenceProbability(self, sentence, word_count_in_each_sentence):
        # What we want to do is find the product of all the individual probabilities of each word.
        # So, first we are going to need to create a variable to store the product of these probabilities
        totalP = 1
        # For every word in the text
        for word in sentence.split():
            # We want to find the probability of every word
            wordP = self.wordProbability(word)
            totalP += -math.log2(wordP)
            word_count_in_each_sentence += 1
        # Now the total P of the sentence should be in totalP
        return (totalP, word_count_in_each_sentence)

# ---------------------------------
# Bigram Class (Subclass of Unigram)
# ---------------------------------


class BigramModel(UnigramModel):
    # Constructor
    def __init__(self, text):
        # Call the super class constructor so that the Bigram has the same variables that we don't have to redefine
        UnigramModel.__init__(self, text)
        # Difference between Bigram and Unigram is that Bigram depends on the previous word.
        # Store the number of lines into the var num_of_lines
        split_text = text.split('\n')
        # Creating a an empty Counter
        self.dict_corpus = Counter()
        for sentence in split_text:
            word_list = sentence.split()
            # Every sentence is a fresh start, as it's the start of a new sentence. Therefore the word preceding the start of the new sentence should not matter.
            previous = None
            # For every ordered pair of words, we increment the count, adding it to the dictionary it it isn't already there
            for word in word_list:
                self.dict_corpus[(previous, word)
                                 ] = self.dict_corpus[(previous, word)] + 1
                # Now that we've added all the words in the sentence to our dictionary, we can change our previous to the current word.
                previous = word
        # Now we should have the ordered pairs of words, including if the pair isn't frequent enough, in which case it is an UNK
        self.length = len(self.dict_corpus)
        self.num_of_words_in_corpus = sum(self.dict_corpus.values())
        self.uni = UnigramModel(text)

    # Function that returns the probability that a given pair of words is likely to appear in that order
    # Function will take in a previous word, and the current word)
    def pairProbability(self, previous, current):
        # Getting the count of the ordered pair
        numerator = self.dict_corpus[(previous, current)]
        denominator = 0
        # If the pair doesn't exist. or there is never a word following the 'previous' we are given, the probability of that is 0
        denom_previous = previous
        if numerator == 0:
            return 1
        # For every key in the dictionary, we are going to sum up the probabilities of all values where previous is there first element of the key.
        denominator = self.uni.dict_corpus[denom_previous]
        if denominator == 0:
            print("Error: pair = ",denom_previous,numerator)
            sys.exit(0)
        prob_without_smoothing = numerator/denominator
        prob_with_smoothing = self.pairSmoothing(prob_without_smoothing,current)
        # Return the probability of the pair of words
        return prob_with_smoothing

    def pairSmoothing(self,prob_without_smoothing,current):
        hyper_param_uni = 0.1
        hyper_param_bi = 0.3
        prob_with_smoothing = (hyper_param_uni * uni.wordProbability(current)) + (hyper_param_bi * prob_without_smoothing)
        return prob_with_smoothing

    def bisentenceProb(self, sentence, word_count_in_each_sentence):
        # Set the totalP as 1. We are going to return this after we multiply all the probabilities of every word
        totalP = 1
        previous = None
        for word in sentence.split():
            if word != '<START>' and word != '<STOP>':
                totalP += -math.log2(self.pairProbability(previous, word))
                word_count_in_each_sentence += 1
            previous = word
        # By end of the for loop, we should have the total Probability of the sentence in totalP. Now we return it
        return (totalP, word_count_in_each_sentence)

# -------------
# Trigram Class
# -------------


class TrigramModel(UnigramModel):
    # Constructor. Going to be very similar to bigram with just a few changes
    # Constructor
    def __init__(self, text):
        # Call the super class constructor so that the Bigram has the same variables that we don't have to redefine
        UnigramModel.__init__(self, text)
        # Difference between Bigram and Unigram is that Bigram depends on the previous word.
        # Store the number of lines into the var num_of_lines
        split_text = text.split('\n')
        # Creating a an empty Counter
        self.dict_corpus = Counter()
        for sentence in split_text:
            word_list = sentence.split()
            # Every sentence is a fresh start, as it's the start of a new sentence. Therefore the word preceding the start of the new sentence should not matter.
            two_previous = None
            one_previous = None
            # For every ordered pair of words, we increment the count, adding it to the dictionary it it isn't already there
            for word in word_list:
                self.dict_corpus[(two_previous, one_previous, word)] = self.dict_corpus[(
                    two_previous, one_previous, word)] + 1
                # Now that we've added all the words in the sentence to our dictionary, we can change our previous to the current word.
                two_previous = one_previous
                one_previous = word
        # Now we should have the ordered pairs of words, including if the pair isn't frequent enough, in which case it is an UNK
        self.length = len(self.dict_corpus)
        self.num_of_words_in_corpus = sum(self.dict_corpus.values())
        self.bi = BigramModel(text)

        # Function will take in a previous word, and the current word)
    def trioProbability(self, second_previous, first_previous, current):
        # Getting the count of the ordered pair
        numerator = self.dict_corpus[(second_previous, first_previous, current)]
        if numerator == 0:
            return 1
        denominator = 0
        # For every key in the dictionary, we are going to sum up the probabilities of all values where both the previous words are there as the two previous and one previous words.
        my_dict = self.bi.dict_corpus
        denominator = my_dict[(second_previous, first_previous)]
        if denominator == 0:
            print("Error")
            sys.exit(0)
        prob_without_smoothing = numerator/denominator
        prob_with_smoothing = self.trioSmoothing(prob_without_smoothing,first_previous,current) 
        # Return the probability of the pair of words
        return prob_with_smoothing

    def trioSmoothing(self,prob_without_smoothing,first_previous,current):
        hyper_param_uni = 0.1
        hyper_param_bi = 0.3
        hyper_param_tri = 0.6
        prob_with_smoothing = (hyper_param_uni * uni.wordProbability(current)) + (hyper_param_bi * bigram.pairProbability(first_previous,current)) + (hyper_param_tri * prob_without_smoothing)
        return prob_with_smoothing

    def trisentenceProb(self, sentence, word_count_in_each_sentence):
        # Set the totalP as 1. We are going to return this after we multiply all the probabilities of every word
        totalP = 1
        one_previous = None
        two_previous = None
        for word in sentence.split():
            if word != '<START>' and word != '<STOP>':
                totalP += - \
                    math.log2(self.trioProbability(
                        two_previous, one_previous, word))
                word_count_in_each_sentence += 1
            two_previous = one_previous
            one_previous = word
        # By end of the for loop, we should have the total Probability of the sentence in totalP. Now we return it
        return (totalP, word_count_in_each_sentence)


# Opening the training file, and storing the reference to it in file
train_file_descriptor = open(
    "A1-Data/1b_benchmark.train.tokens", encoding="utf-8")
# Read the file into the variable train_data
train_data = train_file_descriptor.read()
# Closing the file. Will open it in write mode later
train_file_descriptor.close()
dev_file_name = "A1-Data/1b_benchmark.dev.tokens"
test_file_name = "A1-Data/1b_benchmark.test.tokens"
# ---------------
# Testing Unigram
# ---------------
print("Testing Unigram")
uni = UnigramModel(train_data)
# Store the input file in read mode as f
dev_file_descriptor = open(dev_file_name, encoding="utf-8")
test_file_descriptor = open(test_file_name, encoding="utf-8")
# Read the lines of the text into text_lines
dev_data = dev_file_descriptor.read()
test_data = test_file_descriptor.read()
list_of_sentences_in_dev = dev_data.splitlines()
list_of_sentences_in_test = test_data.splitlines()
word_count_in_each_sentence = sum_of_logs = total_words_in_test = 0
for sentence in list_of_sentences_in_dev:
    P_of_sentence = uni.sentenceProbability(
        sentence, word_count_in_each_sentence)
    sum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of unigram for dev = ", 2**(sum_of_logs/total_words_in_test))
word_count_in_each_sentence = sum_of_logs = total_words_in_test = 0
for sentence in list_of_sentences_in_test:
    P_of_sentence = uni.sentenceProbability(
        sentence, word_count_in_each_sentence)
    sum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of unigram for test = ", 2**(sum_of_logs/total_words_in_test))
word_count_in_each_sentence = sum_of_logs = total_words_in_test = 0
#Getting the list of sentences in train
list_of_sentences_in_train = train_data.splitlines()
for sentence in list_of_sentences_in_train:
    P_of_sentence = uni.sentenceProbability(
        sentence, word_count_in_each_sentence)
    sum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of unigram for train = ", 2**(sum_of_logs/total_words_in_test))


def replacingOOVInData(list_of_sentences):
                # We should now have every sentence
    for index_of_sentence, sentence in enumerate(list_of_sentences):
                # We want to split every sentence into words
        words = sentence.split()
        words.insert(0, '<START>')
        words.append('<STOP>')
        for index_of_word, word in enumerate(words):
            # If the word is not in our unigram dictionary, it's count is too low so we want to change it to UNK
            if uni.dict_corpus[word] == 0:
                words[index_of_word] = '<UNK>'
        # Now, all the infrequent words should be changed to UNK. Change the set of words back to a sentence
        space = ' '
        list_of_sentences[index_of_sentence] = space.join(words)
    newline = '\n'
    # Now we should have the edited text in readfile
    readfile = newline.join(list_of_sentences)
    return readfile


# Over here, we're gonna modify the file
# First, we're gonna split the file per newline
list_of_sentences_in_train = train_data.splitlines()
replaced_data_in_train = replacingOOVInData(list_of_sentences_in_train)
# We also have to change the test files also, so we will do the same thing for the test file and dev file below
dev_file_descriptor.close()
test_file_descriptor.close()
dev_file_descriptor = open(dev_file_name, encoding="utf-8")
dev_data = dev_file_descriptor.read()
dev_file_descriptor.close()
test_file_descriptor = open(test_file_name, encoding="utf-8")
test_data = test_file_descriptor.read()
test_file_descriptor.close()

list_of_sentences_in_dev = dev_data.splitlines()
list_of_sentences_in_train = test_data.splitlines()
replaced_data_in_dev = replacingOOVInData(list_of_sentences_in_dev)
replaced_data_in_test = replacingOOVInData(list_of_sentences_in_train)
# --------------
# Testing Bigram
# --------------
print("\n\nTesting Bigram")
bigram = BigramModel(replaced_data_in_train)
length = bigram.length
# Read the lines of the text into text_lines
list_of_sentences_in_dev = replaced_data_in_dev.splitlines()
lineNum = 1
text_perpexlity = 1.0
bisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for sentence in list_of_sentences_in_dev:
    P_of_sentence = bigram.bisentenceProb(
        sentence, word_count_in_each_sentence)
    bisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of bigram for dev = ", 2**(bisum_of_logs/total_words_in_test))
bisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for sentence in list_of_sentences_in_test:
    P_of_sentence = bigram.bisentenceProb(
        sentence, word_count_in_each_sentence)
    bisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of bigram for test = ", 2**(bisum_of_logs/total_words_in_test))
bisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for sentence in list_of_sentences_in_train:
    P_of_sentence = bigram.bisentenceProb(
        sentence, word_count_in_each_sentence)
    bisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of bigram for train = ", 2**(bisum_of_logs/total_words_in_test))

# #---------------
# Testing Trigram
# ---------------
print("\n\nTesting Trigram")
trigram = TrigramModel(replaced_data_in_train)
length = trigram.length
# Store the input file in read mode as f

# Read the lines of the text into text_lines
list_of_sentences_in_dev = replaced_data_in_dev.splitlines()
text_perpexlity = 1.0
trisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for line in list_of_sentences_in_dev:
    P_of_sentence = trigram.trisentenceProb(line, word_count_in_each_sentence)
    trisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of trigram for dev = ", 2**(trisum_of_logs/total_words_in_test))
trisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for line in list_of_sentences_in_test:
    P_of_sentence = trigram.trisentenceProb(line, word_count_in_each_sentence)
    trisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of trigram for test = ", 2**(trisum_of_logs/total_words_in_test))
trisum_of_logs = word_count_in_each_sentence = total_words_in_test = 0
for line in list_of_sentences_in_train:
    P_of_sentence = trigram.trisentenceProb(line, word_count_in_each_sentence)
    trisum_of_logs += P_of_sentence[0]
    total_words_in_test += P_of_sentence[1]
print("Perplexity of trigram for train = ", 2**(trisum_of_logs/total_words_in_test))




