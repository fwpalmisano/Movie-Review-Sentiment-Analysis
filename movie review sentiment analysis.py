###########
#
# Note: for the example and problem below, you will need
#       three of the "language corpora" (large sourcetexts)
#       from NLTK. To make sure you've downloaded them,
#       run the following:
#
# In[1]: import nltk
# In[2]: nltk.download()
#
# This will open a window. From there, click on the Corpora tab,
# and then double-click to download and install these three corpora:
#
# names
# movie_reviews
# opinion_lexicon
#
###########

import nltk
from nltk.corpus import names  # see the note on installing corpora, above
from nltk.corpus import opinion_lexicon
from nltk.corpus import movie_reviews
import textblob

import random
import math

from sklearn.feature_extraction import DictVectorizer
import sklearn
import sklearn.tree
import sklearn.ensemble
from sklearn.metrics import confusion_matrix

#####################
#
# Problem 4: Movie Review Sentiment starter code...
#
#####################

# a boolean to turn on/off the movie-review-sentiment portion of the code...
RUN_MOVIEREVIEW_CLASSIFIER = True
if RUN_MOVIEREVIEW_CLASSIFIER == True:

        # Read all of the opinion words in from the nltk corpus.
        #
    pos = list(opinion_lexicon.words('positive-words.txt'))
    neg = list(opinion_lexicon.words('negative-words.txt'))

    # Store them as a set (it'll make our feature extractor faster).
    #
    pos_set = set(pos)
    neg_set = set(neg)

    # Read all of the fileids in from the nltk corpus and shuffle them.
    #
    pos_ids = [(fileid, "pos") for fileid in movie_reviews.fileids('pos')]
    neg_ids = [(fileid, "neg") for fileid in movie_reviews.fileids('neg')]
    labeled_fileids = pos_ids + neg_ids

    # Here, we "seed" the random number generator with 0 so that we'll all
    # get the same split, which will make it easier to compare results.
    random.seed(0)   # we'll use the seed for reproduceability...
    random.shuffle(labeled_fileids)

    # Define the feature function
    #  Problem 4's central challeng is to modify this to improve your classifier's performance...
    #
    def opinion_features(fileid):
        """ starter feature engineering for movie reviews... """
        # many features are counts!

        # my most basic opinion features involve counting the positive and negative words in each review
        # this gives me a very basic sense of how positive or negative a review
        # is
        positive_count = 0
        negative_count = 0
        total_word_count = 0

        for word in movie_reviews.words(fileid):
                                                        # looks primarily at
                                                        # raw goodness/badness
                                                        # of every word
            total_word_count += 1
            if word in pos_set:
                positive_count += 1
            if word in neg_set:
                negative_count += 1

        if negative_count != 0:
            pos_neg_ratio = positive_count / negative_count
        else:
            pos_neg_ratio = positive_count / 1

        pos_neg_count = positive_count - negative_count
        pos_per_word = positive_count / total_word_count
        neg_per_word = negative_count / total_word_count

        # the other opinion features that I obtain require the use of the TextBlob library
        # I use the same approach outlined in the 2004 Stanford University paper 'Sentiment Extraction and Classification of Movie Reviews'
        # link to paper write up here: https://nlp.stanford.edu/courses/cs224n/2004/huics_kgoel_final_project.htm
        # using the approach outlined in this paper, I isolate the adjectives in every review since they are
        # the most likely parts of speech to contain information about sentiment towards a movie.
        # After isolating the adjectives, I score them by sentiment. The sum of all of the adjective polarities
        # is the score for the sentiment of the movie. To further tune my algorithm, I reverse the polarity of all
        # adjectives that come after negating words like "no" "never" "not" or "n't". Using this approach, my algorithm
        # can accurately interpret sentiment of a movie review 68% of the time on average but can acheive accuracy as high
        # as 71%.

        review_blob = textblob.TextBlob(movie_reviews.raw(fileid))   # convert text of review to TextBlob
        review_adj_score = []                                        # list where I store adjective scores for sentences
        # review_adj_count = 0
        adj_list = ['JJ', 'JJR', 'JJS']                              # these are the Parts of Speech I'm looking for (all different types of adjectives)

        # this section creates a series of simple features

        for sentence in review_blob.sentences:                       # break every review down into sentences

            # sentence_score = 0

            pre_negator = []                                         # list for words that come before a negating word
            post_negator = []                                        # list for words that come after a negating word 

            word_list = sentence.words

            for word in range(len(word_list)):                       # this for loop looks for negating words and splits the sentence at the index of the negating word
                if word_list[word].lower() == 'not' or word == 'never' or word == 'no' or word == "n't":
                    word_index = word
                    for x in word_list[0:word_index]:
                        pre_negator.append(str(x))
                    for x in word_list[word_index:]:
                        post_negator.append(str(x))

            # need to isolate ADJs to score them (before negation)
            pos_adj_list = []                                        # list to store for all adjectives preceding negating word
            pre_negator_str = ' '.join(pre_negator)
            pre_negator_sentence = textblob.Sentence(pre_negator_str)
            pre_POS_tags = pre_negator_sentence.tags
            for word in pre_POS_tags:
                if word[1] in adj_list:
                    pos_adj_list.append(word[0])

            # need to isolate ADJs to score them (after negation)
            negated_adj_list = []                                    # list to store for all adjectives that come after a negating word           
            post_negator_str = ' '.join(post_negator)
            post_negator_sentence = textblob.Sentence(post_negator_str)
            post_POS_tags = post_negator_sentence.tags
            for word in post_POS_tags:
                if word[1] in adj_list:
                    negated_adj_list.append(word[0])

            # now need to turn adj only lists into sentences that can be scored for sentiment (since lists/individual words can't be scored)
            pos_sen = ' '.join(pos_adj_list)
            pre_negator_adj_only_sentence = textblob.Sentence(pos_sen)
            pre_negator_score = pre_negator_adj_only_sentence.sentiment[0]   # score for all adjectives that come before a negating word

            post_sen = ' '.join(negated_adj_list)
            post_negator_adj_only_sentence = textblob.Sentence(
                post_sen)
            post_negator_score = pre_negator_adj_only_sentence.sentiment[    # score for all adjectives that come after a negating word
                0] * -1

            review_adj_score.append(pre_negator_score + post_negator_score)   # score for all adjectives in a sentence adjusted for negated adjectives

        total_score = sum(review_adj_score)                                   # sum the scores for all sentences and that's the score for the review

        # here is the dictionary of features...
        features = {
            'pos neg count': pos_neg_count, 'pos/word': pos_per_word, 'neg/word': neg_per_word,
            'adj score': total_score,
        }

        return features

        # Ideas
        # count both positive and negative words...
        # is the ABSOUTE count what matters?
        #
        # other ideas:
        #
        # feature ideas from the TextBlob library:
        #   * part-of-speech, average sentence length, sentiment score, subjectivity...
        # feature ideas from TextBlob or NLTK (or just Python):
        # average word length
        # number of parentheses in review
        # number of certain punctuation marks in review
        # number of words in review
        # words near or next-to positive or negative words: "not excellent" ?
        # uniqueness
        #
        # many others are possible...

        # Extract features for all of the movie reviews
        #
    print("Creating features for all reviews...", end="", flush=True)
    features = [opinion_features(fileid)
                for (fileid, opinion) in labeled_fileids]
    labels = [opinion for (fileid, opinion) in labeled_fileids]
    fileids = [fileid for (fileid, opinion) in labeled_fileids]
    print(" ... done.", flush=True)

    # Change the dictionary of features into an array
    #
    print("Transforming from dictionaries of features to vectors...",
          end="", flush=True)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)
    print(" ... done.", flush=True)

    # Split the data into train, devtest, and test

    X_test = X[:100, :]
    Y_test = labels[:100]
    fileids_test = fileids[:100]

    X_devtest = X[100:200, :]
    Y_devtest = labels[100:200]
    fileids_devtest = fileids[100:200]

    X_train = X[200:, :]
    Y_train = labels[200:]
    fileids_train = fileids[200:]

    # Train the decision tree classifier - perhaps try others or add parameters
    #
    dt = sklearn.tree.DecisionTreeClassifier()
    # dt = sklearn.ensemble.RandomForestClassifier()    # random forest has
    # significantly lower performance out of the box!
    dt.fit(X_train, Y_train)

    # Evaluate on the devtest set; report the accuracy and also
    # show the confusion matrix.
    #
    print("Score on devtest set: ", dt.score(X_devtest, Y_devtest))
    # score_list.append(dt.score(X_devtest, Y_devtest))
    # print(score_list)
    Y_guess = dt.predict(X_devtest)
    CM = confusion_matrix(Y_guess, Y_devtest)
    print("Confusion Matrix:\n", CM)

    # Get a list of errors to examine more closely.
    #

    errors = []

    for i in range(len(fileids_devtest)):
        this_fileid = fileids_devtest[i]
        this_features = X_devtest[i:i + 1, :]
        this_label = Y_devtest[i]
        guess = dt.predict(this_features)[0]
        if guess != this_label:
            errors.append((this_label, guess, this_fileid))

    # Now, print out the results: the incorrect guesses
    # Create a flag to turn this printing on/off...

    PRINT_ERRORS = False                            # this function allows for errors in movie review analyzer to bre printed
    if PRINT_ERRORS == True:
        SE = sorted(errors)
        print("There were", len(SE), "errors:")
        print('guess (actual)')
        num_to_print = 10
        for (actual, guess, name) in SE:
            if actual == 'pos' and guess == 'neg':  # adjust these as needed...
                print()
                print('----')
                print((opinion_features(name)))
                print(guess, "(", actual, ')')
                print('----')
                num_to_print -= 1
                if num_to_print == 0:
                    break
        print()

# ## Problem 4 Reflections/Analysis
#
# Include a short summary of
#   (a) how well your final set of features did!
#   (b) what other features you tried and which ones seemed to
#       help the most/least
