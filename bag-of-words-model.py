def main():
    import pandas as pd
    trainData = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", \
			quoting=3)
    print("Training Data Properties: shape={}, headings={}\n").format(\
        trainData.shape, trainData.columns.values)
    cleanAllReviews(trainData)
    print(createBagOfWords(trainData['review']).shape)

def createBagOfWords(cleanTrainReviews):
    """ Return a bag-of-words model for training data.

    The bag-of-words model associates a review with a features vector composed
    of 5000 integers each representing the frequency of one of the words found
    in the feature vector in a review. The feature vector is composed of the
    top 5000 highest frequency words out of the vocabulary of all reviews.

    We choose to limit the feature vector to 5000 words .

    Args:
        cleanTrainReviews (array): 1-D array, all cleaned training reviews.

    Returns:
        array: 2-D feature matrix, a row for each review, 5000 columns for
            each feature word, a cell contains an integer frequency for the
            feature word at column j in review i. Document-feature-word matrix.
    Note:
        CountVectorizer can perform data preprocessing, tokenization, and
        stop-word removal by passing callable functions. Instead, in this
        module are preprocessing functions we wrote.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    # CountVectorizer represents bag-of-words model.
    #   fit_transform() performs fitting (i.e. build vocabulary feature vector)
    #   then transforms the passed list of reviews into a
    #   review-feature-word matrix.
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,\
                    preprocessor=None, stop_words=None, max_features=5000)
    trainDataFeatures = vectorizer.fit_transform(cleanTrainReviews)
    return trainDataFeatures.toarray()

def testReviewCleaning(trainData):
    """ Print a random movie review before and after cleaning.

    Quickly test by inspection the correctness of review cleaning by printing
    a single random review before and after cleaning.

    Args:
        trainData (pandas.DataFrame): Two-dimensional table with three columns
            'id', 'sentiment', 'review' and 25000 entries.
    Note:
        Testing function output correctness requires assertions on the expected
            output. At this level, testing by inspection is acceptable, but
            using pytest/unittest is proper.

        This function is correct after inspection.
    """
    import random
    i = random.randint(1, len(trainData.index))
    print(trainData.get_value(i, 'review')+'\n')
    print(cleanReview(trainData.get_value(i, 'review')))

def cleanAllReviews(trainData):
    """ Clean all the movie reviews.

    Each movie review is cleaned and replaces the unclean one in the training
    data. Display progress of review cleaning.

    Args:
        trainData (pandas.DataFrame): Two-dimensional table with three columns
            'id', 'sentiment', 'review' and 25000 entries.
    """
    import sys
    for i in range(len(trainData.index)):
            sys.stdout.flush()
            trainData.set_value(i, 'review', \
                cleanReview(trainData.get_value(i, 'review')))
            print("\rReview of {} of {} cleaned.").format(i,\
                len(trainData.index)-1),

def cleanReview(rawReview):
    """ Return a single cleaned review.

    A cleaned review is one without HTML tags, contains only lower-case
    alphabetical characters, and only has important descriptive words remaining.

    Args:
        rawReview (str): Uncleaned review to be cleaned.

    Returns:
        str: Review cleaned in accordance with requirements above.
    """
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    import re
    return " ".join([w for w in
        re.sub("[^a-zA-Z]", " ", BeautifulSoup(rawReview,\
        'html.parser').get_text()).lower().split()\
        if not w in stopwords.words("english")])

if __name__ == "__main__":
    main()
