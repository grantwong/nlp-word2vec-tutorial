def main():
    import pandas as pd
    trainData = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", \
			quoting=3)
    print("Training Data Properties: shape={}, headings={}\n").format(\
        trainData.shape, trainData.columns.values)
    #testReviewCleaning(trainData)
    cleanAllReviews(trainData)

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
