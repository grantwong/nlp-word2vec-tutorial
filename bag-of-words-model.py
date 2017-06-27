def main():
    import pandas as pd
    trainData = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", \
			quoting=3)
    print("Training Data Properties: shape={}, headings={}\n").format(\
        trainData.shape, trainData.columns.values)
    #testReviewCleaning(trainData)
    cleanAllReviews(trainData)

def testReviewCleaning(trainData):
    """ Print a random movie review before and after cleaning. """
    import random
    i = random.randint(1, len(trainData.index))
    print(trainData.get_value(i, 'review')+'\n')
    print(cleanReview(trainData.get_value(i, 'review')))

def cleanAllReviews(trainData):
    """ Replace each raw review in DataFrame trainData with its cleaned
    version. """
    import sys
    for i in range(len(trainData.index)):
            sys.stdout.flush()
            trainData.set_value(i, 'review', \
            cleanReview(trainData.get_value(i, 'review')))
            print("\rReview of {} of {} cleaned.").format(i,\
                len(trainData.index)-1),

def cleanReview(rawReview):
    """ Return cleaned movie review text for processing by removing HTML,
    remove punctuation and lower case text, and remove stop-words. """
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    import re
    return " ".join([w for w in
                re.sub("[^a-zA-Z]", " ", BeautifulSoup(rawReview,\
                'html.parser').get_text()).lower().split()\
            if not w in stopwords.words("english")])

if __name__ == "__main__":
    main()
