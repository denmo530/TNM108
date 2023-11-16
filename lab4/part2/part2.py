import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
from sklearn.feature_extraction.text import TfidfTransformer

moviedir = r'./movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

len(movie.data)

# target names ("classes") are automatically generated from subfolder names
movie.target_names

# First file seems to be about a Schwarzenegger movie. 
movie.data[0][:500]

# first file is in "neg" folder
movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie.target[0]

# A DEATOUR 

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, 
                                                          test_size = 0.20, random_state = 12)
# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text 
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
movieVzer.vocabulary_.get('screen')

# Likewise, Mr. Steven Seagal is present...
movieVzer.vocabulary_.get('seagal')

# huge dimensions! 1,600 documents, 3K unique terms. 
docs_train_counts.shape

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
docs_train_tfidf.shape

#
# Next up; TEST DATA
#

docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

#
# Training and testing a Naive Bayes classifier 
#

# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
sklearn.metrics.accuracy_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#
# Trying the classifier on fake movie reviews  
#

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

    # Mr. Seagal simply cannot win!  

#
# GRID SEARCH
#

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

text_clf.fit(movie.data, movie.target)

import numpy as np
docs_test = movie.data
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == movie.target))

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42 ,max_iter=5, tol=None)),
])

text_clf.fit(movie.data, movie.target)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ", np.mean(predicted == movie.target))


from sklearn import metrics
print(metrics.classification_report(movie.target, predicted,
 target_names=movie.target_names))

print(metrics.confusion_matrix(movie.target, predicted))

from sklearn.model_selection import GridSearchCV
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2), (1,3),(2,3)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3, 1e-4),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(movie.data, movie.target)

print(movie.target_names[gs_clf.predict(['Best movie ever'])[0]])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

