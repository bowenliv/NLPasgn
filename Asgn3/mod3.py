import os
import sys
import numpy as np
import json
import csv, re
import string
from tqdm import tqdm
import codecs
import argparse
from collections import Counter
import spacy
from assgn2_fns import *
import spacy
import pandas as pd
from sklearn.cluster import KMeans

from traceback_with_variables import activate_by_import

# Convenient for debugging but feel free to comment out
input_speechfile = "./speeches2020_jan_to_jun.jsonl.gz"
# Hard-wired variables
def cluster_words(nlp, words, numclusters):
    # Convert words into spacy tokens and from there into vectors
    # Only include tokens that actually have a vector in spacy
    tokens = [nlp(word) for word in words]
    vectors = [token.vector for token in tokens if token.has_vector]
    wordlist = [token.text for token in tokens if token.has_vector]

    # Do k-means clustering on the set of vectors
    # Result of clustering is item_labels, a numpy array with one entry per item that was clustered
    # That is, item_labels[0] is the cluster label for the first vector,
    #          item_labels[1] is the cluster label for the next vector, etc.
    sys.stderr.write("Running KMeans...\n")
    df_vectors = pd.DataFrame(vectors)
    km = KMeans(n_clusters=numclusters, init='k-means++', random_state=10, n_init=1)
    km.fit(df_vectors)
    item_labels = km.predict(df_vectors)

    # Convert k-means result into a dictionary (clusterdict) mapping from
    # a cluster label to list of the words in that cluster
    clusterdict = {}
    item_number = 0
    for item_label in item_labels:
        if item_label in clusterdict:
            clusterdict[item_label].append(wordlist[item_number])
        else:
            clusterdict[item_label] = [wordlist[item_number]]
        item_number += 1

    return clusterdict


# Input: dictionary mapping items to numbers
# Prints n items, either the top n or the bottom n
# depending on the order.
def print_sorted_items(dict, n=10, order='descending'):
    keys = []
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n]:
        print(key,value)
        keys.append(key)
    return keys


def main(chamber, party, verb, maxlines, num_to_show, num_to_cluster):
    # Some initializations
    lines_processed = 0
    counts = Counter()

    # Read in the speeches that we want to analyze
    print("Reading {}".format(input_speechfile))
    lines, parties = read_and_clean_lines(input_speechfile, chamber)

    # Make sure maxlines is no longer than the corpus itself
    if maxlines > len(lines):
        maxlines = len(lines)

    # Initialize spacy
    print("Initializing spaCy")
    nlp = spacy.load('en_core_web_sm')

    # Main loop
    # If you're not familiar with this use of 'for x,y in zip(list1,list2)'
    # see https://stackoverflow.com/questions/21098350/python-iterate-over-two-lists-simultaneously
    print("Iterating through {} documents".format(maxlines))
    for line, line_party in tqdm(zip(lines, parties), total=maxlines):

        # Stop after maxlines lines have been processed
        if lines_processed >= maxlines:
            break

        # Skip this speech if it's not the party we're interested in
        if line_party != party:
            continue

        # Do the NLP analysis using spacy
        # There are many tutorials out there for things you can do with spacy;
        # the one at https://www.machinelearningplus.com/spacy-tutorial-nlp/ is particularly good.
        analysis = nlp(line)

        # For each token, see if it's the object of the verb we're interested in
        # and if so, increment its count (normalizing using its lowercased lemma)
        for token in analysis:
            if token.head.lemma_ == verb and token.dep_ == 'dobj':
                counts[token.lemma_.lower()] += 1
                if False:  # Set to True for debugging to see info spacy gives you for dependencies
                    print("{3}/{4} --{2}--> {0}/{1}".format(
                        token.text,  # Text of this token
                        token.tag_,  # POS tag for this token
                        token.dep_,  # Label for dependency link
                        token.head.text,  # Head that this token modifies
                        token.head.tag_))  # POS tag for the head this token modifies

        lines_processed += 1

    print("For {}s in the {}, clustering top direct objects for the verb '{}'".format(party, chamber, verb))
    clusterdict = cluster_words(nlp, print_sorted_items(counts, num_to_show, 'descending'), num_to_cluster)
    print("Clustering result")
    print(clusterdict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Find most frequent objects for a specified verb in political speeches')
    parser.add_argument('--chamber', default='Senate', action='store', help="Senate or House")
    parser.add_argument('--party', default='Republican', action='store', help="Democrat or Republican")
    parser.add_argument('--verb', default='eat', action='store', help="verb of interest")
    parser.add_argument('--maxlines', default=100000, action='store', help="maximum number of speeches to analyze")
    parser.add_argument('--num_to_show', default=20, action='store', help="top-n items to show")
    parser.add_argument('--num_to_cluster', default=2, action='store', help="number of clusters, default 2")
    args = parser.parse_args()
    main(args.chamber, args.party, args.verb, int(args.maxlines), int(args.num_to_show), int(args.num_to_cluster))
