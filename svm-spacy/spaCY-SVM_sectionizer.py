import argparse
import os
import glob
import re

import spacy as spacy

import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def main(input_dir, output_dir):
    ## Load an clinical English medspaCy model trained on i2b2 data
    ## - https://github.com/medspacy/sectionizer/blob/master/notebooks/00-clinical_sectionizer.ipynb
    nlp = spacy.load('en_info_3700_i2b2_2012')

    # Prepare the training data
    features = pd.DataFrame(
        {'uppercase': [], 'title_case': [], 'ending_colon': [], 'contain_verb': [], 'percentile': [],
         'wrongly_sentencized' :[], 'header': [], 'filename': [], 'idx1': [], 'idx2': [], 'content':[]})
    ## Iterate over the files
    filenames = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir,
                                                                     '*.txt'))]  # Filenames of data to be trained on. Data must be in .txt format AND the header info must have the same filename with the extension ".ann"
    for filename in tqdm(filenames):
        # read .ann files for creating the training data
        filename_ann = re.sub("txt$", "ann", filename)
        with open(os.path.join(input_dir, filename_ann), 'r') as fp:
            header_ann = fp.read().splitlines()
            # get the index of headers
            header_idx = []
            for ann_line in header_ann:
                if re.search("^T[0-9]", ann_line):
                    splitted_annline = ann_line.split('\t')
                    header_idx.append(splitted_annline[1].split(" ")[-2:])

        # read .txt files to get features
        with open(os.path.join(input_dir, filename), 'r') as fp:
            note_contents = fp.read()

            # split sentences
            processed_notes = nlp(note_contents)
            sentences = list(processed_notes.sents)

            # get features
            doc_len = len(
                processed_notes)
            span_end = 0
            end_idx = 0
            sent_count_i = 0
            for sent_i, sent in enumerate(sentences):
                # count by tokens
                len_bytokens = sent.__len__()
                span_end += len_bytokens
                span_start = span_end - len_bytokens
                sen_percentile = span_start / doc_len  # FEATURE:  position of the sentence

                ## FEATURE: capitalization type. all-upper case, title case, or others.
                ## FEATURE: containing a verb
                token_i = 0
                all_up, all_tit = True, True
                contain_verb = False
                while (all_up or all_tit) and not contain_verb and token_i < len_bytokens:
                    token = sent[token_i]
                    token_i += 1
                    if re.search('[a-z]|[A-Z]', token.text) and not re.search('[0-9]', token.text): # if the token contains letters and no numbers
                        if not token.text.isupper():
                            all_up = False
                        if not token.text.istitle():
                            all_tit = False
                        if token.pos_ == "VERB":
                            contain_verb = True
                    else:
                        continue

                ## FEATURE: end with a colon
                ending_symbols = sent.text[-2:]
                if re.search("\:",ending_symbols):
                    ending_colon = True
                else:
                    ending_colon = False

                ## FEATURE: probably wrongly sentencized
                # check if is a header
                sent_len = len(sent.text_with_ws)
                end_idx += sent_len # end_idx = index of the ending character + 1
                start_idx = end_idx - sent_len # index of the starting character
                assert sent.text_with_ws == processed_notes.text[start_idx:end_idx] # assert the index is correct

                def check_header(index):
                    in_range = False
                    i = 0
                    while not in_range and i < len(header_idx):
                        idx_range = header_idx[i]
                        if index in range(int(idx_range[0]), int(idx_range[1]) + 1): # inclusive range
                            in_range = True
                        i += 1
                    return in_range

                if check_header(start_idx) & check_header(end_idx-1):
                    is_header, wrongly_sentencized = True, False
                elif not check_header(start_idx) and not check_header(end_idx-1):
                    is_header, wrongly_sentencized = False, False
                else: # if part of the sentence is a header, NOT consider it as a header and mark it as wrongly_sentencized
                    is_header, wrongly_sentencized = False, True


                # features:
                feature = pd.DataFrame({'uppercase': [all_up], 'title_case': [all_tit], 'ending_colon': [ending_colon],
                                        'contain_verb': [contain_verb], 'percentile': [sen_percentile],
                                        'wrongly_sentencized': [wrongly_sentencized], 'header': [is_header],
                                        'filename': [filename], 'idx1': [start_idx], 'idx2': [end_idx-1], 'content': [sent.text_with_ws]})
                features = features.append(feature)
                ## exact header match

    # pre-process the dataframe
    for column in features:
        if column not in ["percentile",'filename','idx1','idx2','content']:
            #    features_binary = normalize(features_binary['percentile'], axis=0)
            # else:
            features[column] = features[column].astype(bool)

    # write the pd.df as a .csv
    features.to_csv(os.path.join(output_dir,"Processed_dataset.csv"),index=True)

    # convert to binary predictors. No need to normalize the continous variable 'percentile'
    features_clean = features.drop(['filename', 'idx1', 'idx2','content'], axis=1).dropna(axis=0)

    # SVM
    target = features_clean['header']
    predictors = features_clean.drop('header', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=444)

    # create a classifier
    clf = svm.SVC(kernel='linear')
    # train the model using the training sets
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print("Accuracy = ", accuracy,
          "\nF1 = ", f1,
          "\nPrecision = ", precision,
          "\nRecall = ", recall)

        # features: to create features, what spacy features can I use? (sentence boundary detection, pos, similarity ?, rule based matching?, text classification?)
        # what features to use as predictors?


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clincial notes sectionizer based on spaCY basic pipeline and SVM')
    parser.add_argument('inputDir',
                        help='Input directory containing plain text and .ann files to train and test the sectionizer')
    parser.add_argument('outputDir',
                        help='Output directory for writing the processed dataset to')
    args = parser.parse_args()
    main(os.path.abspath(args.inputDir),
         os.path.abspath(args.outputDir))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
