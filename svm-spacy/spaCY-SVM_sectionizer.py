import argparse
import os
import glob
import re

import spacy as spacy

import pandas as pd

import pickle

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tqdm import tqdm

def extract_header_index( ann_file ):
    header_idx = []
    with open( ann_file , 'r') as fp:
        header_ann = fp.read().splitlines()
        # get the index of headers
        for ann_line in header_ann:
            if re.search("^T[0-9]", ann_line):
                splitted_annline = ann_line.split('\t')
                header_idx.append(splitted_annline[1].split(" ")[-2:])
    return( header_idx )


def check_header( index , header_idx ):
    in_range = False
    i = 0
    while not in_range and i < len( header_idx ):
        idx_range = header_idx[i]
        if index in range(int(idx_range[0]), int(idx_range[1]) + 1): # inclusive range
            in_range = True
        i += 1
    return in_range


def extract_features( input_dir, output_dir , training_flag = False ):
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
        header_idx = None
        if( training_flag ):
            # read .ann files for creating the training data
            filename_ann = re.sub("txt$", "ann", filename)
            header_idx = extract_header_index( os.path.join( input_dir , filename_ann ) )

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

                if( header_idx is None ):
                    is_header, wrongly_sentencized = False , False
                elif( check_header( start_idx , header_idx ) and
                    check_header( end_idx-1 , header_idx ) ):
                    is_header, wrongly_sentencized = True, False
                elif( not check_header( start_idx , header_idx ) and
                      not check_header( end_idx-1 , header_idx ) ):
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

    # SVM
    raw_features = features.drop( 'header' , axis=1 )
    if( training_flag ):
        target = features[ 'header' ]
        return( raw_features , target )
    else:
        return( raw_features )


def clean_up_features( raw_features , keep_filename = False ):
    # convert to binary predictors. No need to normalize the continous variable 'percentile'
    if( keep_filename ):
        features_clean = raw_features.drop( [ 'idx1', 'idx2','content'], axis=1).dropna(axis=0)
    else:
        features_clean = raw_features.drop( ['filename', 'idx1', 'idx2','content'], axis=1).dropna(axis=0)
    return( features_clean )


def train( input_dir, output_dir , model_dir ):
    raw_features , target = extract_features( input_dir, output_dir , training_flag = True )

    # write the pd.df as a .csv
    raw_features.to_csv( os.path.join( output_dir , "raw_processed_dataset.csv" ) ,
                         index = True )
    
    predictors = clean_up_features( raw_features )
    
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=444)
    
    # create a classifier
    ## TODO - make the kernal a passable argument so you can provide
    ##        something like --svm-kernel "linear" / --svm-kernel "rbf"
    ##        as a command line argument
    clf = svm.SVC(kernel='linear')
    # train the model using the training sets
    clf.fit(x_train, y_train)
    ## Save the model to disk
    model_file = os.path.join( model_dir , 'svm_sectionizer.pkl' )
    pickle.dump( clf , open( model_file , 'wb' ) )
    
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


def annotate( input_dir, output_dir , model_file ):
    raw_features = extract_features( input_dir, output_dir , training_flag = False )

    cleaned_features = clean_up_features( raw_features , keep_filename = True )
    cleaned_features.to_csv( os.path.join( output_dir , "cleaned_processed_dataset.csv" ) ,
                             index = False )
    predictors = clean_up_features( raw_features , keep_filename = False )
    
    ## Load the model from disk
    clf = pickle.load( open( model_file , 'rb' ) )
    
    y_pred = clf.predict( predictors )

    ## TODO - Iterate over the raw_features to write a CAS XMI file per filename


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clincial notes sectionizer based on spaCY basic pipeline and SVM')
    parser.add_argument( '-i' , '--input-dir' , required = True ,
                         dest = 'input_dir' ,
                         help='Input directory containing plain text and .ann files to train and test the sectionizer')
    parser.add_argument( '-o' , '--output-dir' , required = True ,
                         dest = 'output_dir',
                        help='Output directory for writing the processed dataset to')
    parser.add_argument( '--model-dir' , required = False ,
                         default = None ,
                         dest = 'model_dir',
                         help='Output directory for writing the model details' )
    parser.add_argument( '--train' ,
                         dest = 'train' ,
                         help = "Train a new model based on the input directory" ,
                         action = "store_true" )
    args = parser.parse_args()
    if( not os.path.exists( args.output_dir ) ):
        os.mkdir( args.output_dir )
    if( args.model_dir is None ):
        args.model_dir = args.output_dir
    if( args.train ):
        train( os.path.abspath( args.input_dir ) ,
               os.path.abspath( args.output_dir ) ,
               os.path.abspath( args.model_dir ) )
    else:
        annotate( os.path.abspath( args.input_dir ) ,
                  os.path.abspath( args.output_dir ) ,
                  os.path.join( os.path.abspath( args.model_dir ) ,
                                'svm_sectionizer.pkl' ) )
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
