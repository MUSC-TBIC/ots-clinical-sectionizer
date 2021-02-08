import argparse
import os
import glob
import re

import spacy as spacy
import cassis

import pandas as pd

import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

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


def extract_features( input_dir , training_flag = False ):
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
            doc_len = len(processed_notes) # number of tokens
            span_end = 0
            end_idx = 0

            for sent_i, sent in enumerate(sentences):
                # count by tokens
                len_bytokens = sent.__len__()
                span_end += len_bytokens
                span_start = span_end - len_bytokens
                sen_percentile = span_start / doc_len  # FEATURE:  position of the sentence

                ## FEATURE: capitalization type. all-upper case, title case, or others.
                ## FEATURE: containing a verb
                all_up, all_tit, contain_verb = False, False, False
                count_tokens, count_nonletter, count_up, count_tit, count_verb = 0, 0, 0, 0, 0
                for token in sent:
                    count_tokens += 1
                    if re.search('[a-z]|[A-Z]', token.text) and not re.search('[0-9]', token.text):
                        if token.text.isupper():
                            count_up += 1
                        if token.text.istitle():
                            count_tit += 1
                        if token.pos_ == "VERB":
                            count_verb += 1
                    else:
                        count_nonletter += 1
                        continue #check the next token if the token contains a number

                if count_tokens == count_nonletter:
                    all_up, all_tit, contain_verb = False, False, False
                else:
                    if count_up == count_tokens - count_nonletter:
                        all_up = True
                    if count_tit == count_tokens - count_nonletter:
                        all_tit = True
                    if count_verb > 0:
                        contain_verb = True

                ## FEATURE: end with a colon
                ending_symbols = sent.text[-2:]
                if re.search("\:",ending_symbols):
                    ending_colon = True
                else:
                    ending_colon = False

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

    # adjust the datatype
    for column in features:
        if column not in ["percentile",'filename','idx1','idx2','content']:
            features[column] = features[column].astype(bool)
        if column in ['idx1', 'idx2']:
            features[column] = features[column].astype(int)

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
        features_clean = raw_features.drop( [ 'idx1', 'idx2','content', 'wrongly_sentencized'], axis=1).dropna(axis=0)
    else:
        features_clean = raw_features.drop( ['filename', 'idx1', 'idx2','content', 'wrongly_sentencized'], axis=1).dropna(axis=0)
    return( features_clean )


def write_xmi(raw_features, y_pred, output_dir):
    df = raw_features
    df["isHeader"] = y_pred

    typesystem = cassis.TypeSystem()
    SentAnnotation = typesystem.create_type( name = 'TBD.SentenceAnnotation')
    typesystem.add_feature( type_ = SentAnnotation,
                            name = 'text',
                            rangeTypeName= 'uima.cas.String')
    typesystem.add_feature( type_ = SentAnnotation ,
                            name = 'isHeader' ,
                            description= 'If the sentence is a header or not',
                            rangeTypeName = 'uima.cas.Boolean' )

    ## Iterate over the files, covert to CAS, and write the XMI to disk
    filenames = df["filename"].unique()
    for filename in filenames:
        df_byfilename = raw_features[raw_features["filename"]==filename]

        cas = cassis.Cas(typesystem=typesystem)
        cas.sofa_mime = "text/plain"
        s = ""
        cas.sofa_string = s.join(df_byfilename["content"].tolist())
        for i in range(len(df_byfilename)):
            cas.add_annotation(SentAnnotation(begin=df_byfilename["idx1"].iloc[i],
                                           end=df_byfilename["idx2"].iloc[i],
                                           text=df_byfilename["content"].iloc[i],
                                           isHeader=df_byfilename['isHeader'].iloc[i]))
        # write CAS XMI
        xmi_filename = re.sub('.txt$', '.xmi', filename)
        cas.to_xmi(path=os.path.join(output_dir, xmi_filename), pretty_print=True)


def train( input_dir, output_dir , model_dir, svm_kernel):
    raw_features , target = extract_features( input_dir, training_flag = True )
    predictors = clean_up_features( raw_features )

    # write the pd.df as a .csv
    raw_features['isHeader'] = target
    raw_features.to_csv( os.path.join( output_dir , "raw_processed_dataset.csv" ) ,
                         index = True )
    
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=444)
    
    # create a classifier
    if svm_kernel == 'linear':
        model = svm.SVC(kernel='linear')
        model.fit(x_train, y_train)
        print("Linear kernel")
    elif svm_kernel == 'rbf':
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}
        model = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
        model.fit(x_train, y_train)
        print("Best parameters for RBF SVM:")
        print(model.best_params_)


    ## Save the model to disk
    model_file = os.path.join( model_dir , 'svm_sectionizer.pkl' )
    pickle.dump( model , open( model_file , 'wb' ) )
    
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("Training completed.")



def annotate( input_dir, output_dir , model_file ):
    raw_features = extract_features( input_dir, training_flag = False )
    predictors = clean_up_features( raw_features , keep_filename = False )
    
    ## Load the model from disk
    model = pickle.load( open( model_file , 'rb' ) )
    
    y_pred = model.predict( predictors )

    # write features and predicted values as a .csv
    cleaned_features = clean_up_features( raw_features , keep_filename = True )
    assert len(cleaned_features) == len(y_pred)
    cleaned_features["pred_isHeader"] = y_pred
    cleaned_features.to_csv( os.path.join( output_dir , "cleaned_processed_dataset.csv" ) ,
                             index = False )

    ## Write a CAS XMI file per filename
    assert len(raw_features) == len(y_pred)
    write_xmi(raw_features, y_pred, output_dir)
    print("Annotation completed.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clincial notes sectionizer based on spaCY basic pipeline and SVM')
    parser.add_argument( '-i' , '--input-dir' , required = True ,
                         dest = 'input_dir' ,
                         help='Input directory containing plain text (and .ann) files to train or to sectionize')
    parser.add_argument( '-o' , '--output-dir' , required = True ,
                         dest = 'output_dir',
                        help='Output directory for writing the processed dataset or CAS XMI files to')
    parser.add_argument( '--model-dir' , required = False ,
                         default = None ,
                         dest = 'model_dir',
                         help='Output directory for writing the model details' )
    parser.add_argument( '--train' ,
                        dest = 'train' ,
                        help = "Train a new model based on the input directory" ,
                        action = "store_true" )
    parser.add_argument('--svm-kernel',
                         default = 'rbf',
                         dest = 'svm_kernel',
                         help = 'Specify the SVM kernel function. For example, "linear" or "rbf" (radial basis function)')
    args = parser.parse_args()
    if( not os.path.exists( args.output_dir ) ):
        os.mkdir( args.output_dir )
    if( args.model_dir is None ):
        args.model_dir = args.output_dir
    if( args.train ):
        train( os.path.abspath( args.input_dir ) ,
               os.path.abspath( args.output_dir ) ,
               os.path.abspath( args.model_dir ),
               args.svm_kernel)
    else:
        annotate( os.path.abspath( args.input_dir ) ,
                  os.path.abspath( args.output_dir ) ,
                  os.path.join( os.path.abspath( args.model_dir ) ,
                                'svm_sectionizer.pkl' ) )
