import argparse
import glob
import os
import re

from tqdm import tqdm

import spacy
from clinical_sectionizer import Sectionizer

import cassis

def main( inputDir , outputDir ):
    ## Load an clinical English medspaCy model trained on i2b2 data
    ## - https://github.com/medspacy/sectionizer/blob/master/notebooks/00-clinical_sectionizer.ipynb
    nlp_pipeline = spacy.load( 'en_info_3700_i2b2_2012' )
    ## - https://github.com/medspacy/sectionizer/tree/master/resources
    ##   - spacy_section_patterns.jsonl
    ##   - text_section_patterns.json
    ##   - patrict_section_patterns.json
    sectionizer = Sectionizer( nlp_pipeline )
    nlp_pipeline.add_pipe( sectionizer )
    ############################
    ## Create a type system
    ## - https://github.com/dkpro/dkpro-cassis/blob/master/cassis/typesystem.py
    ############
    ## ... for tokens
    typesystem = cassis.TypeSystem()
    TokenAnnotation = typesystem.create_type( name = 'uima.tt.TokenAnnotation' , 
                                              supertypeName = 'uima.tcas.Annotation' )
    typesystem.add_feature( type_ = TokenAnnotation ,
                            name = 'text' , 
                            rangeTypeName = 'uima.cas.String' )
    ############
    ## ... for sections (and, by association, section headers)
    NoteSection = typesystem.create_type( name = 'edu.musc.tbic.uima.NoteSection' ,
                                          supertypeName = 'uima.tcas.Annotation' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'SectionNumber' ,
                            description = '' ,
                            rangeTypeName = 'uima.cas.Integer' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'SectionDepth' ,
                            description = 'Given an hierarchical section schema, how deep is the current section ( 0 = root level/major category)' , 
                            rangeTypeName = 'uima.cas.Integer' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'SectionId' ,
                            description = 'Type (or concept id) of current section' , 
                            rangeTypeName = 'uima.cas.String' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'beginHeader' ,
                            description = 'The start offset for this section\'s header (-1 if no header)' , 
                            rangeTypeName = 'uima.cas.Integer' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'endHeader' ,
                            description = 'The end offset for this section\'s header (-1 if no header)' , 
                            rangeTypeName = 'uima.cas.Integer' )
    typesystem.add_feature( type_ = NoteSection ,
                            name = 'modifiers' ,
                            description = 'Modifiers (key/value pairs) associated with the given section' , 
                            rangeTypeName = 'uima.cas.String' )
    ############################
    ## Iterate over the files, covert to CAS, and write the XMI to disk
    filenames = [ os.path.basename( f ) for f in glob.glob( os.path.join( inputDir ,
                                                                          '*.txt' ) ) ]
    for filename in tqdm( filenames ):
        xmi_filename = re.sub( '.txt$' ,
                               '.xmi' ,
                               filename )
        with open( os.path.join( inputDir , filename ) , 'r' ) as fp:
            note_contents = fp.read()
        cas = cassis.Cas( typesystem = typesystem )
        cas.sofa_string = note_contents
        cas.sofa_mime = "text/plain"
        sectionized_note = nlp_pipeline( note_contents )
        ########################
        ## Tokens
        ## - https://spacy.io/api/token
        for token in sectionized_note:
            cas.add_annotation( TokenAnnotation( begin = token.idx , 
                                                 end = token.idx + token.__len__() ,
                                                 text = token.text ) )
        ########################
        ## Sections
        ## - 
        start_header = 0
        start_span = 0
        end_header = 0
        end_span = 0
        for (title, header, span) in sectionized_note._.sections:
            end_header += len( str( header ) )
            end_span += len( str( span ) )
            span_start = str( start_span )
            span_end = str( end_span )
            if( header is None ):
                ## If the header is None, then we want to set
                ## the title to empty for the span (meaning
                ## it's most definitely a span that occurred
                ## prior to the first header)
                span_title = ''
                header_start = '-1'
                header_end = '-1'
            else:
                header_start = str( start_header )
                header_end = str( end_header )
                header_title = title
                span_title = title
            cas.add_annotation( NoteSection( beginHeader = header_start ,
                                             endHeader = header_end ,
                                             ##title = header_title ,
                                             begin = span_start ,
                                             end = span_end ) )
            start_header += end_span
            start_span = start_header
            end_header = start_header
            end_span = start_span
        cas.to_xmi( path = os.path.join( outputDir , xmi_filename ) ,
                    pretty_print = True )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description = 'Simple medspaCy pipeline for splitting a plain text note into sections and writing them out in CAS XMI format' )
    parser.add_argument( 'inputDir' ,
                         help = 'Input directory containing plain text files to sectionize' )
    parser.add_argument( 'outputDir' ,
                         help = 'Output directory for writing CAS XMI files to' )
    args = parser.parse_args()
    main( os.path.abspath( args.inputDir ) ,
          os.path.abspath( args.outputDir ) )
