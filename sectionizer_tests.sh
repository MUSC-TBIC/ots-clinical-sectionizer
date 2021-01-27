#!/bin/zsh

if [[ -z $ETUDE_DIR ]]; then
    echo "The variable \$ETUDE_DIR is not set"
    exit 0
fi

if [[ -z $ETUDE_CONFIGS ]]; then
    echo "The variable \$ETUDE_CONFIGS is not set"
    exit 0
fi

if [[ -n $BIGODM_DIR ]]; then
    echo "Sectionizing BigODM Corpus (train)"
    mkdir -p data/output/medspaCy_clinical-sectionizer_big-odm_train
    python3 medspacy/medspaCy_sectionizer.py \
	    ${BIGODM_DIR}/train \
	    data/output/medspaCy_clinical-sectionizer_big-odm_train
    python3 ${ETUDE_DIR}/etude.py \
	    --reference-input ${BIGODM_DIR}/train \
	    --reference-config ${ETUDE_CONFIGS}/brat/sections_big-odm_brat.conf \
	    --test-input data/output/medspaCy_clinical-sectionizer_big-odm_train \
	    --test-config ${ETUDE_CONFIGS}/medspacy/sections_cas-xmi.conf \
	    --file-suffix ".ann" ".xmi" \
	    --score-key Parent \
	    --score-value Header \
	    --fuzzy-match-flags partial \
	    -m TP FP FN Precision Recall F1 \
	    --delim "|" --delim-prefix "|"
    ##
    echo "Sectionizing BigODM Corpus (test)"
    mkdir -p data/output/medspaCy_clinical-sectionizer_big-odm_test
    python3 medspacy/medspaCy_sectionizer.py \
	    ${BIGODM_DIR}/test \
	    data/output/medspaCy_clinical-sectionizer_big-odm_test
    python3 ${ETUDE_DIR}/etude.py \
	    --reference-input ${BIGODM_DIR}/train \
	    --reference-config ${ETUDE_CONFIGS}/brat/sections_big-odm_brat.conf \
	    --test-input data/output/medspaCy_clinical-sectionizer_big-odm_test \
	    --test-config ${ETUDE_CONFIGS}/medspacy/sections_cas-xmi.conf \
	    --file-suffix ".ann" ".xmi" \
	    --score-key Parent \
	    --score-value Header \
	    --fuzzy-match-flags partial \
	    -m TP FP FN Precision Recall F1 \
	    --delim "|" --delim-prefix "|"
else
    echo "Skipping BigODM Corpus"
fi

if [[ -n $TEXTRACTOR_DIR ]]; then
    echo "Sectionizing Textractor Corpus"
    mkdir -p data/output/medspaCy_clinical-sectionizer_textractor
    python3 medspacy/medspaCy_sectionizer.py \
	    "${TEXTRACTOR_DIR}/brat2" \
	    data/output/medspaCy_clinical-sectionizer_textractor
    python3 ${ETUDE_DIR}/etude.py \
	    --reference-input "${TEXTRACTOR_DIR}/brat2" \
	    --reference-config ${ETUDE_CONFIGS}/brat/sections_ontology_brat.conf \
	    --test-input data/output/medspaCy_clinical-sectionizer_textractor \
	    --test-config ${ETUDE_CONFIGS}/medspacy/sections_cas-xmi.conf \
	    --file-suffix ".ann" ".xmi" \
	    --score-key Parent \
	    --score-value Header \
	    --fuzzy-match-flags partial \
	    -m TP FP FN Precision Recall F1 \
	    --delim "|" --delim-prefix "|"
else
    echo "Skipping Textractor Corpus"
fi

