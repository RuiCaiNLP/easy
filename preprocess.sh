#!/usr/bin/env bash

python preprocess-conll09.py \
--train CoNLL2009-ST-English-train.txt \
--test CoNLL2009-ST-evaluation-English.txt \
--dev CoNLL2009-ST-English-development.txt \
--train_fr fr-up-train.conllu.txt \
--test_fr fr-up-test.conllu.txt \
--dev_fr fr-up-dev.conllu.txt