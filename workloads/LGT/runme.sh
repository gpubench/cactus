#!/bin/bash

source ../../scripts/common

pip3 list | grep spacy
if [ $? -eq 1 ]; then
    if [ ! -f spaCy-3.0.5.tar.gz ]; then
        wget $BASE_URL/spaCy-3.0.5.tar.gz
        if [ $? -eq 1 ]; then
            echo "[CACTUS] Problem spaCy-3.0.5.tar.gz"
            exit 1
        fi
    fi
    pip3 install spaCy-3.0.5.tar.gz
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing spaCy-3.0.5"
        exit 1
    fi
else
    echo "[CACTUS] Found spacy"
fi

pip3 list | grep de-core-news-sm
if [ $? -eq 1 ]; then
    if [ ! -f de_core_news_sm-3.0.0.tar.gz ]; then
        wget $BASE_URL/de_core_news_sm-3.0.0.tar.gz
        if [ $? -eq 1 ]; then
            echo "[CACTUS] Problem fetching de_core_news_sm"
            exit 1
        fi
    fi
    pip3 install de_core_news_sm-3.0.0.tar.gz
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing de_core_news_sm"
        exit 1
    fi
else
    echo "[CACTUS] Found spacy de-core-news-sm"
fi

pip3 list | grep en-core-web-sm
if [ $? -eq 1 ]; then
    if [ ! -f en_core_web_sm-3.0.0.tar.gz ]; then
        wget $BASE_URL/en_core_web_sm-3.0.0.tar.gz
        if [ $? -eq 1 ]; then
            echo "[CACTUS] Problem fetching en_core_web_sm"
            exit 1
        fi
    fi
    pip3 install en_core_web_sm-3.0.0.tar.gz
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing en_core_web_sm"
        exit 1
    fi
else
    echo "[CACTUS] Found spacy en-core-web-sm"
fi

# Alternative commands to fetch newest versions
# python3 -m spacy download de_core_news_sm en_core_web_sm

python3 main.py

# More information https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
