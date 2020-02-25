from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import wordnet


def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''


def pos_tagging(tokens):
    tags = pos_tag(tokens)
    return tags


def lemmatize(tokens):
    tags = pos_tagging(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = list()
    for token, tag in zip(tokens, tags):
        mapped_tag = wordnet_pos_code(tag[1])
        if mapped_tag != '':
            lemma = lemmatizer.lemmatize(token, pos=mapped_tag)
        else:
            lemma = lemmatizer.lemmatize(token)
        lemmas.append(lemma)
    return lemmas
