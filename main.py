from preprocess import read_corpus, extract_tokens, extract_lemmas


if __name__ == '__main__':
    s1, s2, _, = read_corpus()
    t1 = extract_tokens(s1, 'sen1')
    t2 = extract_tokens(s2, 'sen2')
    l1 = extract_lemmas(t1, 'sen1')
    l2 = extract_lemmas(t2, 'sen2')
