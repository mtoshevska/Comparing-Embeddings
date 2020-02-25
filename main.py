from preprocess import read_corpus, extract_tokens, extract_lemmas, extract_vocabulary

if __name__ == '__main__':
    s1, s2, _, = read_corpus()
    t1 = extract_tokens(s1, 'sen1')
    t2 = extract_tokens(s2, 'sen2')
    l1 = extract_lemmas(t1, 'sen1')
    l2 = extract_lemmas(t2, 'sen2')
    v_t = extract_vocabulary(t1 + t2, 'tokens')
    v_l = extract_vocabulary(l1 + l2, 'lemmas')
