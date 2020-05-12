import pandas as pd
from math import isnan
import numpy as np
from itertools import repeat


def average_word_similarity(dataset_name, emb_name, emb_type, emb_size):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    s1 = dataset['gt_sim'].get_values()
    s2 = dataset['cosine_sim'].get_values()
    nan_values = 0
    for i in range(len(s2) - 1, 0, -1):
        if isnan(s2[i]):
            nan_values += 1
            s1 = np.delete(s1, i)
            s2 = np.delete(s2, i)
    print('==================================================')
    print(f'Dataset: {dataset_name}, Embeddings: {emb_name}-{emb_type}-{emb_size}')
    print(f'NaN: {nan_values}')
    print(f'Average gt similarity: {round(np.mean(s1), 2)}')
    print(f'Average cosine similarity: {round(np.mean(s2), 2)}')
    print('==================================================')
    print()


def average_word_similarity_pos(dataset_name, emb_name, emb_type, emb_size):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    for pos_tag in ['A', 'N', 'V']:
        data = dataset.loc[dataset['POS'] == pos_tag]
        s1 = data['gt_sim'].get_values()
        s2 = data['cosine_sim'].get_values()
        for i in range(len(s2) - 1, 0, -1):
            if isnan(s2[i]):
                s1 = np.delete(s1, i)
                s2 = np.delete(s2, i)
        print('==================================================')
        print(f'Dataset: {dataset_name}, POS: {pos_tag}, Embeddings: {emb_name}-{emb_type}-{emb_size}')
        print(f'Average gt similarity: {round(np.mean(s1), 2)}')
        print(f'Average cosine similarity: {round(np.mean(s2), 2)}')
        print('==================================================')
        print()


def print_average_word_similarities():
    average_word_similarity('WordSim353', 'glove', 'wikipedia', 50)
    average_word_similarity('WordSim353', 'glove', 'wikipedia', 300)
    average_word_similarity('WordSim353', 'glove', 'twitter', 50)
    average_word_similarity('WordSim353', 'glove', 'twitter', 200)
    average_word_similarity('WordSim353', 'fasttext', 'wikipedia', 300)
    average_word_similarity('WordSim353', 'fasttext', 'crawl', 300)
    average_word_similarity('WordSim353', 'lexvec', 'wikipedia', 300)
    average_word_similarity('WordSim353', 'lexvec', 'crawl', 300)
    average_word_similarity('WordSim353', 'word2vec', 'googlenews', 300)
    average_word_similarity('WordSim353', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('SimLex999', 'glove', 'wikipedia', 50)
    average_word_similarity('SimLex999', 'glove', 'wikipedia', 300)
    average_word_similarity('SimLex999', 'glove', 'twitter', 50)
    average_word_similarity('SimLex999', 'glove', 'twitter', 200)
    average_word_similarity('SimLex999', 'fasttext', 'wikipedia', 300)
    average_word_similarity('SimLex999', 'fasttext', 'crawl', 300)
    average_word_similarity('SimLex999', 'lexvec', 'wikipedia', 300)
    average_word_similarity('SimLex999', 'lexvec', 'crawl', 300)
    average_word_similarity('SimLex999', 'word2vec', 'googlenews', 300)
    average_word_similarity('SimLex999', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 50)
    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 50)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 200)
    average_word_similarity('SimVerb3500', 'fasttext', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'fasttext', 'crawl', 300)
    average_word_similarity('SimVerb3500', 'lexvec', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'lexvec', 'crawl', 300)
    average_word_similarity('SimVerb3500', 'word2vec', 'googlenews', 300)
    average_word_similarity('SimVerb3500', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 50)
    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 50)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 200)
    average_word_similarity('SimVerb3500', 'fasttext', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'fasttext', 'crawl', 300)
    average_word_similarity('SimVerb3500', 'lexvec', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'lexvec', 'crawl', 300)
    average_word_similarity('SimVerb3500', 'word2vec', 'googlenews', 300)
    average_word_similarity('SimVerb3500', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('RG65', 'glove', 'wikipedia', 50)
    average_word_similarity('RG65', 'glove', 'wikipedia', 300)
    average_word_similarity('RG65', 'glove', 'twitter', 50)
    average_word_similarity('RG65', 'glove', 'twitter', 200)
    average_word_similarity('RG65', 'fasttext', 'wikipedia', 300)
    average_word_similarity('RG65', 'fasttext', 'crawl', 300)
    average_word_similarity('RG65', 'lexvec', 'wikipedia', 300)
    average_word_similarity('RG65', 'lexvec', 'crawl', 300)
    average_word_similarity('RG65', 'word2vec', 'googlenews', 300)
    average_word_similarity('RG65', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('RW2034', 'glove', 'wikipedia', 50)
    average_word_similarity('RW2034', 'glove', 'wikipedia', 300)
    average_word_similarity('RW2034', 'glove', 'twitter', 50)
    average_word_similarity('RW2034', 'glove', 'twitter', 200)
    average_word_similarity('RW2034', 'fasttext', 'wikipedia', 300)
    average_word_similarity('RW2034', 'fasttext', 'crawl', 300)
    average_word_similarity('RW2034', 'lexvec', 'wikipedia', 300)
    average_word_similarity('RW2034', 'lexvec', 'crawl', 300)
    average_word_similarity('RW2034', 'word2vec', 'googlenews', 300)
    average_word_similarity('RW2034', 'numberbatch', 'conceptnet', 300)

    average_word_similarity('Verb143', 'glove', 'wikipedia', 50)
    average_word_similarity('Verb143', 'glove', 'wikipedia', 300)
    average_word_similarity('Verb143', 'glove', 'twitter', 50)
    average_word_similarity('Verb143', 'glove', 'twitter', 200)
    average_word_similarity('Verb143', 'fasttext', 'wikipedia', 300)
    average_word_similarity('Verb143', 'fasttext', 'crawl', 300)
    average_word_similarity('Verb143', 'lexvec', 'wikipedia', 300)
    average_word_similarity('Verb143', 'lexvec', 'crawl', 300)
    average_word_similarity('Verb143', 'word2vec', 'googlenews', 300)
    average_word_similarity('Verb143', 'numberbatch', 'conceptnet', 300)

    average_word_similarity_pos('SimLex999', 'glove', 'wikipedia', 50)
    average_word_similarity_pos('SimLex999', 'glove', 'wikipedia', 300)
    average_word_similarity_pos('SimLex999', 'glove', 'twitter', 50)
    average_word_similarity_pos('SimLex999', 'glove', 'twitter', 200)
    average_word_similarity_pos('SimLex999', 'fasttext', 'wikipedia', 300)
    average_word_similarity_pos('SimLex999', 'fasttext', 'crawl', 300)
    average_word_similarity_pos('SimLex999', 'lexvec', 'wikipedia', 300)
    average_word_similarity_pos('SimLex999', 'lexvec', 'crawl', 300)
    average_word_similarity_pos('SimLex999', 'word2vec', 'googlenews', 300)
    average_word_similarity_pos('SimLex999', 'numberbatch', 'conceptnet', 300)


def construct_correlation_table(datasets, embeddings):
    data = pd.DataFrame()
    embeddings_list = ['-'.join([str(i) for i in e]) for e in embeddings]
    data['Word Embeddings'] = [x for item in embeddings_list for x in repeat(item, 3)]
    data['Correlation'] = ['S', 'Pr', 'K'] * len(embeddings_list)
    for dataset_name in datasets:
        values = []
        for emb_name, emb_type, emb_size in embeddings:
            p, k, s = \
                pd.read_table(f'results/correlation/{dataset_name}_{emb_name}_{emb_type}_{emb_size}.csv', sep=',')[
                    'correlation_coefficient'].get_values()
            values.extend([s, p, k])
        data[dataset_name] = values
    data.to_csv('results/correlation_coefficients.csv', index=None)


if __name__ == '__main__':
    print_average_word_similarities()

    construct_correlation_table(['WordSim353', 'SimLex999', 'SimVerb3500', 'RG65', 'RW2034', 'Verb143'],
                                [('glove', 'wikipedia', 50), ('glove', 'wikipedia', 300),
                                 ('glove', 'twitter', 50), ('glove', 'twitter', 200),
                                 ('fasttext', 'wikipedia', 300), ('fasttext', 'crawl', 300),
                                 ('lexvec', 'wikipedia', 300), ('lexvec', 'crawl', 300),
                                 ('word2vec', 'googlenews', 300), ('numberbatch', 'conceptnet', 300)])
