import pandas as pd
from math import isnan
import numpy as np


def average_word_similarity(dataset_name, emb_name, emb_type, emb_size):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    s1 = dataset['gt_sim'].get_values()
    s2 = dataset['cosine_sim'].get_values()
    for i in range(len(s2) - 1, 0, -1):
        if isnan(s2[i]):
            s1 = np.delete(s1, i)
            s2 = np.delete(s2, i)
    print('==================================================')
    print(f'Dataset: {dataset_name}, Embeddings: {emb_name}-{emb_type}-{emb_size}')
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


if __name__ == '__main__':
    average_word_similarity('WordSim353', 'glove', 'wikipedia', 50)
    average_word_similarity('WordSim353', 'glove', 'wikipedia', 300)
    average_word_similarity('WordSim353', 'glove', 'twitter', 50)
    average_word_similarity('WordSim353', 'glove', 'twitter', 200)
    average_word_similarity('WordSim353', 'fasttext', 'wikipedia', 300)
    average_word_similarity('WordSim353', 'fasttext', 'crawl', 300)
    average_word_similarity('WordSim353', 'word2vec', 'freebase', 1000)
    average_word_similarity('WordSim353', 'word2vec', 'google_news', 300)

    average_word_similarity('SimLex999', 'glove', 'wikipedia', 50)
    average_word_similarity('SimLex999', 'glove', 'wikipedia', 300)
    average_word_similarity('SimLex999', 'glove', 'twitter', 50)
    average_word_similarity('SimLex999', 'glove', 'twitter', 200)
    average_word_similarity('SimLex999', 'fasttext', 'wikipedia', 300)
    average_word_similarity('SimLex999', 'fasttext', 'crawl', 300)
    average_word_similarity('SimLex999', 'word2vec', 'freebase', 1000)
    average_word_similarity('SimLex999', 'word2vec', 'google_news', 300)

    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 50)
    average_word_similarity('SimVerb3500', 'glove', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 50)
    average_word_similarity('SimVerb3500', 'glove', 'twitter', 200)
    average_word_similarity('SimVerb3500', 'fasttext', 'wikipedia', 300)
    average_word_similarity('SimVerb3500', 'fasttext', 'crawl', 300)
    average_word_similarity('SimVerb3500', 'word2vec', 'freebase', 1000)
    average_word_similarity('SimVerb3500', 'word2vec', 'google_news', 300)

    average_word_similarity_pos('SimLex999', 'glove', 'wikipedia', 50)
    average_word_similarity_pos('SimLex999', 'glove', 'wikipedia', 300)
    average_word_similarity_pos('SimLex999', 'glove', 'twitter', 50)
    average_word_similarity_pos('SimLex999', 'glove', 'twitter', 200)
    average_word_similarity_pos('SimLex999', 'fasttext', 'wikipedia', 300)
    average_word_similarity_pos('SimLex999', 'fasttext', 'crawl', 300)
    average_word_similarity_pos('SimLex999', 'word2vec', 'freebase', 1000)
    average_word_similarity_pos('SimLex999', 'word2vec', 'google_news', 300)
