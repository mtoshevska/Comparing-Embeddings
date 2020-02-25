import numpy as np
import os
import _pickle as pickle
from preprocess import extract_wordsim_vocabulary, extract_simlex_vocabulary


def load_glove_embeddings(embeddings_type, embeddings_size, words, dataset_name):
    if os.path.exists(f'data/{dataset_name}_{embeddings_type}_{embeddings_size}.pkl'):
        with open(f'data/{dataset_name}_{embeddings_type}_{embeddings_size}.pkl', 'rb') as doc:
            embeddings = pickle.load(doc)
    else:
        embeddings = dict()
        emb_type = '6B' if embeddings_type == 'wikipedia' else 'twitter.27B'
        file_name = f'data/glove.{emb_type}.{embeddings_size}d.txt'
        with open(file_name, 'r', encoding='utf-8') as doc:
            line = doc.readline()
            while line != '':
                line = line.rstrip('\n').lower()
                parts = line.split(' ')
                vals = np.array(parts[1:], dtype=np.float)
                if parts[0] in words:
                    embeddings[parts[0]] = vals
                line = doc.readline()
        with open(f'data/{dataset_name}_{embeddings_type}_{embeddings_size}.pkl', 'wb') as doc:
            pickle.dump(embeddings, doc)
    return embeddings


if __name__ == '__main__':
    wordsim_vocab = extract_wordsim_vocabulary()
    simlex_vocab = extract_simlex_vocabulary()

    wordsim_wikipedia_50 = load_glove_embeddings('wikipedia', 50, wordsim_vocab, 'WordSim353')
    wordsim_wikipedia_300 = load_glove_embeddings('wikipedia', 300, wordsim_vocab, 'WordSim353')
    wordsim_twitter_50 = load_glove_embeddings('twitter', 50, wordsim_vocab, 'WordSim353')
    wordsim_twitter_200 = load_glove_embeddings('twitter', 200, wordsim_vocab, 'WordSim353')

    simlex_wikipedia_50 = load_glove_embeddings('wikipedia', 50, simlex_vocab, 'SimLex999')
    simlex_wikipedia_300 = load_glove_embeddings('wikipedia', 300, simlex_vocab, 'SimLex999')
    simlex_twitter_50 = load_glove_embeddings('twitter', 50, simlex_vocab, 'SimLex999')
    simlex_twitter_200 = load_glove_embeddings('twitter', 200, simlex_vocab, 'SimLex999')
