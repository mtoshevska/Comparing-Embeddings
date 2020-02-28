import _pickle as pickle
from preprocess import read_wordsim, read_simlex
from sklearn.metrics.pairwise import cosine_similarity


def read_dataset(dataset_name):
    if dataset_name == 'WordSim353':
        return read_wordsim()
    else:
        return read_simlex()


def read_embeddings(dataset_name, emb_name, emb_type, emb_size):
    with open(f'data/{dataset_name}_{emb_name}_{emb_type}_{emb_size}.pkl', 'rb') as doc:
        embeddings = pickle.load(doc)
    return embeddings


def calculate_cosine_similarity(dataset_name, emb_name, emb_type, emb_size):
    cosine = list()
    dataset = read_dataset(dataset_name)
    embeddings = read_embeddings(dataset_name, emb_name, emb_type, emb_size)
    for _, row in dataset.iterrows():
        if row['word1'].lower() in embeddings.keys() and row['word2'].lower() in embeddings.keys():
            vec1 = embeddings[row['word1'].lower()]
            vec2 = embeddings[row['word2'].lower()]
            cosine.append(
                round(cosine_similarity([vec1], [vec2])[0][0] * 10, 2))
        else:
            cosine.append(None)
    dataset['cosine_sim'] = cosine
    dataset.to_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv', index=None)


if __name__ == '__main__':
    calculate_cosine_similarity('WordSim353', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('WordSim353', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('WordSim353', 'glove', 'twitter', 50)
    calculate_cosine_similarity('WordSim353', 'glove', 'twitter', 200)
    calculate_cosine_similarity('WordSim353', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('WordSim353', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('WordSim353', 'word2vec', 'google', 300)

    calculate_cosine_similarity('SimLex999', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('SimLex999', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('SimLex999', 'glove', 'twitter', 50)
    calculate_cosine_similarity('SimLex999', 'glove', 'twitter', 200)
    calculate_cosine_similarity('SimLex999', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('SimLex999', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('SimLex999', 'word2vec', 'google', 300)
