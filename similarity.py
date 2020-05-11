from preprocess import read_dataset, read_embeddings
from sklearn.metrics.pairwise import cosine_similarity


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
    calculate_cosine_similarity('WordSim353', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('WordSim353', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('WordSim353', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('WordSim353', 'numberbatch', 'conceptnet', 300)

    calculate_cosine_similarity('SimLex999', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('SimLex999', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('SimLex999', 'glove', 'twitter', 50)
    calculate_cosine_similarity('SimLex999', 'glove', 'twitter', 200)
    calculate_cosine_similarity('SimLex999', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('SimLex999', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('SimLex999', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('SimLex999', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('SimLex999', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('SimLex999', 'numberbatch', 'conceptnet', 300)

    calculate_cosine_similarity('SimVerb3500', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('SimVerb3500', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('SimVerb3500', 'glove', 'twitter', 50)
    calculate_cosine_similarity('SimVerb3500', 'glove', 'twitter', 200)
    calculate_cosine_similarity('SimVerb3500', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('SimVerb3500', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('SimVerb3500', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('SimVerb3500', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('SimVerb3500', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('SimVerb3500', 'numberbatch', 'conceptnet', 300)

    calculate_cosine_similarity('RG65', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('RG65', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('RG65', 'glove', 'twitter', 50)
    calculate_cosine_similarity('RG65', 'glove', 'twitter', 200)
    calculate_cosine_similarity('RG65', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('RG65', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('RG65', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('RG65', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('RG65', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('RG65', 'numberbatch', 'conceptnet', 300)

    calculate_cosine_similarity('RW2034', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('RW2034', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('RW2034', 'glove', 'twitter', 50)
    calculate_cosine_similarity('RW2034', 'glove', 'twitter', 200)
    calculate_cosine_similarity('RW2034', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('RW2034', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('RW2034', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('RW2034', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('RW2034', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('RW2034', 'numberbatch', 'conceptnet', 300)

    calculate_cosine_similarity('Verb143', 'glove', 'wikipedia', 50)
    calculate_cosine_similarity('Verb143', 'glove', 'wikipedia', 300)
    calculate_cosine_similarity('Verb143', 'glove', 'twitter', 50)
    calculate_cosine_similarity('Verb143', 'glove', 'twitter', 200)
    calculate_cosine_similarity('Verb143', 'fasttext', 'wikipedia', 300)
    calculate_cosine_similarity('Verb143', 'fasttext', 'crawl', 300)
    calculate_cosine_similarity('Verb143', 'lexvec', 'wikipedia', 300)
    calculate_cosine_similarity('Verb143', 'lexvec', 'crawl', 300)
    calculate_cosine_similarity('Verb143', 'word2vec', 'googlenews', 300)
    calculate_cosine_similarity('Verb143', 'numberbatch', 'conceptnet', 300)
