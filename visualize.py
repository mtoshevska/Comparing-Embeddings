import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import read_dataset


def plot_correlation(dataset_name, emb_name, emb_type, emb_size, kind, save_file=False):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    sns.set(style='darkgrid', context='poster', font='Verdana')
    sns.jointplot('gt_sim', 'cosine_sim', data=dataset, kind=kind, color='mediumseagreen')

    if save_file:
        figure = plt.gcf()
        figure.set_size_inches(9, 9)
        plt.suptitle(f'Dataset: {dataset_name}\nEmbeddings: {emb_name}-{emb_type}-{emb_size}')
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.1)
        plt.savefig(f'results/img/{kind}/{dataset_name}_{emb_name}_{emb_type}_{emb_size}.png')
    else:
        plt.show()


def plot_similarity(dataset_name, embeddings):
    dataset = read_dataset(dataset_name)
    data = pd.DataFrame()
    pairs = list(range(len(dataset['gt_sim'])))
    embedding_names = ['gt_similarity'] * len(dataset['gt_sim'])
    similarities = list(dataset['gt_sim'].get_values())
    for embedding in embeddings:
        dataset = pd.read_csv(f'results/similarity/{dataset_name}_{embedding}_cosine.csv')
        pairs += list(range(len(dataset['cosine_sim'])))
        embedding_names += [embedding] * len(dataset['cosine_sim'])
        similarities += list(dataset['cosine_sim'].get_values())
    data['pairs'] = pairs
    data['embeddings'] = embedding_names
    data['similarities'] = similarities
    sns.set(style='darkgrid', context='poster', font='Verdana', font_scale=0.5)
    sns.lineplot(x='pairs', y='similarities', hue='embeddings', style='embeddings', dashes=False, data=data)
    plt.show()


if __name__ == '__main__':
    plot_correlation('WordSim353', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('WordSim353', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('WordSim353', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('WordSim353', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('WordSim353', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('WordSim353', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('WordSim353', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('WordSim353', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('WordSim353', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('WordSim353', 'numberbatch', 'conceptnet', 300, 'hex', True)

    plot_correlation('SimLex999', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('SimLex999', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimLex999', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('SimLex999', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('SimLex999', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimLex999', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('SimLex999', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimLex999', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('SimLex999', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('SimLex999', 'numberbatch', 'conceptnet', 300, 'hex', True)

    plot_correlation('SimVerb3500', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('SimVerb3500', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('SimVerb3500', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('SimVerb3500', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('SimVerb3500', 'numberbatch', 'conceptnet', 300, 'hex', True)

    plot_correlation('WordSim353', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('WordSim353', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('WordSim353', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('WordSim353', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('WordSim353', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('WordSim353', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('WordSim353', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('WordSim353', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('WordSim353', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('WordSim353', 'numberbatch', 'conceptnet', 300, 'reg', True)

    plot_correlation('SimLex999', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('SimLex999', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimLex999', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('SimLex999', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('SimLex999', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimLex999', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('SimLex999', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimLex999', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('SimLex999', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('SimLex999', 'numberbatch', 'conceptnet', 300, 'reg', True)

    plot_correlation('SimVerb3500', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('SimVerb3500', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('SimVerb3500', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('SimVerb3500', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('SimVerb3500', 'numberbatch', 'conceptnet', 300, 'reg', True)
