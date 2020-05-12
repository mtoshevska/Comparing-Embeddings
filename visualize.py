import pandas as pd
import numpy as np
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import read_dataset


def plot_correlation(dataset_name, emb_name, emb_type, emb_size, kind, save_file=False):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    sns.set(style='darkgrid', context='poster', font='Verdana', font_scale=2.5)
    sns.jointplot('gt_sim', 'cosine_sim', data=dataset, kind=kind, color='mediumseagreen')
    if save_file:
        figure = plt.gcf()
        figure.set_size_inches(18, 18)
        if emb_name == 'numberbatch':
            plt.suptitle(f'{emb_type}-{emb_name}-{emb_size}')
        else:
            plt.suptitle(f'{emb_name}-{emb_type}-{emb_size}')
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


def combine_images_correlation(dataset_name, kind, grid_size):
    paths = [path for path in os.listdir(f'results/img/{kind}') if path.startswith(dataset_name)]
    images = [Image.open(f'results/img/{kind}/{image}') for image in paths]
    if grid_size == '2x5':
        result = np.vstack((np.hstack([np.asarray(images[i]) for i in range(int(len(images) / 2))]),
                            np.hstack([np.asarray(images[i]) for i in range(int(len(images) / 2), len(images))])))
    elif grid_size == '4x3':
        np_images = [np.asarray(images[i]) for i in range(2)]
        np_images.append(np.zeros((len(np_images[0]), len(np_images[0][0]), len(np_images[0][0][0])), dtype=np.uint8))
        np_images.extend([np.asarray(images[i]) for i in range(2, 10)])
        np_images.append(np.zeros((len(np_images[0]), len(np_images[0][0]), len(np_images[0][0][0])), dtype=np.uint8))

        result = np.vstack((np.hstack([np_images[i] for i in range(3)]),
                            np.hstack([np_images[i] for i in range(3, 6)]),
                            np.hstack([np_images[i] for i in range(6, 9)]),
                            np.hstack([np_images[i] for i in range(9, 12)])))
    else:
        raise Exception(f'No such grid size supported: {grid_size}')
    result_image = Image.fromarray(result)
    result_image.save(f'results/img/{dataset_name}_{kind}_{grid_size}.png')


def plot_avg_similarities(dataset_name, save_file=False):
    paths = [path for path in os.listdir('results/similarity') if path.startswith(dataset_name)]
    values = [np.mean(read_dataset(dataset_name)['gt_sim'].get_values())]
    embeddings = ['GT']
    for path in paths:
        values.append(np.nanmean(pd.read_csv(f'results/similarity/{path}')['cosine_sim'].get_values()))
        emb_name = path.split('_')
        if emb_name[1] == 'numberbatch':
            embeddings.append(f'{emb_name[2][0].upper()}-{emb_name[1][0].upper()}-{emb_name[3]}')
        else:
            embeddings.append(f'{emb_name[1][0].upper()}-{emb_name[2][0].upper()}-{emb_name[3]}')
    data = pd.DataFrame()
    data['embeddings'] = embeddings
    data['similarities'] = values
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.barplot(x='embeddings', y='similarities', ax=ax, data=data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
    ax.axhline(0, color='k', clip_on=False)
    plt.ylim(0, 10)
    for bar, value in zip(ax.patches, data['similarities'].get_values()):
        text_x = bar.get_x() + bar.get_width() / 2.0
        text_y = bar.get_height() + 0.025
        text = f'{round(value, 5)}'
        ax.text(text_x, text_y, text, fontsize=20, ha='center', va='bottom', rotation=90, color='k')
    sns.despine(bottom=True)
    plt.title(dataset_name)
    if save_file:
        figure = plt.gcf()
        figure.set_size_inches(10, 8)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.3)
        plt.savefig(f'results/img/{dataset_name}_avg_sim.png')
    else:
        plt.show()


def combine_images_similarity():
    paths = [path for path in os.listdir(f'results/img') if '_avg_sim' in path]
    images = [Image.open(f'results/img/{image}') for image in paths]
    result = np.hstack((np.vstack([np.asarray(i) for i in images[:3]]),
                        np.vstack([np.asarray(i) for i in images[3:]])))
    result_image = Image.fromarray(result)
    result_image.save('results/img/avg_sim.png')


if __name__ == '__main__':
    plot_avg_similarities('WordSim353', True)
    plot_avg_similarities('SimLex999', True)
    plot_avg_similarities('SimVerb3500', True)
    plot_avg_similarities('RG65', True)
    plot_avg_similarities('RW2034', True)
    plot_avg_similarities('Verb143', True)

    combine_images_similarity()

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

    plot_correlation('RG65', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('RG65', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('RG65', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('RG65', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('RG65', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('RG65', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('RG65', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('RG65', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('RG65', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('RG65', 'numberbatch', 'conceptnet', 300, 'hex', True)

    plot_correlation('RW2034', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('RW2034', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('RW2034', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('RW2034', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('RW2034', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('RW2034', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('RW2034', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('RW2034', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('RW2034', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('RW2034', 'numberbatch', 'conceptnet', 300, 'hex', True)

    plot_correlation('Verb143', 'glove', 'wikipedia', 50, 'hex', True)
    plot_correlation('Verb143', 'glove', 'wikipedia', 300, 'hex', True)
    plot_correlation('Verb143', 'glove', 'twitter', 50, 'hex', True)
    plot_correlation('Verb143', 'glove', 'twitter', 200, 'hex', True)
    plot_correlation('Verb143', 'fasttext', 'wikipedia', 300, 'hex', True)
    plot_correlation('Verb143', 'fasttext', 'crawl', 300, 'hex', True)
    plot_correlation('Verb143', 'lexvec', 'wikipedia', 300, 'hex', True)
    plot_correlation('Verb143', 'lexvec', 'crawl', 300, 'hex', True)
    plot_correlation('Verb143', 'word2vec', 'googlenews', 300, 'hex', True)
    plot_correlation('Verb143', 'numberbatch', 'conceptnet', 300, 'hex', True)

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

    plot_correlation('RG65', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('RG65', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('RG65', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('RG65', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('RG65', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('RG65', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('RG65', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('RG65', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('RG65', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('RG65', 'numberbatch', 'conceptnet', 300, 'reg', True)

    plot_correlation('RW2034', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('RW2034', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('RW2034', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('RW2034', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('RW2034', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('RW2034', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('RW2034', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('RW2034', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('RW2034', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('RW2034', 'numberbatch', 'conceptnet', 300, 'reg', True)

    plot_correlation('Verb143', 'glove', 'wikipedia', 50, 'reg', True)
    plot_correlation('Verb143', 'glove', 'wikipedia', 300, 'reg', True)
    plot_correlation('Verb143', 'glove', 'twitter', 50, 'reg', True)
    plot_correlation('Verb143', 'glove', 'twitter', 200, 'reg', True)
    plot_correlation('Verb143', 'fasttext', 'wikipedia', 300, 'reg', True)
    plot_correlation('Verb143', 'fasttext', 'crawl', 300, 'reg', True)
    plot_correlation('Verb143', 'lexvec', 'wikipedia', 300, 'reg', True)
    plot_correlation('Verb143', 'lexvec', 'crawl', 300, 'reg', True)
    plot_correlation('Verb143', 'word2vec', 'googlenews', 300, 'reg', True)
    plot_correlation('Verb143', 'numberbatch', 'conceptnet', 300, 'reg', True)

    combine_images_correlation('WordSim353', 'hex', '2x5')
    combine_images_correlation('SimLex999', 'hex', '2x5')
    combine_images_correlation('SimVerb3500', 'hex', '2x5')
    combine_images_correlation('RG65', 'hex', '2x5')
    combine_images_correlation('RW2034', 'hex', '2x5')
    combine_images_correlation('Verb143', 'hex', '2x5')
    combine_images_correlation('WordSim353', 'reg', '2x5')
    combine_images_correlation('SimLex999', 'reg', '2x5')
    combine_images_correlation('SimVerb3500', 'reg', '2x5')
    combine_images_correlation('RG65', 'reg', '2x5')
    combine_images_correlation('RW2034', 'reg', '2x5')
    combine_images_correlation('Verb143', 'reg', '2x5')

    combine_images_correlation('WordSim353', 'hex', '4x3')
    combine_images_correlation('SimLex999', 'hex', '4x3')
    combine_images_correlation('SimVerb3500', 'hex', '4x3')
    combine_images_correlation('RG65', 'hex', '4x3')
    combine_images_correlation('RW2034', 'hex', '4x3')
    combine_images_correlation('Verb143', 'hex', '4x3')
    combine_images_correlation('WordSim353', 'reg', '4x3')
    combine_images_correlation('SimLex999', 'reg', '4x3')
    combine_images_correlation('SimVerb3500', 'reg', '4x3')
    combine_images_correlation('RG65', 'reg', '4x3')
    combine_images_correlation('RW2034', 'reg', '4x3')
    combine_images_correlation('Verb143', 'reg', '4x3')
