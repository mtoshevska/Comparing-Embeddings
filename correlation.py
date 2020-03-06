import pandas as pd
import numpy as np
from math import isnan
from scipy.stats import pearsonr, kendalltau, spearmanr


def calculate_correlation(dataset_name, emb_name, emb_type, emb_size):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    s1 = dataset['gt_sim'].get_values()
    s2 = dataset['cosine_sim'].get_values()
    for i in range(len(s2) - 1, 0, -1):
        if isnan(s2[i]):
            s1 = np.delete(s1, i)
            s2 = np.delete(s2, i)
    r_p, pvalue_p = pearsonr(s1, s2)
    correlation_k, pvalue_k = kendalltau(s1, s2)
    rho_s, pvalue_s = spearmanr(s1, s2)
    results = pd.DataFrame()
    results['correlation_type'] = ['Pearson', 'Kendall’s tau', 'Spearman']
    results['correlation_coefficient'] = [r_p, correlation_k, rho_s]
    results['p_value'] = [pvalue_p, pvalue_k, pvalue_s]
    results.to_csv(f'results/correlation/{dataset_name}_{emb_name}_{emb_type}_{emb_size}.csv', index=None)


def calculate_correlation_pos(dataset_name, emb_name, emb_type, emb_size):
    dataset = pd.read_csv(f'results/similarity/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_cosine.csv')
    for pos_tag in ['A', 'N', 'V']:
        data = dataset.loc[dataset['POS'] == pos_tag]
        s1 = data['gt_sim'].get_values()
        s2 = data['cosine_sim'].get_values()
        for i in range(len(s2) - 1, 0, -1):
            if isnan(s2[i]):
                s1 = np.delete(s1, i)
                s2 = np.delete(s2, i)
        r_p, pvalue_p = pearsonr(s1, s2)
        correlation_k, pvalue_k = kendalltau(s1, s2)
        rho_s, pvalue_s = spearmanr(s1, s2)
        results = pd.DataFrame()
        results['correlation_type'] = ['Pearson', 'Kendall’s tau', 'Spearman']
        results['correlation_coefficient'] = [r_p, correlation_k, rho_s]
        results['p_value'] = [pvalue_p, pvalue_k, pvalue_s]
        results.to_csv(f'results/correlation/{dataset_name}_{emb_name}_{emb_type}_{emb_size}_{pos_tag}.csv', index=None)


if __name__ == '__main__':
    calculate_correlation('WordSim353', 'glove', 'wikipedia', 50)
    calculate_correlation('WordSim353', 'glove', 'wikipedia', 300)
    calculate_correlation('WordSim353', 'glove', 'twitter', 50)
    calculate_correlation('WordSim353', 'glove', 'twitter', 200)
    calculate_correlation('WordSim353', 'fasttext', 'wikipedia', 300)
    calculate_correlation('WordSim353', 'fasttext', 'crawl', 300)
    calculate_correlation('WordSim353', 'lexvec', 'wikipedia', 300)
    calculate_correlation('WordSim353', 'lexvec', 'crawl', 300)
    calculate_correlation('WordSim353', 'word2vec', 'googlenews', 300)
    calculate_correlation('WordSim353', 'numberbatch', 'conceptnet', 300)

    calculate_correlation('SimLex999', 'glove', 'wikipedia', 50)
    calculate_correlation('SimLex999', 'glove', 'wikipedia', 300)
    calculate_correlation('SimLex999', 'glove', 'twitter', 50)
    calculate_correlation('SimLex999', 'glove', 'twitter', 200)
    calculate_correlation('SimLex999', 'fasttext', 'wikipedia', 300)
    calculate_correlation('SimLex999', 'fasttext', 'crawl', 300)
    calculate_correlation('SimLex999', 'lexvec', 'wikipedia', 300)
    calculate_correlation('SimLex999', 'lexvec', 'crawl', 300)
    calculate_correlation('SimLex999', 'word2vec', 'googlenews', 300)
    calculate_correlation('SimLex999', 'numberbatch', 'conceptnet', 300)

    calculate_correlation('SimVerb3500', 'glove', 'wikipedia', 50)
    calculate_correlation('SimVerb3500', 'glove', 'wikipedia', 300)
    calculate_correlation('SimVerb3500', 'glove', 'twitter', 50)
    calculate_correlation('SimVerb3500', 'glove', 'twitter', 200)
    calculate_correlation('SimVerb3500', 'fasttext', 'wikipedia', 300)
    calculate_correlation('SimVerb3500', 'fasttext', 'crawl', 300)
    calculate_correlation('SimVerb3500', 'lexvec', 'wikipedia', 300)
    calculate_correlation('SimVerb3500', 'lexvec', 'crawl', 300)
    calculate_correlation('SimVerb3500', 'word2vec', 'googlenews', 300)
    calculate_correlation('SimVerb3500', 'numberbatch', 'conceptnet', 300)

    calculate_correlation_pos('SimLex999', 'glove', 'wikipedia', 50)
    calculate_correlation_pos('SimLex999', 'glove', 'wikipedia', 300)
    calculate_correlation_pos('SimLex999', 'glove', 'twitter', 50)
    calculate_correlation_pos('SimLex999', 'glove', 'twitter', 200)
    calculate_correlation_pos('SimLex999', 'fasttext', 'wikipedia', 300)
    calculate_correlation_pos('SimLex999', 'fasttext', 'crawl', 300)
    calculate_correlation_pos('SimLex999', 'lexvec', 'wikipedia', 300)
    calculate_correlation_pos('SimLex999', 'lexvec', 'crawl', 300)
    calculate_correlation_pos('SimLex999', 'word2vec', 'googlenews', 300)
    calculate_correlation_pos('SimLex999', 'numberbatch', 'conceptnet', 300)
