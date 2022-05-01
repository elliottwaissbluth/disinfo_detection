
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def tfidf_transformation(df, col):
    # First, bag of words
    bow_converter = CountVectorizer(ngram_range=(1,1),
                                    min_df=15,
                                    max_df=0.9)
    corpus = df[col].tolist()
    delisted_corpus = []
    for words in corpus:
        delisted_corpus.append(' '.join(words))

    X = bow_converter.fit_transform(delisted_corpus)
    
    # Then, tfidf
    tfidf_transform = text.TfidfTransformer(norm=None)
    X_tfidf = tfidf_transform.fit_transform(X)
    tf_vocab = {v: k for k, v in bow_converter.vocabulary_.items()}

    return X_tfidf, tf_vocab, X

def train_logistic_regression(df, col, meta_name=False):
    '''
    model, tf_X_train, tf_X_test, tf_y_train, tf_y_test, X, tf_vocab = train_logistic_regression(df, 'tokenized')
    tf_y_pred = model.predict(tf_X_test)
    '''
    if not meta_name:
        X_tfidf, tf_vocab, X = tfidf_transformation(df, col)
    else:
        X_tfidf = np.array(df[col].tolist())
        tf_vocab = None

    y = df['y'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y)
    
    # Model optimized for tf-idf combined
    model = LogisticRegression(C=0.0006951927961775605, solver='liblinear').fit(X_train, y_train)

    if not meta_name:
        return model, X_train, X_test, y_train, y_test, X.toarray(), tf_vocab
    else:
        return model, X_train, X_test, y_train, y_test, X_tfidf, tf_vocab

def evaluate_logistic_regression(model, tf_X_test, tf_y_test, tf_vocab, PRINT=True):
    
    tf_y_pred = model.predict(tf_X_test)
    
    f1 = f1_score(tf_y_test, tf_y_pred)
    precision = precision_score(tf_y_test, tf_y_pred)
    recall = recall_score(tf_y_test, tf_y_pred)
    accuracy = accuracy_score(tf_y_test, tf_y_pred)
    
    baseline = np.sum(tf_y_test) / tf_y_test.shape[0]
    if baseline < 0.5:
        baseline = 1 - baseline

    if PRINT:
        print('\n\nlogistic regression classifier\n-------------\naccuracy: {:.4} %\nbaseline: {:.4} %'.format(accuracy*100, np.max(baseline)*100))
        print('\nf1:         {:.4f}\nprecision:  {:.4f}\nrecall:     {:.4f}\n\n'.format(f1, precision, recall))

        print('size of vocab: {}\n'.format(len(tf_vocab)))

        fake_idx = model.coef_.argsort()[0][-20:][::-1]
        real_idx = model.coef_.argsort()[0][:20][::-1]

        real_words = []
        fake_words = []

        for i in range(len(real_idx)):
            real_words.append(tf_vocab[real_idx[i]])
            fake_words.append(tf_vocab[fake_idx[i]])

        top_words = pd.DataFrame(list(zip(real_words, fake_words)), columns=['top "real" words', 'top "fake" words'])
        print(top_words)

    return accuracy, f1, precision, recall

def confidence_interval_logistic_regression(num_iterations, df, col='tokenized', meta_name=False):
    '''
    Trains logistic regression classifiers on unmodified TF-IDF vectors from corpus
    '''
    results = np.zeros(shape=(4, num_iterations))
    for j in tqdm(range(num_iterations)):
        model, _, X_test, _, y_test, _, vocab = train_logistic_regression(df, col, meta_name=meta_name)
        accuracy, f1, precision, recall = evaluate_logistic_regression(model, X_test, y_test, vocab, PRINT=False)
        results[0, j] = accuracy
        results[1, j] = f1
        results[2, j] = precision
        results[3, j] = recall

    results_df = populate_results_dataframe(results)
    print(results_df)
    boxplot_results(results)

    return results

def boxplot_results(results):
    acc = results[0,:]
    f1 = results[1,:]
    precision = results[2,:]
    recall = results[3,:]

    data = [acc, f1, precision, recall]
    fig = plt.figure(figsize=(10,7))

    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data, showmeans=True)
    plt.xticks([1, 2, 3, 4], ['accuracy', 'f1', 'precision', 'recall'])
    top_y = ax.get_ylim()[1]
    text = 'mean accuracy: {:.4f}\nmean f1:            {:.4f}\nmean precision: {:.4f}\nmean recall:      {:.4f}'.format(
        np.mean(acc), np.mean(f1), np.mean(precision), np.mean(recall)
    )
    plt.text(1, top_y - 0.05*top_y, text, fontsize=11)
    plt.show()

def mean_std(series):
    mean = np.mean(series)
    std = np.std(series)
    return mean, std

def populate_results_dataframe(results):
    accuracy_mean, accuracy_std = mean_std(results[0,:])
    f1_mean, f1_std = mean_std(results[1,:])
    precision_mean, precision_std = mean_std(results[2,:])
    recall_mean, recall_std = mean_std(results[3,:])

    results_dict = {
        'metric' : [],
        'lower_bound' : [],
        'mean' : [],
        'upper_bound' : []
    }

    results_dict['metric'].append('accuracy')
    results_dict['lower_bound'].append(accuracy_mean - accuracy_std)
    results_dict['mean'].append(accuracy_mean)
    results_dict['upper_bound'].append(accuracy_mean + accuracy_std)

    results_dict['metric'].append('f1')
    results_dict['lower_bound'].append(f1_mean - f1_std)
    results_dict['mean'].append(f1_mean)
    results_dict['upper_bound'].append(f1_mean + f1_std)

    results_dict['metric'].append('precision')
    results_dict['lower_bound'].append(precision_mean - precision_std)
    results_dict['mean'].append(precision_mean)
    results_dict['upper_bound'].append(precision_mean + precision_std)

    results_dict['metric'].append('recall')
    results_dict['lower_bound'].append(recall_mean - recall_std)
    results_dict['mean'].append(recall_mean)
    results_dict['upper_bound'].append(recall_mean + recall_std)

    df = pd.DataFrame.from_dict(results_dict)
    return df

def get_most_informative_logits(model, n, split=False, print_=False, vocab=None):
    '''Gets the n most informative logits
       if split=True, gets the n most informative logits from both the 'fake' side of the model and the 'real' side
       if print_=True, prints the tokens associated with the most informative logits
       
       returns a list of the indices of the most informative logits, or two lists if split=True'''

    if print_:
        assert vocab is not None

    if split:
        fake_idx = model.coef_.argsort()[0][n:][::-1]
        real_idx = model.coef_.argsort()[0][:n][::-1]
    

        if print_:    
            real_words = []
            fake_words = []
            for i in range(n):
                real_words.append(vocab[real_idx[i]])
                fake_words.append(vocab[fake_idx[i]])

            top_words = pd.DataFrame(list(zip(real_words, fake_words)), columns=['top "real" words', 'top "fake" words'])
            print(top_words)
        top_idx = fake_idx + real_idx


    else:
        top_idx = np.abs(model.coef_).argsort()[0][-n:][::-1]
        if print_:
            top_words = []
            for i in range(n): #what
                print('{}: {}'.format(i, vocab[top_idx[i]]))

    return list(top_idx.flatten())    

def logit_explained_variance(df, col, num_iterations=100):
    '''
    - The purpose of this cell is to get a dictionary with key value pairs {logit rank : weight}
    - Another dictionary will be constructed with {logit rank : positive or negative}
    - Another dictionary will be constructed with {logit rank : token pair}
    '''
    # vocab_score is arranged index : weight of vocab
    for i in tqdm(range(num_iterations)):
        model, _, X_test, _, y_test, _, vocab = train_logistic_regression(df, col, meta_name=False)
        if i == 0:
            vocab_score = np.zeros(len(vocab))
        vocab_score = vocab_score + model.coef_[0]

    vocab_score = vocab_score / num_iterations
    coef_sign = np.array([1 if x > 0 else 0 for x in list(vocab_score)])
    sorted_idx = np.flip(np.abs(vocab_score).argsort())
    
    logit_weights = {}
    logit_signs = {}
    logit_tokens = {}

    for i in range(len(vocab_score)):
        logit_weights[i+1] = vocab_score[sorted_idx[i]]
        logit_signs[i+1] = coef_sign[sorted_idx[i]]
        logit_tokens[i+1] = vocab[sorted_idx[i]]

    return logit_weights, logit_signs, logit_tokens


