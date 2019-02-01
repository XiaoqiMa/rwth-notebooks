##################################################################
##
# Statistical Classification and Machine Learning
# Exercise 3
##
##################################################################

import numpy as np
import sys
from sklearn.metrics import confusion_matrix


class Gaussian_Classifier():

    def __init__(self):
        pass

    """
        read observation vectors and true class from train/test files
        parameters:
        ------------
        file_path: read observation vectors from designated file path

        returns:
        ------------
        (class, ob_vectors) pairs
    """

    def read_ob_vectors(self, file_path):

        with open(file_path) as fr:
            lines = fr.readlines()

        # ommit the first two line, starting from line 3
        lines = lines[2:]
        ob_vectors = []
        l_count = 0  # keep track of the line count

        while l_count < len(lines):
            data = []
            c = int(lines[l_count].strip())  # record the true class
            l_count += 1  # move the line cursor below one line

            # remove space, each line as a list of float
            for i in range(l_count, l_count + 16):
                data += [float(x) for x in lines[i].split()]

            l_count += 16  # move line cursor to next image
            data = np.array(data)  # convert data to numpy array
            ob_vectors.append((c, data))
        return ob_vectors

    """
        train Gaussian Model with train observation vectors
        parameters:
        --------------
        ob_vectors: trained vectors, list of tuples --> (class, vectors)

        returns:
        ---------------
        mu: numpy array, mean vector for class k (1st to Dth component)
        priors: P(k), priors for class k
        pooled_cov: pooled covariance matrix (same for all classes)
    """

    def train_model(self, ob_vectors):
        # initialize dictionay with keys defined by 10 classes
        mu = {key: [] for key in list(range(1, 11))}
        priors = {key: [] for key in list(range(1, 11))}
        sigma = {key: [] for key in list(range(1, 11))}
        N_samples = len(ob_vectors)

        for k in range(1, 11):
            # each iteration, get a subset of data with designated k
            data_k = [data_k for c, data_k in ob_vectors if c == k]

            priors[k] = float(len(data_k) / N_samples)  # get priors for class k

            # convert to matrix, then calculate mean on each column,
            # then convert back to ndarray, save to dictionay
            matrix_data_k = np.matrix(data_k)
            mu_k = np.mean(matrix_data_k, axis=0)
            mu[k] = np.array(mu_k)
            # sigma vector for specific k, return type: numpy matrix
            sigma_k = np.sum(np.square(matrix_data_k - mu_k), axis=0)
            sigma[k] = sigma_k
        # calculate pooled covariance according to the fomular
        pooled_cov = np.array([np.zeros(256)])
        for k in range(1, 11):
            pooled_cov += sigma[k][0]
        pooled_cov = pooled_cov / N_samples

        return mu, priors, pooled_cov

    """
        after getting the trained model, we use discriminant function
        to test dataset, to obtain the predicted class

        parameters:
        -------------
        ob_vectors: test vectors, list of tuples --> (class, vectors)
        mu: numpy array, mean vector for class k (1st to Dth component)
        priors: P(k), priors for class k
        pooled_cov: pooled covariance matrix (same for all classes)

        returns:
        ---------------
        predict_c: list of predicted classes
        true_c : list of true classes
    """

    def discriminant_function(self, estimators, ob_vectors):
        mu = estimators[0]
        priors = estimators[1]
        sigma = estimators[2]
        sigma_ = np.diag(sigma[0])
        inv_sigma_ = np.linalg.inv(sigma_)
        predict_c = []
        true_c = []
        for i in range(len(ob_vectors)):
            # convert array to diagonal matrix
            g_k = []
            true_c.append(ob_vectors[i][0])
            # calcualte the likelihood for each class k
            for k in range(1, 11):
                # g(x, k) = log(p(k)) - 0.5 * (x-mu_k)^T * inv(sigma) * (x-mu_k)
                likelihood = np.log(priors[k]) - 0.5 * np.dot(np.dot(ob_vectors[i]
                                                                     [1] - mu[k][0], inv_sigma_), ob_vectors[i][1]-mu[k][0])
                g_k.append(likelihood)
            # predicted class = indice of maximum likelihood + 1(since indice starts from 0)
            c = np.argmax(g_k) + 1
            predict_c.append(c)
        return predict_c, true_c

    """
        write estimators to file

        parameters:
        --------------
        estimators: tuple of (mu, priors, sigma)
    """

    def output_params(self, estimators):
        output_param_path = 'usps_d.param'
        mu = estimators[0]
        priors = estimators[1]
        sigma = estimators[2]
        with open(output_param_path, 'w') as fw:
            fw.write('d\n')
            fw.write('10\n')
            fw.write('256\n')
            for k in range(1, 11):
                fw.write('{}\n'.format(str(k)))
                fw.write('{}\n'.format(str(priors[k])))
                fw.write('{}\n'.format(str(mu[k][0])))
                fw.write('{}\n'.format(str(sigma[0])))
    """
        calcualte empirical error with formula:
        empirical error rate = wrong calssifications / all events

        parameters:
        -------------
        predict_c: using trained model to predic class based on test dataset
        true_c: true class based on test dataset
    """

    def get_empirical_error(self, predict_c, true_c):
        em_error = 1 - sum([1 for i, j in zip(predict_c, true_c) if i == j]) / len(predict_c)
        with open('usps_d.error', 'w') as fw:
            fw.write('empirical error rate: ' + str(em_error))
    """
        generate confusion matrix based on predicted class and true class

        parameters:
        --------------
        predict_c: using trained model to predic class based on test dataset
        true_c: true class based on test dataset
    """

    def get_confusion_matrix(self, predict_c, true_c):
        conf_matrix = confusion_matrix(true_c, predict_c)
        with open('usps_d.cm', 'w') as fw:
            for i in range(conf_matrix.shape[0]):
                # convert each row of the matrix to string (prepare to write into file)
                str_matrix = [str(x) for x in conf_matrix[i]]
                fw.write('\t'.join(str_matrix) + '\n')

"""
    start image recognition

    parameters:
    --------------
    train_f_path: file path to train dataset
    test_f_path: file path to test dataset
"""
def image_recognition(train_f_path, test_f_path):

    clf = Gaussian_Classifier()
    train_ob_vectors = clf.read_ob_vectors(train_f_path)
    test_ob_vectors = clf.read_ob_vectors(test_f_path)

    # task (a): Implement the above estimators for the parameters: mu, priors, sigma
    mu, priors, sigma = clf.train_model(train_ob_vectors)
    estimators = (mu, priors, sigma)

    # task (b)
    clf.output_params(estimators)
    print('parameters output succeed')
    # task (c)
    # simplified discriminant function:
    # g(x, k) = log(p(k)) - 0.5 * (x-mu_k)^T * inv(sigma) * (x-mu_k)

    # task (d) & task (e)
    predict_c, true_c = clf.discriminant_function(estimators, test_ob_vectors)
    clf.get_empirical_error(predict_c, true_c)
    print('empirical error output succeed')
    clf.get_confusion_matrix(predict_c, true_c)
    print('confusion matrix output succeed')

if __name__ == '__main__':
    try:
        train_f_path = sys.argv[1]
        test_f_path = sys.argv[2]
        image_recognition(train_f_path, test_f_path)
    except:
        print('incorrect system arguments')
        print('Correct Format: python3 SCML_Exercise.py [train_file][test_file]')
