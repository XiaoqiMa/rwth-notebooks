##################################################################
##
# Statistical Classification and Machine Learning
# Exercise 5
# Valeriia Volkovaia 384024, Xiaoqi Ma 383420, Paul Weiser 356606
##
##################################################################

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Gaussian_Classifier():

    def __init__(self, train_file, test_file, class_num=10, dimension=256, class_dep=True, full_cov=True):

        self.train_file = train_file
        self.test_file = test_file
        self.class_num = class_num
        self.dimension = dimension
        self.class_dep = class_dep
        self.full_cov = full_cov

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
        pooled_diag_cov: pooled diagonal covariance matrix
        or pooled_full_cov: pooled full covariance matrix
        or sigma: class-specific covariance matrix
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

            # get priors for class k
            priors[k] = float(len(data_k) / N_samples)

            # convert to matrix, then calculate mean on each column,
            # then convert back to ndarray, save to dictionay
            matrix_data_k = np.matrix(data_k)
            mu_k = np.mean(matrix_data_k, axis=0)
            mu[k] = np.array(mu_k)

            # Class-specific full covariance matrix
            if self.class_dep == True and self.full_cov == True:
                sigma_k = 0
                for n in range(len(data_k)):
                    vector_n = np.array(data_k[n])
                    sigma_k += np.dot((vector_n - mu_k).T, (vector_n - mu_k))
                sigma_k = sigma_k / len(data_k)
                sigma[k] = sigma_k

            # Class-specific diagonal covariance matrix
            if self.class_dep == True and self.full_cov == False:
                sigma_k = np.sum(np.square(matrix_data_k - mu_k), axis=0) / len(data_k)
                sigma[k] = sigma_k

            # Pooled full covariance matrix
            if self.class_dep == False and self.full_cov == True:
                sigma_k = 0
                for n in range(len(data_k)):
                    vector_n = np.array(data_k[n])
                    sigma_k += np.dot((vector_n - mu_k).T, (vector_n - mu_k))
                sigma_k = sigma_k / len(data_k)
                sigma[k] = sigma_k

            # Pooled diagonal covariance matrix
            if self.class_dep == False and self.full_cov == False:
                sigma_k = np.sum(np.square(matrix_data_k - mu_k), axis=0)
                sigma[k] = sigma_k

        # calculate the pooled full covariance matrix
        if self.class_dep == False and self.full_cov == True:
            pooled_full_cov = np.zeros(shape=(256,256))
            for k in range(1, 11):
                pooled_full_cov += sigma[k]
            pooled_full_cov = pooled_full_cov / N_samples
            return (mu, priors, pooled_full_cov)

        # calculate pooled diagonal covariance
        elif self.class_dep == False and self.full_cov == False:
            pooled_diag_cov = np.array([np.zeros(256)])
            for k in range(1, 11):
                pooled_diag_cov += sigma[k][0]
            pooled_diag_cov = pooled_diag_cov / N_samples
            return (mu, priors, pooled_diag_cov)
        else:
            return (mu, priors, sigma)

    """
        after getting the trained model, we use discriminant function
        to test dataset, to obtain the predicted class

        parameters:
        -------------
        ob_vectors: test vectors, list of tuples --> (class, vectors)
        mu: numpy array, mean vector for class k (1st to Dth component)
        priors: P(k), priors for class k
        sigma: covariance matrix (class specific or pooled)

        returns:
        ---------------
        predict_c: list of predicted classes
        true_c : list of true classes
    """

    def discriminant_function(self, estimators, ob_vectors, class_dep, full_cov):
        mu = estimators[0]
        priors = estimators[1]
        sigma = estimators[2]
        predict_c = []
        true_c = []
        if class_dep == False:
            if full_cov == True:
                sigma_ = sigma
            else:
                sigma_ = np.diag(sigma[0])

            inv_sigma_ = np.linalg.pinv(sigma_)
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
        else:
            sigma_inv = {}
            for k in range(1, 11):
                # calcualte pseudo inverse (avoid singular matrix)
                sigma_inv[k] = np.linalg.pinv(sigma[k])
            for i in range(len(ob_vectors)):
                g_k = []
                true_c.append(ob_vectors[i][0])
                # calcualte the likelihood for each class k
                for k in range(1, 11):
                    # g(x, k) = log(p(k)) - 0.5 * (x-mu_k)^T * inv(sigma) * (x-mu_k)
                    likelihood = np.log(priors[k]) - 0.5 * np.dot(np.dot(ob_vectors[i]
                                                                         [1] - mu[k][0], sigma_inv[k]), ob_vectors[i][1]-mu[k][0])
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

    def output_params(self, estimators, file_path):
        output_param_path = file_path
        mu = estimators[0]
        priors = estimators[1]
        sigma = estimators[2]
        pattern = r'[\[\]]'
        with open(output_param_path, 'w') as fw:
            if self.full_cov == True:
                fw.write('f\n')
            else:
                fw.write('d\n')
            fw.write('{0}\n'.format(str(self.class_num)))
            fw.write('{0}\n'.format(str(self.dimension)))
            for k in range(1, 11):
                fw.write('{}\n'.format(str(k)))
                fw.write('{}\n'.format(str(priors[k])))
                # remove brackets
                fw.write('{}\n'.format(re.sub(pattern, ' ', str(mu[k][0]))))
                if self.class_dep == True and self.full_cov == True:
                    for i in range(256):
                        fw.write('{}\n'.format(re.sub(pattern, ' ', str(sigma[k][i]))))
                elif self.class_dep == True and self.full_cov == False:
                    fw.write('{}\n'.format(re.sub(pattern, ' ', str(sigma[k]))))
                elif self.class_dep == False and self.full_cov == True:
                    for i in range(256):
                        fw.write('{}\n'.format(re.sub(pattern, ' ', str(sigma[i]))))
                else:
                    fw.write('{}\n'.format(re.sub(pattern, ' ', str(sigma[0]))))
    """
        calcualte empirical error with formula:
        empirical error rate = wrong calssifications / all events

        parameters:
        -------------
        predict_c: using trained model to predic class based on test dataset
        true_c: true class based on test dataset
    """

    def get_empirical_error(self, file_path, predict_c, true_c):
        em_error = 1 - sum([1 for i, j in zip(predict_c, true_c) if i == j]) / len(predict_c)
        with open(file_path, 'w') as fw:
            fw.write('empirical error rate: ' + str(em_error))
    """
        generate confusion matrix based on predicted class and true class

        parameters:
        --------------
        predict_c: using trained model to predic class based on test dataset
        true_c: true class based on test dataset
    """

    def get_confusion_matrix(self, file_path, predict_c, true_c):
        conf_matrix = confusion_matrix(true_c, predict_c)
        with open(file_path, 'w') as fw:
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

    returns:
    pd_clf: Gaussian classifier
    cf_estimators: estimators inlcuding Class-specific full covariance matrix
    cd_estimators: estimators including Class-specific diagonal covariance matrix
    test_ob_vectors: test observation vectors
"""
def image_recognition(train_f_path, test_f_path):

    ############################ Exercise 3  ##############################################
    os.makedirs('./Exercise3', exist_ok=True)
    clf = Gaussian_Classifier(train_f_path, test_f_path, class_num=10, dimension=256, class_dep=False, full_cov=False)
    train_ob_vectors = clf.read_ob_vectors(clf.train_file)
    test_ob_vectors = clf.read_ob_vectors(clf.test_file)

    # task (a): Implement the above estimators for the parameters: mu, priors, sigma
    mu, priors, sigma = clf.train_model(train_ob_vectors)
    estimators = (mu, priors, sigma)

    # task (b)
    clf.output_params(estimators, './Exercise3/usps_d.param')
    print('Exercise3: parameters output succeed')
    # task (c)
    # simplified discriminant function:
    # g(x, k) = log(p(k)) - 0.5 * (x-mu_k)^T * inv(sigma) * (x-mu_k)

    # task (d) & task (e)
    predict_c, true_c = clf.discriminant_function(estimators, test_ob_vectors, clf.class_dep, clf.full_cov)
    clf.get_empirical_error('./Exercise3/usps_d.error', predict_c, true_c)
    print('Exercise3: empirical error output succeed')
    clf.get_confusion_matrix('./Exercise3/usps_d.cm', predict_c, true_c)
    print('Exercise3: confusion matrix output succeed')

    ####################################################################################


    ##################################### Exercise 5 #######################################
    os.makedirs('./Exercise5', exist_ok=True)

    # task 1(a)
    # implementation --> train_model()
    cf_clf = Gaussian_Classifier(train_f_path, test_f_path, class_num=10, dimension=256, class_dep=True, full_cov=True)
    cf_estimators = cf_clf.train_model(train_ob_vectors)
    cf_clf.output_params(cf_estimators, './Exercise5/usps_cf.param')
    print('Exercise5 1(a): usps_cf.param output succeed')

    cd_clf = Gaussian_Classifier(train_f_path, test_f_path, class_num=10, dimension=256, class_dep=True, full_cov=False)
    cd_estimators = cd_clf.train_model(train_ob_vectors)
    cd_clf.output_params(cd_estimators, './Exercise5/usps_cd.param')
    print('Exercise5 1(a): usps_cd.param output succeed')

    pf_clf = Gaussian_Classifier(train_f_path, test_f_path, class_num=10, dimension=256, class_dep=False, full_cov=True)
    pf_estimators = pf_clf.train_model(train_ob_vectors)
    pf_clf.output_params(pf_estimators, './Exercise5/usps_pf.param')
    print('Exercise5 1(a): usps_pf.param output succeed')

    pd_clf = Gaussian_Classifier(train_f_path, test_f_path, class_num=10, dimension=256, class_dep=False, full_cov=False)
    pd_estimators = pd_clf.train_model(train_ob_vectors)
    pd_clf.output_params(pd_estimators, './Exercise5/usps_pd.param')
    print('Exercise5 1(a): usps_pd.param output succeed')


    # Task 1(b)
    predict_c, true_c = pf_clf.discriminant_function(pf_estimators, test_ob_vectors, pf_clf.class_dep, pf_clf.full_cov)
    pf_clf.get_empirical_error('./Exercise5/usps_pf.error', predict_c, true_c)
    print('Exercise5 1(b): usps_pf.error output succeed')

    predict_c, true_c = pd_clf.discriminant_function(pd_estimators, test_ob_vectors, pd_clf.class_dep, pd_clf.full_cov)
    pd_clf.get_empirical_error('./Exercise5/usps_pd.error', predict_c, true_c)
    print('Exercise5 1(b): usps_pd.error output succeed')

    return pd_clf, cf_estimators, cd_estimators, test_ob_vectors

"""
    plot error rate curve, save figure

    params:
    -----------------
    clf: Gaussian classifier
    cf_estimators: estimators inlcuding Class-specific full covariance matrix
    cd_estimators: estimators including Class-specific diagonal covariance matrix
    test_ob_vectors: test observation vectors

"""
# Task 1(c)
def plot_error_rate(clf, cf_estimators, cd_estimators, test_ob_vectors):
    lambda_ = [1, 0.5, 0.1, 0.01, 10**(-3), 10**(-4), 10**(-5), 10**(-6)]
    error_rate = {}
    for l in lambda_:
        new_sigma = {key: [] for key in list(range(1, 11))}
        for k in range(1, 11):
            new_sigma[k] = l * cf_estimators[2][k] + (1-l) * np.diag(cd_estimators[2][k].A1)
        new_estimators = (cf_estimators[0], cf_estimators[1], new_sigma)
        predict_c, true_c = clf.discriminant_function(new_estimators, test_ob_vectors, True, True)
        em_error = 1 - sum([1 for i, j in zip(predict_c, true_c) if i == j]) / len(predict_c)
        error_rate[np.log10(l)] = em_error * 100

    plt.plot(error_rate.keys(), error_rate.values())
    plt.xlabel('λ(logarithmic)')
    plt.ylabel('Error rate (%)')
    plt.title('Error rate curve')
    plt.show(block=False)
    plt.savefig('./Exercise5/error_rate.png')
    print('Exercise5: 1(c) Error rate curve plot has been saved')

if __name__ == '__main__':
    try:
        train_f_path = sys.argv[1]
        test_f_path = sys.argv[2]
        clf, cf_estimators, cd_estimators, test_ob_vectors = image_recognition(train_f_path, test_f_path)
        plot_error_rate(clf, cf_estimators, cd_estimators, test_ob_vectors)
    except:
        print('incorrect system arguments')
        print('Correct Format: python3 SCML_Exercise.py [train_file][test_file]')
