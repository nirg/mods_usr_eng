# coding=utf-8
"""Latent Dirichlet allocation using collapsed Gibbs sampling"""

from __future__ import absolute_import, division, unicode_literals  # noqa
import logging
import sys

import numpy as np
from scipy.special import gamma, gammaln  # , psi

import _tune
import _lda
import tune_utils

import dirichlet as dir

logger = logging.getLogger('tune')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange


class Tune:
    """Latent Dirichlet allocation using collapsed Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 2000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    eta : float, default 0.01
        Dirichlet parameter for distribution over words

    random_state : int or RandomState, optional
        The generator used for the initial topics.

    Attributes
    ----------
    `components_` : array, shape = [n_topics, n_features]
        Point estimate of the topic-word distributions (Phi in literature)
    `topic_word_` :
        Alias for `components_`
    `nzw_` : array, shape = [n_topics, n_features]
        Matrix of counts recording topic-word assignments in final iteration.
    `ndz_` : array, shape = [n_samples, n_topics]
        Matrix of counts recording document-topic assignments in final iteration.
    `doc_topic_` : array, shape = [n_samples, n_features]
        Point estimate of the document-topic distributions (Theta in literature)
    `nz_` : array, shape = [n_topics]
        Array of topic assignment counts in final iteration.
    `psi_`: array, shape = [k, n_eng_modes]
        Point estimate of topic-engagement distributions (Psi in literature)
    `ndj_`: array, shape = [n_samples, n_eng_modes]
        Point estimate of document-engagement mode  distribution

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> import lda
    >>> model = lda.LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.

    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.

    """

    def __init__(self, n_topics, n_iter=2000, alpha=0.1, eta=0.01, random_state=None,
                 refresh=10):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        rng = tune_utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    def fit(self, X, Y, ll_tol=1e-4):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        Y: array-like, shape (n_samples, n_eng_modes)
            Enagement modes proportions observed in data, where n_samples in the number of samples
            and n_eng_modes is the number of engagement modes. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = tune_utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X, Y)
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        ydz = np.zeros((self.Y.shape[0], self.n_topics), dtype=np.float64)
        for it in range(self.n_iter):
            # print "Starting iteration " + str(it)
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                print "<{}> log likelihood: {:.0f}".format(it, ll)
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
                if it >= 50 and len(self.loglikelihoods_) > 1:
                    ll_prev = self.loglikelihoods_[-2]
                    if np.abs((ll - ll_prev) / ll) < ll_tol * self.refresh:
                        break
            # Samples topic assignments
            for k in xrange(self.n_topics):
                ydz[:, k] = self._psi_coefs[k] * np.prod(np.power(self.Y, self.psi_[k, :] - 1), 1)
            _tune._sample_topics(self.WS, self.DS, self.ZS, ydz, self.nzw_, self.ndz_, self.nz_, alpha, eta, rands)
            # self._sample_topics(ydz, alpha, eta, rands)
            self._update_psi()
        ll = self.loglikelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(it + 1, ll))
        # print "<{}> log likelihood: {:.0f}".format(it + 1, ll)
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X, Y):
        print "initializing tune..."
        D, W = X.shape
        D2, E = Y.shape
        np.testing.assert_equal(D, D2)
        self.Y = Y
        self._logy = np.log(Y)
        self.n_eng_modes = E
        N = int(X.sum())
        self.nd_ = np.sum(X, 1, dtype=np.intc)
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = tune_utils.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        self._update_psi()
        self.loglikelihoods_ = []

    def _update_psi(self):
        new_psi = np.zeros((self.n_topics, self.n_eng_modes), dtype=np.float64)
        for k in xrange(self.n_topics):
            if np.sum(self.ndz_[:, k]) > 0.0:
                new_psi[k, :] = dir.mle(self.Y, weights=self.ndz_[:, k])
            else:
                new_psi[k, :] = self.eta
        self._psi_coefs = gamma(np.sum(new_psi, 1)) / np.prod(gamma(new_psi), 1)
        self.psi_ = new_psi

    def train_test(self, X_train, Y_train, X_test, n_iter_test=100, ll_tol=1e-4):
        """ Fit the model with X.

        Parameters
        ----------
        X_train: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        Y_train: array-like, shape (n_samples, n_eng_modes)
            Enagement modes proportions observed in data, where n_samples in the number of samples
            and n_eng_modes is the number of engagement modes. Sparse matrix allowed.
        X_test: array-like, shape (n_test_samples, n_features)
            Training data, where n_test_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit(X_train, Y_train)

        random_state = tune_utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize_for_test(X_test, n_iter_test)
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)

        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood_inference()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                print "<{}> log likelihood: {:.0f}".format(it, ll)
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
                # if it>=50 and len(self.loglikelihoods_) > 1:
                #     ll_prev = self.loglikelihoods_[-2]
                #     if np.abs((ll - ll_prev)/ll) < ll_tol * self.refresh:
                #         break
                _tune._infer_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                    alpha, eta, rands)
        ll = self.loglikelihood_inference()
        logger.info("<{}> log likelihood: {:.0f}".format(it + 1, ll))
        # print "<{}> log likelihood: {:.0f}".format(it + 1, ll)

        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]
        exp_eng = self.psi_ / np.sum(self.psi_, axis=1)[:, np.newaxis]
        self.y_hat_ = np.dot(self.doc_topic_, exp_eng)
        self.y_hat_ /= np.sum(self.y_hat_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        # del self.ES
        return self

    def _initialize_for_test(self, X, n_iter_test):
        print "initializing tune for testing..."
        D, W = X.shape
        N = int(X.sum())
        self.nd_ = np.sum(X, 1, dtype=np.intc)
        n_topics = self.n_topics
        self.n_iter = n_iter_test
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        # self.ndj_ = ndj_ = np.zeros((D, self.n_eng_modes), dtype=np.intc)
        self.WS, self.DS = WS, DS = tune_utils.matrix_to_lists(X)
        self.ZS = ZS = np.random.choice(n_topics, size=N, p=self.nz_.astype(float) / np.sum(self.nz_)).astype(np.intc)
        # self.ES = ES = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            ndz_[DS[i], ZS[i]] += 1
            # e_new = i % self.n_eng_modes
            # ES[i] = e_new
            # ndj_[d, e_new] += 1
        self.loglikelihoods_ = []

    def fit_transform(self, X, y=None):
        """Apply dimensionality reduction on X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        self._fit(X)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        This uses the "iterated pseudo-counts" approach described
        in Wallach et al. (2009) and discussed in Buntine (2009).

        """
        if isinstance(X, np.ndarray):
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = tune_utils.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1):  # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis]  # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z,y) = log p(w|z) + log p(y|z) + log p(z)
        """
        psi_ll = 0.0
        for k in xrange(self.n_topics):
            # Weighted averaging of observations
            if np.sum(self.ndz_[:, k]) <= 0.0:
                continue
            logp = np.average(self._logy, axis=0, weights=self.ndz_[:, k])
            a = self.psi_[k, :]
            psi_ll += np.sum(self.ndz_[:, k]) * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())
        lda_ll = self.loglikelihood_inference()
        # print str(psi_ll) + " + " + str(lda_ll)
        return psi_ll + lda_ll

    def loglikelihood_inference(self):
        """ Computes the log-likelihood using just the LDA part:
            log p(w,z) = log p(w|z) + log p(z)
        """
        return _lda._loglikelihood(self.nzw_, self.ndz_, self.nz_, self.nd_,
                                   self.alpha, self.eta)

if __name__ == "__main__":

    loaded = np.load('input_example.npz')
    X_train = loaded['X_train']  # Doc to term matrix (300 docs by 12832 terms in this case)
    Y_train = loaded['Y_train']  # Engagement mode proportions (rows sum up to 1)
    X_test = loaded['X_test']    # Test documents
    del loaded

    k = 5
    model = Tune(n_topics=k, n_iter=100, alpha=50.0 / k, eta=0.1, random_state=1, refresh=1)
    model.train_test(X_train, Y_train, X_test, n_iter_test=100)

    print 'Here are the first 10 predictions:'
    print model.y_hat_[:10, :]
