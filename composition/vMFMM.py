import numpy as np
import scipy
if tuple(map(int, scipy.__version__.split('.'))) < (1, 0, 0):
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
import time

# Use k++ algorithm to initialize mu, the kernel centres

# Normalize the features by dividing the summed sqrt root
def normalize_features(features):
	'''features: n by d matrix'''
	assert(len(features.shape)==2)
	norma=np.sqrt(np.sum(features ** 2, axis=1).reshape(-1, 1))+1e-6
	return features/norma

class vMFMM:
	# input number of classes -> vMF kernel numbers 512
	# initialization method is random
	def __init__(self, cls_num, init_method = 'k++'):
		self.cls_num = cls_num
		self.init_method = init_method

	# features: extracted features of all data
	# kappa: vMF kernels
	# max_it: maximum iterations
	# tol: toleration to converge
	# normalized: normalize the features or not
	def fit(self, features, kappa, max_it=300, tol=5e-5, normalized=False):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)

		# features shape: nxd
		self.n, self.d = self.features.shape
		self.kappa = kappa

		# self.pi: self.cls_num random floats in [0, 1], then normalize it by dividing with the summation
		self.pi = np.random.random(self.cls_num)
		self.pi /= np.sum(self.pi)

		# random initialization: mu is cls_num x d random floats in [0, 1]
		# then normalise mu
		if self.init_method =='random':
			self.mu = np.random.random((self.cls_num, self.d))
			self.mu = normalize_features(self.mu)
		# k++ initialization
		elif self.init_method =='k++':
			centers = []
			centers_i = []

			# only use 50000 feature vectors
			if self.n > 50000:
				rdn_index = np.random.choice(self.n, size=(50000,), replace=False)
			else:
				rdn_index = np.array(range(self.n), dtype=int)

			# calculate cosine distance
			cos_dis = 1-np.dot(self.features[rdn_index], self.features[rdn_index].T)

			centers_i.append(np.random.choice(rdn_index))
			centers.append(self.features[centers_i[0]])
			for i in range(self.cls_num-1):

				cdisidx = [np.where(rdn_index==cci)[0][0] for cci in centers_i]
				prob = np.min(cos_dis[:,cdisidx], axis=1)**2
				prob /= np.sum(prob)
				centers_i.append(np.random.choice(rdn_index, p=prob))
				centers.append(self.features[centers_i[-1]])

			self.mu = np.array(centers)
			del(cos_dis)

		self.mllk_rec = []
		for itt in range(max_it):
			_st = time.time()
			self.e_step()
			self.m_step()
			_et = time.time()
			itr_time = _et-_st
			self.mllk_rec.append(self.mllk)
			print('K++ clustering iteration{}'.format(itt))
			print('One iteration takes{}'.format(itr_time))
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break

	def e_step(self):
		# update p
		logP = np.dot(self.features, self.mu.T)*self.kappa + np.log(self.pi).reshape(1,-1)  # n by k
		logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
		self.p = np.exp(logP_norm)
		self.mllk = np.mean(logsumexp(logP, axis=1))


	def m_step(self):
		# update pi and mu
		self.pi = np.sum(self.p, axis=0)/self.n

		# fast version, requires more memory
		self.mu = np.dot(self.p.T, self.features)/np.sum(self.p, axis=0).reshape(-1,1)

		self.mu = normalize_features(self.mu)





