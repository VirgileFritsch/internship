"""
Module for computation of statistics in replacement of
nipy.neurospin, which provides erratic result
"""

import numpy as np

def generate_data(gmean, gvar, indiv_var):
    """Generates a group of individuals from the provided parameters

    Parameters
    ----------
    gmean: float, group mean
    gvar: float, group variance
    indiv_var: array of shape(nsubj),
               the individual variances 

    Returns
    -------
    data: array of shape(nsubj)
          the individual data related to the two-level normal model
    """
    # check that the variances are  positive
    nsubj = len(indiv_var)
    data = np.random.randn(nsubj)
    data *= np.sqrt(gvar+indiv_var)
    data += gmean
    return data


def t_stat(data):
    """Returns the t stat of the sample on each row of the matrix

    Parameters
    ----------
    data, array of shape (n_datasets, n_samples)

    Returns
    -------
    t_variates, array of shape (n_datasets)
    
    """
    return data.mean(1)/data.std(1)*np.sqrt(data.shape[1])



class MixedEffectModel(object):
    """
    Class to handle simple mixed effects models
    """
    
    def __init__(self, data, vardata, mean=None):
        """
        Sets the effects and first-level variance,
        and initializes related quantities
        
        Parameters
        ----------
        data, array of shape (n_datasets, n_samples)
              the estimated effects
        vardata, array of shape (n_datasets, n_samples)
                 first-level variance
        mean, array of shape (n_datasets), optional
              mean for each sample
              (by default, it is estimated from the data)
        """
        # Basic data checks
        if (vardata<0).any():
            raise ValueError,"a negative variance has been provided"

        if np.size(data)==data.shape[0]:
            data = np.reshape(data,(1,np.size(data)))

        if np.size(vardata)==vardata.shape[0]:
            data = np.reshape(vardata,(1,np.size(vardata)))

        if data.shape != vardata.shape:
            raise ValueError, "data and vardata do not have the same shape"

        self.nsamples = data.shape[1]
        self.ndataset = data.shape[0]
        self.data = data
        self.vardata = vardata
        if mean==None:
            self.var = data.var(1)
            self.mean = data.mean(1)
        else:
            if np.size(mean)!=self.ndataset:
                raise ValueError, 'incorrect dimension for mean'
            self.mean = np.squeeze(mean)
            self.var = np.mean((data.T-self.mean)**2, 0)
        

    def log_like(self):
        """
        Returns the log-likelihood of self (array of shape self.ndataset)
        """
        tvar = self.var + self.vardata.T
        ll = np.sum((self.data.T - self.mean)**2/tvar, 0)
        ll += np.sum(np.log(tvar), 0)
        ll += np.log(2*np.pi)*self.nsamples
        ll *= -0.5
        return ll

    def one_step(self, mean=None):
        """
        Applies one step of an EM algorithm to estimate self.mean and self.var

        Parameters
        ----------
        mean=None: array of shape (self.ndataset),
                   if note None, self.mean is forced to be equal to it 
        """
        if mean==None:
            mean = self.mean
        
        # E step
        prec = 1./(self.var + self.vardata.T)
        cdata = prec * (self.vardata.T * mean + self.var * self.data.T)
        cvar = self.vardata.T * self.var * prec

        # M step
        if mean==None:
            self.mean = cdata.mean(0)
        self.var = cvar.mean(0) + cdata.var(0) 

    def fit(self, mean=None, niter=5, verbose=0):
        """
        Launches the EM algorithm to estimate self

        Parameters
        ----------
        mean=None: array of shape (self.ndataset),
                   if note None, self.mean is forced to be equal to it
        niter=5, integer, number of iterations of the EM algorithm
        verbose = 0, verbosity mode
        """
        if verbose:
            print self.log_like()
        for i in range(niter):
            self.one_step(mean)
            if verbose:
                print i, self.log_like()
        
    
def mfx_t_stat(data, vardata, niter=5):
    """
    Returns the mixed effects stat for each row of the data
    (one sample test)
    This uses the Formula in Roche et al., Neuyroimage 2007

    Parameters
    ----------
    data, array of shape (n_datasets, n_samples)
          the estimated effects
    vardata, array of shape (n_datasets, n_samples)
             first-level variance
    niter: int, optional,
	   nuber of iterations of the EM algorithm

    Returns
    -------
    tstat, array of shape (n_datasets),
           statistical values obtained from the likelihood ratio test
    """
    zmean = np.zeros(data.shape[0])
    M1 = MixedEffectModel(data,vardata)
    M1.fit(niter=niter)
    M0 = MixedEffectModel(data, vardata, mean=zmean)
    M0.fit(mean=zmean, niter=niter)
    dll = 2*(M1.log_like()-M0.log_like())
    tstat = np.sign(M1.mean)*np.sqrt(dll)
    tstat[dll<0] = 0
    return tstat

def compare_with_nipy():
    """
    Check whether we obtain the same results as nipy
    """
    import nipy.neurospin.group.onesample as fos
    nsamples = 15
    ndata = 1000
    
    vardata = np.random.rand(ndata,nsamples)
    data = np.zeros((ndata, nsamples))
    for i in range(ndata):
        data[i] = generate_data(0, 1, vardata[i])
        
    # my computation
    t1 =  t_stat(data)
    t2 =  mfx_t_stat(data,vardata)

    #nipy model
    t3 = fos.stat(data, id='student', axis=1)
    t4 = np.squeeze(fos.stat_mfx(data, vardata, id='student_mfx', axis=1))
    diff = t4-t2
    maxdiff = (diff**2).max()
    if maxdiff>1:
	i = np.abs(diff).argmax()
	print t1[i], t2[i], t4[i]
	print mfx_t_stat(np.array([data[i]]),np.array([vardata[i]]), niter=20)
    print ((t1-t2)**2).sum(), ((t1-t4)**2).sum() 

if __name__ == "__main__":

    nsamples = 15
    ndata = 50000

    vardata = np.random.rand(ndata,nsamples)
    data = np.zeros((ndata, nsamples))
    for i in range(ndata):
        data[i] = generate_data(0, 1, vardata[i])
        

    t1 =  t_stat(data)
    t2 =  mfx_t_stat(data,vardata)
    import pylab
    pylab.figure()
    pylab.plot(t1,t2,'.')
    pylab.show()

