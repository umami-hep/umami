import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt

def sigBkgEff(myModel, X_test, y_test, returnDisc=False, fc=0.07):

    '''
    Given a model, make the histograms of the model outputs to get the ROC curves.

    Input:
        myModel: A keras model
        X_test: Model inputs of the test set
        y_test: Truth labels for the test set
        returnDisc: If True, also return the raw discriminant 
        fc: The amount by which to weight the c-jet prob in the disc. The
            default value of 0.07 corresponds to the fraction of c-jet bkg
            in ttbar.

    Output:
        effs: A list with 3 entries for the l, c, and b effs
        disc: b-tagging discriminant (will only be returned if returnDisc is True)
    '''

    # Evaluate the performance with the ROC curves!
    predictions = myModel.predict(X_test,verbose=True)

    # To make sure you're not discarding the b-values with high
    # discriminant values that you're good at classifying, use the
    # max from the distribution
    disc = np.log(np.divide(predictions[:,2], fc*predictions[:,1] + (1 - fc) * predictions[:,0]))
    
    '''
    Note: For jets w/o any tracks
    '''
    
    discMax = np.max(disc)
    discMin = np.min(disc)
    
    myRange=(discMin,discMax)
    nBins = 200

    effs = []
    plt.figure()
    for output, flavor in zip([0,1,2], ['l','c','b']):

        ix = (np.argmax(y_test,axis=-1) == output)
        
        # Plot the discriminant output
        nEntries, edges ,_ = plt.hist(disc[ix],alpha=0.5,label='{}-jets'.format(flavor),
                                      bins=nBins, range=myRange, density=True, log=True)

        '''
        nEntries is just a sum of the weight of each bin in the histogram.
        
        
        Since high Db scores correspond to more b-like jets, compute the cummulative density function
        from summing from high to low values, this is why we reverse the order of the bins in nEntries
        using the "::-1" numpy indexing.
        '''
        eff = np.add.accumulate(nEntries[::-1]) / np.sum(nEntries)
        effs.append(eff)

    plt.legend()
    plt.xlabel('$D = \ln [ p_b / (f_c p_c + (1- f_c)p_l ) ]$',fontsize=14)
    plt.ylabel('"Normalized" counts')

    if returnDisc:
        return effs, disc
    else:
        return effs