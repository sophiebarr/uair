# ********************************
# IMPORT REQUIRED MODULES
# ********************************
import sklearn_crfsuite
from collections import defaultdict

# ********************************
# TRAIN A LINEAR CHAIN CRF
# ********************************
def train(X,y):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True
    )

    crf.fit(X, y)
    
    return crf
    
# ********************************
# TRAIN TEMPORAL CLASSIFIER
# ********************************
def TCTrain(X,y):
    crf = train(X,y)
    
    return crf

# ********************************
# RUN SPATIAL CLASSIFIER PREDICTIONS
# ********************************
def TCPredict(X,crf):
    
    y_pred_freq = defaultdict(int)
    
    # Last three days
    y_pred = crf.predict_single(X)
    y_pred_label = y_pred[-1]    
    y_pred_prob = crf.predict_marginals_single(X)
    y_pred_freq[y_pred_label] = y_pred_prob[-1][y_pred_label]

    return [y_pred_freq, y_pred]
