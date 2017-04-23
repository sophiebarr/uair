# ********************************
# IMPORT REQUIRED MODULES
# ********************************
from sklearn.neural_network import MLPClassifier
import random
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from collections import defaultdict

m = 10
n = 3

# ********************************
# TRAIN AN ANN
# ********************************
def train(X,y):
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='logistic', max_iter = 1000)
    mlp.fit(X,y)

    return mlp

# ********************************
# PERFORM ITERATION
# ********************************
def CalculateInput(X_ulab,X_lab,y_lab):
        res = []    
    
        # calculate pearson correlation on roads
        Frx = [X_ulab[2], X_ulab[3]]
        Frk = [X_lab[2], X_lab[3]]

        res.append(pearsonr(Frx, Frk)[0])
        
        # calculate pearson correlation on poi
        Fpx = [X_ulab[4], X_ulab[5], X_ulab[6], X_ulab[7]]
        Fpk = [X_lab[4], X_lab[5], X_lab[6], X_lab[7]]
        
        res.append(pearsonr(Fpx, Fpk)[0])
        
        # calculate distance
        ulab_pos = [X_ulab[0], X_ulab[1]]
        lab_pos = [X_lab[0], X_lab[1]]
        res.append(euclidean(ulab_pos, lab_pos))
        
        # add AQI
        res.append(y_lab)
        
        return res
        
# ********************************
# UNLABELED GRID GENERATION
# ********************************
def SCInputGrid(X_lab,y_lab,X_ulab):
    
    X_in = []
    
    index = list(range(len(X_lab)))
    
    for j in random.sample(index, n):
        res = CalculateInput(X_lab[j],X_ulab,y_lab[j])
            
        for i in res:
            X_in.append(i)
    
    return X_in

# ********************************
# INPUT GENERATION
# ********************************
def SCInputGeneration(X_lab,y_lab):
   
    X_in = []
    y_in = []
    
    index = list(range(len(X_lab)))
    
    for i in index:
        X_ulab = X_lab[i]
        X_lab2 = list(X_lab)
        y_lab2 = list(y_lab)
        del(X_lab2[i])
        del(y_lab2[i])
    
        res = SCInputGrid(X_lab, y_lab, X_ulab)
        X_in.append(res)
        y_in.append(y_lab[i])
            
    return [X_in, y_in]


# ********************************
# TRAIN SPATIAL CLASSIFIER
# ********************************
def SCTrain(X,y):
    [X,y] = SCInputGeneration(X,y)
    mlp = train(X,y)

    return mlp

# ********************************
# RUN SPATIAL CLASSIFIER PREDICTIONS
# ********************************
def SCPredict(X_lab, y_lab, X_ulab, mlp):
    
    y_pred_freq = defaultdict(int)
    
    for i in list(range(m)):
        X = SCInputGrid(X_lab,y_lab,X_ulab)
        y_pred = mlp.predict([X])
        y_pred_freq[int(y_pred)] += 1
    
    return y_pred_freq
