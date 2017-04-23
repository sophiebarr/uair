from SpatialClassifier import SCTrain,SCPredict
from TemporalClassifier import TCTrain,TCPredict
from collections import defaultdict
import operator

# ********************************
# SPATIAL CLASSIFER LABELLED DATA (G1)
# ********************************
def SCLabelledData():
    X = [[0.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [2.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [3.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [4.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [5.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [6.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [7.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [8.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [9.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0]
    ]
    
    y = [0,1,2,0,1,2,3,0,1,0]
    
    return {'X':X,'y':y}

# ********************************
# SPATIAL CLASSIFER UNLABELLED DATA (G2)
# ********************************
def SCUnlabelledData():
    X = [[0.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [2.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [3.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [4.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [5.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [7.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [8.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0],
    [9.0, 5.0, 1.3, 9.0, 1.0, 0.0, 0.0, 0.0]
    ]
    
    return {'X':X}

# ********************************
# TEMPORAL CLASSIFER LABELLED DATA (G1)
# ********************************
def TCLabelledData():
    X = [
            # Grid (0,0)
            [
                {
                    'time': 1,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 2,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 3,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 4,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 5,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 6,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 7,
                    'temp': 20.0,
                    'ws': 5
                }
            ],
            # Grid (0,1)
            [
                {
                    'time': 10,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 11,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 12,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 13,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 14,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 15,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 16,
                    'temp': 20.0,
                    'ws': 5
                }
            ]
        ]
    
    y = [
            # Grid (0,0)
            [
                '0',
                '0',
                '0',
                '1',
                '2',
                '2',
                '2'
             ],
             # Grid (0,1)
             [
                '0',
                '0',
                '0',
                '1',
                '2',
                '2',
                '2'
             ]
        ]

    return {'X':X,'y':y}

# ********************************
# TEMPORAL CLASSIFER UNLABELLED DATA (G2)
# ********************************
def TCUnlabelledData():
    X = [
            # Grid (0,0)
            [
                {
                    'time': 100,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 101,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 102,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 103,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 104,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 105,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 106,
                    'temp': 20.0,
                    'ws': 5
                }
            ],
            # Grid (0,1)
            [
                {
                    'time': 107,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 108,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 109,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 110,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 111,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 112,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 113,
                    'temp': 20.0,
                    'ws': 5
                }
            ],
            # Grid (0,2)
            [
                {
                    'time': 107,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 108,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 109,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 110,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 111,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 112,
                    'temp': 20.0,
                    'ws': 5
                },
                {
                    'time': 113,
                    'temp': 20.0,
                    'ws': 5
                }
            ]
        ]

    return {'X':X}

# ********************************
# TC TRAIN
# ********************************
def TCCoTrain(TC_X_Lab, TC_y_Lab, TC_X_Ulab):
    tc = TCTrain(TC_X_Lab, TC_y_Lab)    
    preds = []
    y_preds = []
    aqis = defaultdict()
    for i in range(len(TC_X_Ulab)):
        [pred, y] = TCPredict(TC_X_Ulab[i], tc)
        preds.append(pred)
        y_preds.append(y)

        for aqi in pred:
            aqis.setdefault(aqi,{})
            aqis[aqi].setdefault(i,{})
            aqis[aqi][i] = pred[aqi]
    
    best_grids = []
    for aqi in aqis:
        sorted_x = sorted(aqis[aqi].items(), key=operator.itemgetter(1), reverse=True)
        aqis[aqi] = [int(i[0]) for i in sorted_x][0]
        best_grids.append(aqis[aqi])

    best_grids = sorted(list(set(best_grids)), reverse=True)

    for g in range(len(preds)):
        preds[g] = sorted(preds[g].items(), key=operator.itemgetter(1), reverse=True)
        preds[g] = [i[0] for i in preds[g]][0]
        
    return [y_preds, best_grids]

# ********************************
# SC TRAIN
# ********************************
def SCCoTrain(SC_X_Lab, SC_y_Lab, SC_X_Ulab):
    sc = SCTrain(SC_X_Lab, SC_y_Lab)

    preds = []
    aqis = defaultdict()
    for i in range(len(SC_X_Ulab)):
        pred = SCPredict(SC_X_Lab, SC_y_Lab, SC_X_Ulab[i], sc)
        preds.append(pred)
        for aqi in pred:
            aqis.setdefault(aqi,{})
            aqis[aqi].setdefault(i,{})
            aqis[aqi][i] = pred[aqi]
    
    best_grids = []
    for aqi in aqis:
        sorted_x = sorted(aqis[aqi].items(), key=operator.itemgetter(1), reverse=True)
        aqis[aqi] = [int(i[0]) for i in sorted_x][0]
        best_grids.append(aqis[aqi])

    best_grids = sorted(list(set(best_grids)), reverse=True)

    for g in range(len(preds)):
        preds[g] = sorted(preds[g].items(), key=operator.itemgetter(1), reverse=True)
        preds[g] = [i[0] for i in preds[g]][0]
    
    return [preds, best_grids]

# ********************************
# TRAIN THE MODELS ON THE LABELLED DATA
# ********************************
def CoTrain():
    SC_X_Lab = SCLabelledData()['X']
    SC_y_Lab = SCLabelledData()['y']
    SC_X_Ulab = SCUnlabelledData()['X']

    TC_X_Lab = TCLabelledData()['X']
    TC_y_Lab = TCLabelledData()['y']
    TC_X_Ulab = TCUnlabelledData()['X']
    
    theta = 50
    i_cnt = 0
    while (len(SC_X_Ulab) > 0) and (i_cnt < theta):
        [preds, best_grids] = SCCoTrain(SC_X_Lab, SC_y_Lab, SC_X_Ulab)
        
        for g in best_grids:
            SC_X_Lab.append(SC_X_Ulab[g])
            SC_y_Lab.append(preds[g])
            del(SC_X_Ulab[g])
        
        [preds, best_grids] = TCCoTrain(TC_X_Lab, TC_y_Lab, TC_X_Ulab)
                
        for g in best_grids:
            TC_X_Lab.append(TC_X_Ulab[g])
            TC_y_Lab.append(preds[g])
            del(TC_X_Ulab[g])
        
        i_cnt += 1
        
    return sc

# ********************************
# MAIN
# ********************************
sc = CoTrain()
