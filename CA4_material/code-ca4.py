import numpy as np
import random
import math
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


## TRAINING STAGE

def external_node(T):
    if T['Left']==None and T['Right']==None:
        return 1
    return 0

def c(n):
    if n==0:
        return 0 # not sure about this
    return 2*(np.log(n)+0.5772156649)-2*(n-1)/n

def PathLength(x,T,e):
    if external_node(T):
        return e + c(T['Size'])
    a = T['SplitAtt']
    if x[a] < T['SplitValue']:
        return PathLength(x,T['Left'],e+1)
    return PathLength(x,T['Right'],e+1)

def exNode(size): # this function return an external node
    return {'Size':size, 'Left':None, 'Right':None}

def inNode(left,right,splitatt,splitval): # this function return an internal node
    return {'Left':left, 'Right':right, 'SplitAtt':splitatt, 'SplitValue':splitval}

def randomly_select_attribute(data):
    random_record = random.choice(data)
    keys = list(random_record.keys())
    keys.remove('PathLength')
    random_attribute_name = random.choice(keys)
    # random_attribute_value = random_record[random_attribute_name]
    return random_attribute_name

def randomly_select_split_point(data, attribute_name):
    attribute_values = [record[attribute_name] for record in data if record[attribute_name] is not None]
    max_value = max(attribute_values)
    min_value = min(attribute_values)
    split_point = random.uniform(min_value, max_value)
    return split_point

def filter_data_by_attribute(data, attribute_name, split_point, sign):
    if sign == '<':
        filtered_data = [record for record in data if record[attribute_name] < split_point]
    elif sign == '>=':
        filtered_data = [record for record in data if record[attribute_name] >= split_point]
    return filtered_data

def iTree(X,e,l):
    if e>=l or np.size(X)<=1:
        return exNode(np.size(X))
    q_name = randomly_select_attribute(X)
    p = randomly_select_split_point(X,q_name)
    X_l = filter_data_by_attribute(X, q_name, p, '<')
    X_r = filter_data_by_attribute(X, q_name, p, '>=')
    return inNode(left=iTree(X_l,e+1,l), right=iTree(X_r,e+1,l),splitatt=q_name,splitval=p)

def iForest(X,t,phi):
    l = np.ceil(np.log2(phi))
    forest = []
    for i in range(1,t):
        Xi = random.sample(X, phi)
        forest.append(iTree(Xi,0,l))
    return forest

## EVALUATING STAGE

# def h(x,T):
#     # computing path length
#     for tree in T:

#     return PathLength(x,T,0)

def anomaly_score(x,n):
    # compute anomaly score for the datum x
    s = np.power(2,-np.mean(x['PathLength'])/c(n))
    # it is not clear to me
    return s

def open_file(name):
    with open(name, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        attributes = line.strip().split(',')
        # skipping the lines with missing values:
        # if '?' in attributes:
        #     continue
        attributes = [-1 if x == '?' else x for x in attributes]

        # I know that it's wrong to substitute the missing values with -1, I'm thinking about a better handling of this
        record = []
        record = {  'Age': int(attributes[0]),
                    'Sex': int(attributes[1]),
                    'Height': int(attributes[2]),
                    'Weight': int(attributes[3]),
                    'QRS_duration': int(attributes[4]),
                    'P_R_interval': int(attributes[5]),
                    'Q_T_interval': int(attributes[6]),
                    'T_interval': int(attributes[7]),
                    'P_interval': int(attributes[8]),
                    'Vector_angles_QRS': int(attributes[9]),
                    'Vector_angles_T': int(attributes[10]),
                    'Vector_angles_P': int(attributes[11]),
                    'Vector_angles_QRST': int(attributes[12]),
                    'Vector_angles_J': int(attributes[13]),
                    'Heart_rate': int(attributes[14]),
                    'DI_Q_wave_width': int(attributes[15]),
                    'DI_R_wave_width': int(attributes[16]),
                    'DI_S_wave_width': int(attributes[17]),
                    'DI_R_prime_wave_width': int(attributes[18]),
                    'DI_S_prime_wave_width': int(attributes[19]),
                    'DI_num_intrinsic_deflections': int(attributes[20]),
                    'DI_ragged_R_wave': int(attributes[21]),
                    'DI_diphasic_derivation_R_wave': int(attributes[22]),
                    'DI_ragged_P_wave': int(attributes[23]),
                    'DI_diphasic_derivation_P_wave': int(attributes[24]),
                    'DI_ragged_T_wave': int(attributes[25]),
                    'DI_diphasic_derivation_T_wave': int(attributes[26]),
                    'DI_JJ_wave_amplitude': int(attributes[27]),
                    'DI_Q_wave_amplitude': int(attributes[28]),
                    'DI_R_wave_amplitude': int(attributes[29]),
                    'DI_S_wave_amplitude': int(attributes[30]),
                    'DI_R_prime_wave_amplitude': int(attributes[31]),
                    'DI_S_prime_wave_amplitude': int(attributes[32]),
                    'DI_P_wave_amplitude': int(attributes[33]),
                    'DI_T_wave_amplitude': int(attributes[34]),
                    'DI_QRSA': int(attributes[35]),
                    'DI_QRSTA': int(attributes[36]),
                    'PathLength': [],
                    'AnomalyScore': 0}
        # note: I didn't keep all the attributes but a reasonable number of them
        data.append(record)
    return data

# def find_top_anomalies(data, m):
#     sorted_data = sorted(data, key=lambda x: x['AnomalyScore'], reverse=True)
#     top_anomalies = sorted_data[:m]
#     return top_anomalies

def find_top_anomalies(data, m): # returns the indexes
    sorted_indexes = sorted(range(len(data)), key=lambda i: data[i]['AnomalyScore'], reverse=True)
    top_anomaly_indexes = sorted_indexes[:m]
    return top_anomaly_indexes


# def knn(train_data, test_point, k):
#     for train_point in train_data:
#         distance = euclidean_distance(train_point, test_point)
#         train_point['Distanceknn'] = distance
#     distances.sort(key=lambda x: x[1])  # Sort by distance
#     neighbors = distances[:k]
#     return neighbors

# def euclidean_distance(point1, point2):
#     distance = 0
#     for key in point1.keys():
#         if key != 'PathLength' and key != 'AnomalyScore':
#             distance += (point1[key] - point2[key]) ** 2
#     return math.sqrt(distance)

def main():
    data = open_file('dataset_arrhythmia.txt')
    random.seed(2304423) # student number
    split_ratio = 0.8
    data_length = len(data)
    train_size = int(data_length * split_ratio)
    random.shuffle(data)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # TRAINING
    t = 100 # number of trees (ensemble size)
    phi = 256 # sub-sampling size
    forest = iForest(train_data,t,phi)
    
    # EVALUATING
    n = 120 # testing data size

    # when a forest of random trees collectively 
    # produce shorter path lengths for some particular points, then
    # they are highly likely to be anomalies
    for tree in forest:
        for datum in test_data:
            datum['PathLength'].append(PathLength(datum,tree,0))

    for datum in test_data:
        datum['AnomalyScore'] = anomaly_score(datum,n)
        datum['Label']=0

    # find the top m anomalies
    m = int(np.floor(0.15*len(test_data))) # first 15% of data
    top_m = find_top_anomalies(test_data,m)

    print("Number of anomalies detected in the test set: ", m)
    print("Anomaly scores:")

    for index in top_m:
        print(test_data[index]['AnomalyScore'])
        test_data[index]['Label']=1 # classify as anomaly


    # EVALUATION
            
    # Comparison with kNN
    # X_train = [[point[key] for key in point.keys() if (key != 'AnomalyScore' and key != 'PathLength')] for point in train_data]
    # y_train = [1 if point['AnomalyScore'] > 0 else 0 for point in train_data] # I am not sure what to put here
    # X_test = [[point[key] for key in point.keys() if (key != 'AnomalyScore' and key != 'PathLength' and key != 'Label')] for point in test_data]
    # y_test = [1 if point['AnomalyScore'] > 0 else 0 for point in test_data]
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # print(y_pred) # the knn is not correctly trained because the y_train is full of zeros


if __name__=="__main__": 
    main() 
