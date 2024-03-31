import classification_resources
import matplotlib.pyplot as plt
import numpy
import csv

def cross_validation(data_params, algo_params_series, num_rounds=10, num_folds=5):
    dataset, _, _ = classification_resources.load_csv(**data_params)
    num_samples = len(dataset)
    fold_size = num_samples // num_folds
    accuracies = []
    rma = []
    rva = []

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        round_accuracies = []
        shuffled_indices = classification_resources.RandomNumGen.get_gen().permutation(num_samples)
        dataset_shuffled = dataset[shuffled_indices]

        for fold_num in range(num_folds):
            print(f"Fold {fold_num + 1}/{num_folds}")
            test_start = fold_num * fold_size
            test_end = (fold_num + 1) * fold_size
            test_set = dataset_shuffled[test_start:test_end]
            train_set = numpy.concatenate([dataset_shuffled[:test_start], dataset_shuffled[test_end:]])

            for algo_params in algo_params_series:
                model, svi = classification_resources.prepare_svm_model(train_set[:, :-1], train_set[:, -1], **algo_params)
                predictions = classification_resources.svm_predict_vs(test_set[:, :-1], model)
                predicted_labels = 1 * (predictions > 0)
                evals, cm = classification_resources.get_CM_evals(test_set[:, -1], predicted_labels)
                accuracy = evals['accuracy']
                round_accuracies.append(accuracy)
                print("Accuracy:", accuracy)
                print("")
        round_mean_accuracy = numpy.mean(round_accuracies)
        round_variance_accuracy = numpy.var(round_accuracies)
        print(f"Mean Accuracy for Round {round_num + 1}: {round_mean_accuracy}")
        print(f"Variance of Accuracy for Round {round_num + 1}: {round_variance_accuracy}\n")
        accuracies.extend(round_accuracies)
        rma.append(round_mean_accuracy)
        rva.append(round_variance_accuracy)
    overall_mean_accuracy = numpy.mean(rma)
    overall_variance_accuracy = numpy.mean(rva)

    print("Overall Mean Accuracy:", overall_mean_accuracy)
    print("Overall Variance of Accuracy:", overall_variance_accuracy)

def ada_boost(train_set, test_set, algo_params, num_classifiers, beta = 0.5):
    
    print(f"Beta: {beta}")

    # initialization of the weights
    N = train_set.shape[0]
    weights = numpy.ones(N) / N # w_i = 1/N 
    i = 0
    error_rate = 0.1 # random initialization
    models = []
    alphas = numpy.zeros(num_classifiers)
    alpha = 0

    # training of the classifiers
    print("Training:")
    while i < num_classifiers and error_rate != 0:
        
        print(f"Classifier n.{i+1}")
        
        # resample training set based on instance weights
        sampled_indices = numpy.random.choice(range(N), size=N, replace=True, p=weights)
        sampled_train_set = train_set[sampled_indices]

        # train model M(t) on D weighted by w(t)
        model, _ = classification_resources.prepare_svm_model(sampled_train_set[:, :-1], sampled_train_set[:, -1], **algo_params)

        # predictions & error rate
        predictions = classification_resources.svm_predict_vs(train_set[:, :-1], model)
        result = numpy.where(numpy.sign(predictions) >= 0, 1, 0)
        errors = (result != train_set[:, -1]).astype(int)
        error_rate = numpy.sum(errors) / N
        
        print(f"Error rate: {error_rate}")

        # updating weights
        if (beta != 0):
            alpha = beta * numpy.log((1 - error_rate) / error_rate)
            segno = -numpy.sign(errors-0.5) # sign of the exponential
            weights *= numpy.exp(alpha * segno)
            weights = weights / numpy.sum(weights) # normalization of the weights

        if beta==0:
            alpha = 1/num_classifiers

        # storing each classifier with the corresponding weight
        models.append(model)
        alphas[i] = alpha

        if error_rate > 0.5:
            weights = numpy.ones(N)/N

        i += 1

    print("Testing:")
    # final adaboost model = a weighted sum of the weak learners:
    # 'models' contains all the learners and 'weights' contains the weight of the prediction for each learner
    # let's make a prediction on the TEST SET with each model
    pred = numpy.zeros((num_classifiers, test_set.shape[0]))
    predictions = numpy.zeros(test_set.shape[0])

    for i in range(len(models)):
        pred[i] = (classification_resources.svm_predict_vs(test_set[:, :-1], models[i]))
        pred[i] = (pred[i]+1)/2
        pred[i] = 1*((pred[i]>=0)*alphas[i])
        # collect the weighted sum of these predictions
        predictions = predictions + pred[i]

    predictions = numpy.sign(predictions)

    evals, cm = classification_resources.get_CM_evals(test_set[:, -1], predictions)
    print(evals)
    print(cm)

    return

def class_distr(dataset):
    class_counts = {0: 0, 1: 0}
    for data_point in dataset:
        class_label = data_point[-1]
        class_counts[class_label] += 1
    print("Class Counts:")
    for class_label, count in class_counts.items():
        print(f"{class_label}: {count}")
    colors = ['red', 'blue']
    plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(list(class_counts.keys()), ['0', '1'])
    plt.show()

def bagging_training(train_set, algo_params_series, num_classifiers):
    models = []

    for i in range(num_classifiers):
        # create a random subset of the data with replacement (bootstrap)
        indices = numpy.random.choice(train_set.shape[0], train_set.shape[0], replace=True)
        X = train_set[indices]
        model, _ = classification_resources.prepare_svm_model(X[:, :-1], X[:, -1], **algo_params_series)
        models.append(model)

    return models
    
def bagging(test_set, models):

    predicted = [classification_resources.svm_predict_vs(test_set[:, :-1], model) for model in models]

    # use majority voting for classification
    bagged_predictions = numpy.mean(predicted, axis = 0)
    bagged_predictions = numpy.where(numpy.sign(bagged_predictions) >= 0, 1, 0)
    accuracy = numpy.mean(bagged_predictions == test_set[:, -1])
    evals, cm = classification_resources.get_CM_evals(test_set[:, -1], bagged_predictions)

    return bagged_predictions, accuracy, evals, cm

def balance_dataset(dataset, minority_class, majority_class): # oversampling method
    minority_indices = numpy.where(dataset[:, -1] == minority_class)[0]
    majority_indices = numpy.where(dataset[:, -1] == majority_class)[0]
    minority_size = len(minority_indices)
    majority_size = len(majority_indices)
    oversample_ratio = int(majority_size / minority_size)
    oversampled_indices = numpy.random.choice(minority_indices, size=minority_size * oversample_ratio, replace=True)
    oversampled_dataset = numpy.concatenate((dataset, dataset[oversampled_indices]))
    return oversampled_dataset
    
def main(): 
    SEED = 2304423
    classification_resources.RandomNumGen.set_seed(SEED)
    ratio_train = .8
    classification_resources.cvxopt.solvers.options['show_progress'] = False

    print("Task 1:")
    runs = [("iris-sepal", {"filename": "iris-SV-sepal.csv", "last_column_str": True}, [{"c": 0, "ktype": "linear", "kparams": {}}]),
         ("iris-length", {"filename": "iris-VV-length.csv", "last_column_str": True}, [{"c": 2, "ktype": "linear", "kparams": {}}])]
    for series, data_params, algo_params_series in runs:
        classification_resources.run_holdout_experiment_series(data_params, algo_params_series, ratio_train)

    print("Task 2:")
    series = "iris-length"
    data_params = {"filename": "iris-VV-length.csv", "last_column_str": True}
    algo_params_series = [{"c": 2, "ktype": "linear", "kparams": {}}]
    cross_validation(data_params, algo_params_series)

    print("Task 3:")
    series = "creditDE"
    data_params = {"filename": "creditDE.csv", "last_column_str": False}
    algo_params_series = {"c": 1, "ktype": "linear", "kparams": {}}
    num_classifiers = 20
    betas = [1, 0.8, 0.5, 0.2, 0]
    dataset, head, classes = classification_resources.load_csv(**data_params)
    class_distr(dataset)
    dataset = balance_dataset(dataset, minority_class=1, majority_class=0)
    class_distr(dataset)
    ids = classification_resources.RandomNumGen.get_gen().permutation(dataset.shape[0])
    split_pos = int(len(ids)*ratio_train)
    train_ids, test_ids = ids[:split_pos], ids[split_pos:]
    train_set = dataset[train_ids]
    test_set = dataset[test_ids]
    for beta in betas:
        ada_boost(train_set, test_set, algo_params_series, num_classifiers, beta) # predicted values using adaboost
        
    print("Task 4:")
    series = "creditDE"
    data_params = {"filename": "creditDE.csv", "last_column_str": False}
    algo_params_series = {"c": 1, "ktype": "linear", "kparams": {}}
    print("Without kernel:")
    dataset, _, _ = classification_resources.load_csv(**data_params)
    dataset = balance_dataset(dataset, minority_class=1, majority_class=0)
    class_distr(dataset)
    ids = classification_resources.RandomNumGen.get_gen().permutation(dataset.shape[0])
    split_pos = int(len(ids)*ratio_train)
    train_ids, test_ids = ids[:split_pos], ids[split_pos:]
    train_set = dataset[train_ids]
    test_set = dataset[test_ids]
    num_classifiers = 15
    models = bagging_training(train_set, algo_params_series, num_classifiers)
    predictions, accuracy, evals, cm = bagging(test_set, models)
    print(f"Accuracy: {accuracy}")
    print(evals)
    print(cm)
    ada_boost(train_set, test_set, algo_params_series, num_classifiers, 0.5) # predicted values using adaboost
    algo_params_series = {"c": 1, "ktype": "RBF", "kparams": {"sigma": 1.}}
    print("RBF kernel:")
    dataset, _, _ = classification_resources.load_csv(**data_params)
    dataset = balance_dataset(dataset, minority_class=1, majority_class=0)
    class_distr(dataset)
    ids = classification_resources.RandomNumGen.get_gen().permutation(dataset.shape[0])
    split_pos = int(len(ids)*ratio_train)
    train_ids, test_ids = ids[:split_pos], ids[split_pos:]
    train_set = dataset[train_ids]
    test_set = dataset[test_ids]
    num_classifiers = 15
    models = bagging_training(train_set, algo_params_series, num_classifiers)
    predictions, accuracy, evals, cm = bagging(test_set, models)
    print(f"Accuracy: {accuracy}")
    print(evals)
    print(cm)
    ada_boost(train_set, test_set, algo_params_series, num_classifiers, 0.5) # predicted values using adaboost

if __name__=="__main__": 
    main() 