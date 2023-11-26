import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Separa atributos preditivos e atributos-alvo em um ndarray (Numpy)
def split_pred_target_feats(dataset):
    target_feat_pos = dataset.shape[1] - 1
    pred_data = dataset[:,:target_feat_pos]
    target_data = dataset[:,target_feat_pos]
    return pred_data, target_data

# Separa atributos preditivos de atributos-alvo e normaliza seus valores
def split_standardize_dataset(dataset):
    scaler = StandardScaler()
    pred_data, target_data = split_pred_target_feats(dataset)
    std_pred_data = scaler.fit_transform(pred_data)
    std_target_data = target_data.astype(int)
    return std_pred_data, std_target_data

# Calcula a média das métricas de performance para o k conjuntos de treino/teste
def mean_scores(scores):
    return np.mean(scores['test_accuracy']), np.mean(scores['test_f1'])

# Treina e testa algoritmos e obtém métricas de sua perfomance, utilizando k-fold cross validation
def get_algorithm_performance_metrics(classifier, x, y, cv):
    scores = cross_validate(estimator=classifier, X=x, y=y, scoring=('accuracy','f1'), cv=cv)
    accuracy, f1 = mean_scores(scores)
    return accuracy, f1

raw_meta_database = np.load("./meta_database.npy")
finite_values_meta_database = np.nan_to_num(raw_meta_database)
raw_x, y = split_standardize_dataset(finite_values_meta_database)
finite_values_x = np.nan_to_num(raw_x)

classifier_1 = RandomForestClassifier()
classifier_2 = GradientBoostingClassifier()
classifier_3 = MLPClassifier()

accuracy_1, f1_1 = get_algorithm_performance_metrics(classifier_1, finite_values_x, y, 10)
accuracy_2, f1_2 = get_algorithm_performance_metrics(classifier_2, finite_values_x, y, 10)
accuracy_3, f1_3 = get_algorithm_performance_metrics(classifier_3, finite_values_x, y, 10)

print(f"Acurácia 1: {accuracy_1}\nAcurácia 2: {accuracy_2}\nAcurácia 3: {accuracy_3}\nF1 1= {f1_1}\nF1 2= {f1_2}\nF1 3= {f1_3}")