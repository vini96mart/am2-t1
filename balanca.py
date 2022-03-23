import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier

# Pré-processamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Métricas
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Nosso classificador
modelo = StackingClassifier([
            ('dtc', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier()),
            ('svm', SVC())
], RidgeClassifier())

# Classificador para comparação
baseline = AdaBoostClassifier()

# Ler o dataset
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data
dados = pd.read_csv("balance-scale.data", sep=',', names=['bal_class', 'l_weight', 'l_dist', 'r_weight', 'r_dist'])

# Mostra as 5 primeiras linhas
# dados.head(5)


# Scatterplot dos atributos multiplicados
dados = dados.eval('MultR = r_weight * r_dist')
dados = dados.eval('MultL = l_weight * l_dist')
dados['Color'] = dados['bal_class'].map({'B': 'Black', 'L': 'Red', 'R': 'Blue'})
# ax = dados.plot.scatter(x="MultR", y="MultL", c="Color")


# Treino e teste das variáveis
bal_data = dados[['l_weight', 'l_dist', 'r_weight', 'r_dist']]
bal_label = dados['bal_class']

X_train, X_test, Y_train, Y_test = train_test_split(bal_data, 
                                                    bal_label, 
                                                    test_size=0.3,
                                                    stratify=bal_label)

# Tamanho dos dados de treino e teste
print("Amostras de treino:", len(X_train))
print("Amostras de teste:", len(X_test))

# Quantidade de valores para cada classe no teste realizado
Y_test.value_counts()

# One Hot Encoding do dataset categórico
encoder = OneHotEncoder().fit(X_train) 
X_train_scaler = pd.DataFrame(encoder.transform(X_train).toarray())


# Otimização de hiperparâmetros do AdaBoost
param_grid_baseline = {
    'n_estimators': [100, 250, 500, 1000],
    'learning_rate': [0.01, 0.1, 1, 10]
}

gs_baseline_dataset3 = GridSearchCV(baseline, param_grid_baseline)
gs_baseline_dataset3.fit(X_train_scaler, Y_train)
gs_baseline_dataset3.best_estimator_

# Uso do Baseline
classificador_baseline = AdaBoostClassifier(learning_rate=1, n_estimators=1000)
classificador_baseline.fit(X_train_scaler, Y_train)
Y_pred = classificador_baseline.predict(pd.DataFrame(encoder.transform(X_test).toarray()))

print('MÉTRICAS - BASELINE')
print(classification_report(Y_test, Y_pred))
acuracia = accuracy_score(Y_test, Y_pred)*100
print("Acurácia do classificador baseline: {:.2f}%".format(acuracia))


# Encontrar os melhores hiperparâmetros para o nosso modelo
param_grid_modelo = {
    'dtc__criterion': ['entropy', 'gini'],
    'knn__n_neighbors': [1, 3, 5],
    'knn__weights': ['uniform', 'distance'],
    'svm__C': [0.01, 0.1, 1.0, 10]
}

gs_modelo_dataset3 = GridSearchCV(modelo, param_grid_modelo)
gs_modelo_dataset3.fit(X_train_scaler, Y_train)
gs_modelo_dataset3.best_estimator_

#Classificando pelo nosso modelo
classificador = StackingClassifier(estimators=[('dtc',
                                DecisionTreeClassifier(criterion='entropy')),
                               ('knn', KNeighborsClassifier(n_neighbors=1)),
                               ('svm', SVC(C=0.1))],
                   final_estimator=RidgeClassifier())
classificador.fit(X_train_scaler, Y_train)
Y_pred = classificador.predict(pd.DataFrame(encoder.transform(X_test).toarray()))

print('MÉTRICAS - MODELO IMPLEMENTADO')
print(classification_report(Y_test, Y_pred, zero_division=0))
acuracia = accuracy_score(Y_test, Y_pred)*100
print("Acurácia do nosso modelo: {:.2f}%\n".format(acuracia))
