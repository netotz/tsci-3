import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, tree, model_selection

# cargar dataset de flores
dataset = datasets.load_iris()

# usar pandas para modelar el dataset
dataframe = pd.DataFrame(dataset.data, columns=dataset.feature_names)
dataframe['target'] = dataset.target

# dividir el dataset para entrenar el árbol de decisión y después probarlo
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    dataframe[dataset.feature_names],
    dataframe['target'])

# instanciar nuevo clasificador de árbol de decisión
decision_tree = tree.DecisionTreeClassifier()
# entrenar clasificador
decision_tree = decision_tree.fit(x_train, y_train)
# calcular precisión de predicción
score = decision_tree.score(x_test, y_test)

# imprimir resultados en consola
print('Precisión de predicción:', score)

# graficar árbol de decisión
# plt.figure(figsize=(12, 12))
plt.subplots(1, 1, figsize=(3, 3), dpi=300)
tree.plot_tree(
    decision_tree,
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True)
plt.show()
