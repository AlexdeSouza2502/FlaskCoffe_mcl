import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Substitua 'seu_arquivo_cafeina.csv' pelo nome real do seu arquivo
cafeina_data = pd.read_csv('arquivos-csv\caffeine.csv')

# Certifique-se de incluir todas as features necessárias
X = cafeina_data[['Volume (ml)', 'Calories', 'Caffeine (mg)']]
y = cafeina_data['type']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)

# Certifique-se de fornecer os nomes das características ao treinar o modelo
clf = DecisionTreeClassifier()
clf = clf.fit(X_treino, y_treino, feature_names=X.columns)

preditos = clf.predict(X_teste)
print("Acuracia:", accuracy_score(y_teste, preditos))

# Salve o modelo treinado
with open('caffeine_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
