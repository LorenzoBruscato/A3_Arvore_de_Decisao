import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Dados de Treinamento
data = {
    "NivelSatisfacao": [ "Baixo", "Baixo", "Baixo","Alto", "Alto", "Alto", "Baixo", "Alto"],
    "NumeroProjetos": [  "Muitos", "Muitos", "Muitos","Poucos", "Poucos", "Muitos","Poucos", "Muitos"],
    "Saiu": [  "Sim", "Não", "Sim", "Não", "Não", "Não","Sim","Não"],
    "Salario": ["Forte", "Fraco", "Forte","Fraco", "Forte", "Fraco","Fraco", "Forte"],
}
df = pd.DataFrame(data)

# 2. Pré-processamento (Codificação Categórica)
le_sat = LabelEncoder()
le_proj = LabelEncoder()
le_sal = LabelEncoder()
le_target = LabelEncoder()

df["NivelSatisfacao"] = le_sat.fit_transform(df["NivelSatisfacao"])
df["NumeroProjetos"] = le_proj.fit_transform(df["NumeroProjetos"])
df["Salario"] = le_sal.fit_transform(df["Salario"])
df["Saiu"] = le_target.fit_transform(df["Saiu"])

X = df[["NivelSatisfacao", "NumeroProjetos", "Salario"]]
y = df["Saiu"]

# 3. Treinamento do Modelo (Critério ID3: Entropia)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# --- SEÇÃO DE TESTE (SAÍDAS) ---
print("--- Simulador de Decisão ---")
# Exemplo: Funcionário com Satisfação 'Baixa' (0) e 'Muitos' projetos (1)
nova_entrada = [[0, 0, 0]]
predicao = clf.predict(nova_entrada)
resultado = le_target.inverse_transform(predicao)

print(f"Resultado para Satisfação Baixa e Muitos Projetos: {resultado[0]}")

# 4. Visualização

plt.figure(figsize=(12,8))

plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Fica", "Sai"],
     filled=True,
    rounded=True,
    fontsize=10
)
plt.show()