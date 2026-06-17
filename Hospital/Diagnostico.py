import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Dados de Treinamento
data = {
    "Febre": ["Sim", "Nao", "Sim", "Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim"],
    "Muitos_Dias": ["Sim", "Nao", "Sim", "Nao", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Nao"],
    "Area_de_risco": ["Sim", "Nao", "Nao", "Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim"],
    "Dor_atras_olhos": ["Sim", "Nao", "Nao", "Nao", "Nao", "Sim", "Nao", "Nao", "Sim", "Sim", "Nao", "Sim"],
    
    # Novos sintomas
    "Dor_cabeca": ["Sim", "Nao", "Sim", "Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim"],
    "Dor_muscular": ["Sim", "Nao", "Nao", "Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim"],
    "Manchas_vermelhas": ["Sim", "Nao", "Nao", "Nao", "Sim", "Nao", "Sim", "Nao", "Nao", "Sim", "Nao", "Nao"],
    "Cansaco": ["Sim", "Nao", "Sim", "Nao", "Sim", "Sim", "Nao", "Sim", "Nao", "Sim", "Nao", "Sim"],
    "Nausea": ["Sim", "Nao", "Nao", "Nao", "Sim", "Sim", "Nao", "Nao", "Sim", "Sim", "Nao", "Nao"],
    "Tosse": ["Nao", "Nao", "Sim", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Sim", "Nao"],
    "Coriza": ["Nao", "Nao", "Sim", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Sim", "Nao"],
    
    "Diagnostico": ["Dengue", "Saudavel", "Resfriado", "Saudavel", "Dengue", "Internar", "Chikungunya", "Saudavel", "Dengue", "Chikungunya", "ver_medico", "Dengue"]
}

df = pd.DataFrame(data)

# 2. Pré-processamento
# Converte "Sim"/"Nao" para 0/1 e os diagnósticos para números
le_febre = LabelEncoder()
le_dias = LabelEncoder()
le_area = LabelEncoder()
le_dor = LabelEncoder()
le_dor_cabeca = LabelEncoder()
le_dor_muscular = LabelEncoder()
le_manchas = LabelEncoder()
le_cansaco = LabelEncoder()
le_nausea = LabelEncoder()
le_tosse = LabelEncoder()
le_coriza = LabelEncoder()
le_target = LabelEncoder()

# Aplica a codificação em cada coluna
df["Febre"] = le_febre.fit_transform(df["Febre"])
df["Muitos_Dias"] = le_dias.fit_transform(df["Muitos_Dias"])
df["Area_de_risco"] = le_area.fit_transform(df["Area_de_risco"])
df["Dor_atras_olhos"] = le_dor.fit_transform(df["Dor_atras_olhos"])
df["Dor_cabeca"] = le_dor_cabeca.fit_transform(df["Dor_cabeca"])
df["Dor_muscular"] = le_dor_muscular.fit_transform(df["Dor_muscular"])
df["Manchas_vermelhas"] = le_manchas.fit_transform(df["Manchas_vermelhas"])
df["Cansaco"] = le_cansaco.fit_transform(df["Cansaco"])
df["Nausea"] = le_nausea.fit_transform(df["Nausea"])
df["Tosse"] = le_tosse.fit_transform(df["Tosse"])
df["Coriza"] = le_coriza.fit_transform(df["Coriza"])
df["Diagnostico"] = le_target.fit_transform(df["Diagnostico"])

# Features e Target
X = df[["Febre", "Muitos_Dias", "Area_de_risco", "Dor_atras_olhos", "Dor_cabeca", "Dor_muscular", "Manchas_vermelhas", "Cansaco", "Nausea", "Tosse", "Coriza"]]
y = df["Diagnostico"]

# 3. Treinamento do Modelo (Critério ID3: Entropia)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# --- SEÇÃO DE TESTE (SAÍDAS) ---
print("--- Simulador de Decisão ---")

# Teste com paciente que tem TODOS os sintomas (0 = Sim)
# Posições: [Febre, Muitos_Dias, Area_risco, Dor_olhos, Dor_cabeca, Dor_muscular, Manchas, Cansaco, Nausea, Tosse, Coriza]nova_entrada = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
nova_entrada = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
predicao = clf.predict(nova_entrada)
resultado = le_target.inverse_transform(predicao)

print(f"Resultado para Febre Sim, Muitos Dias Sim, Area de risco Sim, Dor atras olhos Sim: {resultado[0]}")

# 4. Visualização da Árvore de Decisão
plt.figure(figsize=(12, 8))

plot_tree(
    clf,
    feature_names=X.columns,           # Nomes dos sintomas
    class_names=le_target.classes_,     # Nomes dos diagnósticos
    filled=True,                        # Preenche os nós com cores
    rounded=True,                       # Bordas arredondadas
    fontsize=10                         # Tamanho da fonte
)

plt.title("Árvore de Decisão para Diagnóstico de Doenças", fontsize=14, pad=20)
plt.tight_layout()
plt.show()