# ÁRVORES DE DECISÃO EM PYTHON (ID3)

## Diagnóstico de Doenças e Previsão de Turnover

**Aluno:** Lorenzo Bruscato  
**Curso:** Ciência da Computação  
**Semestre:** 3º

---

# Sobre o Projeto

Este projeto apresenta duas aplicações utilizando Árvores de Decisão (ID3),
implementadas com a biblioteca scikit-learn e o critério de entropia.

Os projetos demonstram:

- Criação de modelos de classificação;
- Uso do DecisionTreeClassifier;
- Pré-processamento de dados com LabelEncoder;
- Treinamento e teste de modelos preditivos;
- Visualização gráfica das árvores de decisão;
- Simulação de novos cenários.

---

# Projeto 1 - Diagnóstico de Doenças

## Objetivo

Classificar pacientes com base em sintomas para auxiliar no diagnóstico de
doenças como Dengue, Chikungunya e Resfriado.

## Sintomas Utilizados

| Sintoma | Valores |
|----------|----------|
| Febre | Sim / Não |
| Muitos dias de sintomas | Sim / Não |
| Área de risco | Sim / Não |
| Dor atrás dos olhos | Sim / Não |
| Dor de cabeça | Sim / Não |
| Dor muscular | Sim / Não |
| Manchas vermelhas | Sim / Não |
| Cansaço | Sim / Não |
| Náusea | Sim / Não |
| Tosse | Sim / Não |
| Coriza | Sim / Não |

## Diagnósticos Possíveis

- Dengue
- Chikungunya
- Resfriado
- Internar
- Ver médico
- Saudável

---

# Projeto 2 - Previsão de Turnover

## Objetivo

Prever se um funcionário deixará a empresa com base em características
profissionais.

## Variáveis de Entrada

| Atributo | Valores |
|-----------|----------|
| Nível de satisfação | Baixo / Alto |
| Número de projetos | Muitos / Poucos |
| Salário | Forte / Fraco |

## Saída

- Saiu: Sim
- Saiu: Não

---

# Instalação

## Linux

### Criar ambiente virtual

```bash
python3 -m venv venv
```

### Ativar ambiente virtual

```bash
source venv/bin/activate
```

### Instalar dependências

```bash
pip install numpy matplotlib scipy scikit-fuzzy networkx
```

### Mover o arquivo para dentro do ambiente virtual

```bash
mv Logica_de_Fuzzy_Ventilador.py venv/
```

### Entrar na pasta do ambiente virtual

```bash
cd venv
```

### Executar o programa

```bash
python3 Logica_de_Fuzzy_Ventilador.py
```

---

## Windows

### Criar ambiente virtual

```cmd
python -m venv venv
```

### Ativar ambiente virtual

Prompt de Comando:

```cmd
venv\Scripts\activate.bat
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Instalar dependências

```cmd
pip install numpy matplotlib scipy scikit-fuzzy networkx
```

### Mover o arquivo para dentro do ambiente virtual

Prompt de Comando:

```cmd
move Logica_de_Fuzzy_Ventilador.py venv\
```

PowerShell:

```powershell
Move-Item Logica_de_Fuzzy_Ventilador.py venv\
```

### Entrar na pasta do ambiente virtual

```cmd
cd venv
```

### Executar o programa

```cmd
python Logica_de_Fuzzy_Ventilador.py
```


---

# Estrutura do Projeto

A3_Arvore_de_Decisao/

├── Diagnostico_Doencas.py

├── Previsao_Turnover.py

├── README.txt

└── venv/

---

# Bibliotecas Utilizadas

| Biblioteca | Função |
|------------|---------|
| pandas | Manipulação de dados |
| scikit-learn | Árvore de decisão ID3 |
| matplotlib | Visualização gráfica |

---

# Exemplo de Resultado

## Diagnóstico de Doenças

Resultado:

Paciente com febre, muitos dias de sintomas, área de risco e dor atrás dos olhos.

Diagnóstico previsto: Dengue

## Previsão de Turnover

Resultado:

Funcionário com baixa satisfação e muitos projetos.

Previsão: Sim, o funcionário pode deixar a empresa.

---

# Observações

- O projeto utiliza o critério de entropia para construção da árvore.
- Os gráficos são exibidos automaticamente durante a execução.
- Novos cenários podem ser testados alterando a variável
  `nova_entrada` nos arquivos Python.
- Recomenda-se utilizar ambiente virtual para evitar conflitos de dependências.

---

# Referências

• Material disponibilizado pelo professor Saulo Arisa

• Árvores de Decisão em Python
  https://wiki.arisa.com.br/index.php?title=%C3%81rvores_de_Decis%C3%A3o_em_Python
