from dataset import DataSet
from random_forest import RandomForest

# criando um objeto da classe DataSet
data = DataSet("water_potability.csv")

# realizando a leitura da base de dados com o objeto data
df = data.dataframe

# visualizando as 5 primeiras instâncias dos atributos da base de dados
print(df.head())

# verificando a quantidade de linhas e colunas da base de dados
print(df.shape)

# verificando a existência de valores inválidos na base de dados
percentual_invalidos = data.verificar_valores_invalidos()

# exibindo o percentual de valores inválidos na base de dados
print(percentual_invalidos)

# transformando a base de dados após a verificação
# da existência de valores inválidos
data.transformar()

# realizando a leitura da base de dados com o objeto data
df = data.dataframe

# visualizando as 5 primeiras instâncias dos atributos da base de dados
print(df.head())

# verificando a quantidade de instâncias e atributos da base de dados
print(df.shape)

# verificando a existência de valores inválidos na base de dados
percentual_invalidos = data.verificar_valores_invalidos()

# exibindo o percentual de valores inválidos na base de dados
print(percentual_invalidos)

# verificando a quantidade de cada classe do atributo alvo
classes = df['Potability'].value_counts()
print(classes)

# criando os atributos previsores e o atributo alvo
X, y = data.getXy()

print(X.shape, y.shape)

print(X.iloc[0:5, :])

# criando o objeto scaler e os atributos previsores normalizado
scaler, X_norm = data.normalizar(X)

# visualizando as 5 primeiras linhas
print(X_norm[0:5, :])

# dividindo a base de dados em dados de treinamento e teste
X_train, X_test, y_train, y_test = data.get_train_test_split()

# verificando a quantidade de cada classe do atributo alvo
print(y_train.value_counts())
print(f'{(y_train.value_counts()[0]/y_train.shape[0]) * 100:.1f}')
print(f'{(y_train.value_counts()[1]/y_train.shape[0]) * 100:.1f}')

print(y_test.value_counts())
print(f'{(y_test.value_counts()[0]/y_test.shape[0]) * 100:.1f}')
print(f'{(y_test.value_counts()[1]/y_test.shape[0]) * 100:.1f}')

# verificando as dimensões dos arrays
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# criando uma instância da classe RandomForest
rf = RandomForest()

# criando o modelo Random Forest
modelo_rf = rf.criar(10)

# selecionando os atributos com o método RFE
rfe, X_train_rfe = rf.aplicar_rfe(modelo_rf, X_train, y_train)

# verificando a quantidade de linhas e colunas após aplicar do método de seleção de features RFE
print(X_train_rfe.shape)

# exibindo os atributos selecionados com o método RFE
rf.exibir_informacoes_rfe(rfe)

# treinando o modelo após aplicar do método de seleção de features RFE
modelo_rf_treinado_rfe = rf.treinar(modelo_rf, X_train_rfe, y_train)

# identificando os atributos selecionados com o método RFE
atributos_selecionados = rf.identificar_recursos_rfe(rfe)

# selecionando os atributos selecionados na base de dados de teste
X_test_rfe = X_test[:, atributos_selecionados]

# verificando a quantidade de linhas e colunas
print(X_test_rfe.shape)

# realizando previsões com os dados de teste após aplicar o RFE
y_pred_rfe = rf.prever(modelo_rf_treinado_rfe, X_test_rfe)

# avaliando o desempenho do modelo após aplicar o RFE
avaliacao_classificador_rfe = rf.avaliar(y_test, y_pred_rfe, ['0', '1'])
rf.imprimir_avaliacao(avaliacao_classificador_rfe)
rf.plotar_avaliacao(modelo_rf, X_train_rfe, X_test_rfe, y_train, y_test)
