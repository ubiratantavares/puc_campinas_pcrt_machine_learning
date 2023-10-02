from dataset import DataSet
from random_forest import RandomForest

# criando um objeto da classe DataSet 
data = DataSet("water_potability.csv")

# realizando a leitura da base de dados com o objeto data
df = data.dataframe

# visualizando as 5 primeiras instâncias 
# dos atributos da base de dados
print(df.head())

# verificando a quantidade de 
# linhas e colunas da base de dados
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

# treinando o modelo com os dados de treinamento
modelo_rf_treinado = rf.treinar(modelo_rf, X_train, y_train)

# utilizando o modelo treinado para prever o 
# atributo alvo a partir dos dados de teste
y_pred = rf.prever(modelo_rf_treinado, X_test)

# avaliando a performance do modelo classificador a 
# partir dos dados de teste e o resultado das previsões
avaliacao_classificador = rf.avaliar(y_test, y_pred, ['0', '1'])

# exibindo avaliação do classificador
rf.imprimir_avaliacao(avaliacao_classificador)

# plotando a avaliação do classificador
rf.plotar_avaliacao(modelo_rf, X_train, X_test, y_train, y_test)
