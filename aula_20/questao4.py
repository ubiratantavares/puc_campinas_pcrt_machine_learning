
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

# aplicando PCA com 3 componentes principais
pca, X_train_pca, X_test_pca = rf.aplicar_pca(X_train, X_test)

#  visualizando as informações relevantes do modelo PCA
rf.exibir_informacoes_pca(pca)

# buscando o melhor número de árvores por meio da validação cruzada
media_scores, melhor_n_arvores = rf.buscar_melhor_numero_arvores(X_train, y_train)
print(f"Melhor número de árvores encontrado: {melhor_n_arvores}")

rf.plot_resultados_avaliacao_cruzada(media_scores)

# criando uma instância da classe RandomForest
rf = RandomForest()

# criando o modelo Random Forest
modelo_rf = rf.criar(melhor_n_arvores)

# aplicando PCA com 3 componentes principais
pca, X_train_pca, X_test_pca = rf.aplicar_pca(X_train, X_test)

#  visualizando as informações relevantes do modelo PCA
rf.exibir_informacoes_pca(pca)

# treinando o modelo Random Forest com os dados de treinamento após o PCA
modelo_rf_treinado_pca = rf.treinar(modelo_rf, X_train_pca, y_train)

# realizando previsões com os dados de teste após o PCA
y_pred_pca = rf.prever(modelo_rf_treinado_pca, X_test_pca)

# avaliando o desempenho do modelo após o PCA
avaliacao_classificador_pca = rf.avaliar(y_test, y_pred_pca, ['0', '1'])
rf.imprimir_avaliacao(avaliacao_classificador_pca)
rf.plotar_avaliacao(modelo_rf, X_train_pca, X_test_pca, y_train, y_test)
