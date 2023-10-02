import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataSet:
    
    def __init__(self, nome_arquivo):
        self.__dataframe = pd.read_csv(nome_arquivo, delimiter=',', header=0)

    @property
    def dataframe(self):
        return self.__dataframe
    
    def getXy(self):
        X = self.dataframe.iloc[:, :-1]
        y = self.dataframe.iloc[:, -1]        
        return X, y    
    
    # verificando a existência de valores inválidos
    def verificar_valores_invalidos(self):
        return self.dataframe.isnull().sum() / len(self.dataframe) * 100    
  
    # transformando a base de dados após a verificação 
    # da existência de valores inválidos
    def transformar(self):
        percentual_invalidos = self.verificar_valores_invalidos()
        # iterando sobre cada coluna da base de dados
        for coluna in self.dataframe.columns:
            if percentual_invalidos[coluna] > 20:
                # excluindo a coluna se a porcentagem de valores 
                # inválidos for maior que 20%
                self.dataframe.drop(coluna, axis=1, inplace=True)
            elif percentual_invalidos[coluna] > 0:
                # substituindo os valores inválidos pelo 
                # valor imediatamente anterior utilizando o método bfill
                self.dataframe[coluna].fillna(method='bfill', inplace=True)
                
    # normalizando os atributos previsores (atributos de entrada)
    def normalizar(self, X):
        # criando uma instância do MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))        
        # ajustando e transformando com o scaler 
        # os dados dos atributos previsores
        X_norm = scaler.fit_transform(X)
        return scaler, X_norm
    
    # desnormalizando os atributos previsores (atributos de entrada)
    def desnormalizar(self, scaler, X_norm):
        X = scaler.inverse_transform(X_norm)
        colunas = len(X_norm[-1, :])
        X = X.reshape(-1, colunas)
        return X        
        
    def get_train_test_split(self, test_size=0.3, random_state=42):
        X, y = self.getXy()
        _, X_norm = self.normalizar(X)
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test