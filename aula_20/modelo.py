# Criando a classe abstrata para ser herdado 
# pelos modelos k-Nearest Neighbor e Random Forest

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport

import numpy as np

class Modelo(ABC):    
   
    @abstractmethod
    def criar(self, parametro):
        pass
    
    # treinando o classificador com os dados de treinamento
    def treinar(self, modelo, X_train, y_train):
        modelo_fit = modelo.fit(X_train, y_train)
        return modelo_fit
    
    # aplicando o método de seleção de recursos 
    # com RFE, considerando n recursos
    def aplicar_rfe(self, modelo, X_train, y_train, n_recursos=3):
        rfe = RFE(estimator=modelo, n_features_to_select=n_recursos, step=1)
        X_rfe = rfe.fit_transform(X_train, y_train)
        return rfe, X_rfe
    
    # identificando os recursos 
    # selecionados com o método RFE
    def identificar_recursos_rfe(self, rfe):
        return np.where(rfe.support_)[0]
    
    def exibir_informacoes_rfe(self, rfe):
        recursos = np.where(rfe.support_)[0]
        print(f"Ranking dos atributos: {rfe.ranking_}")
        print(f"\nSuporte dos atributos selecionados: {rfe.support_}")
        print(f"\nAtributos selecionados: {self.identificar_recursos_rfe(rfe)}\n")        

    # aplicando o método de extração de 
    # recursos com PCA, considerando n componentes
    def aplicar_pca(self, X_train, X_test, n_components=3):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return pca, X_train_pca, X_test_pca
    
    # exibindo as informações relevantes do modelo PCA
    def exibir_informacoes_pca(self, pca):
        print(f"Variância explicada por cada componente principal: {pca.explained_variance_ratio_}\n")
        print(f"\nVariância explicada acumulada: {np.cumsum(pca.explained_variance_ratio_)}\n")
    
    # realizando as previsões com os dados de teste
    def prever(self, modelo_treinado, X_test):
        return modelo_treinado.predict(X_test)
    
    # avaliando o desempenho do modelo
    def avaliar(self, y_test, y_pred, classes):
        return classification_report(y_test, y_pred, target_names=classes)
    
    # imprimindo o resultado da avaliação do desempenho do modelo
    def imprimir_avaliacao(self, avaliacao):
        print(avaliacao)
        
    # plotando o resultado da avaliação do desempenho 
    # do modelo com a classe ClassificationReport da biblioteca yellowbrick
    def plotar_avaliacao(self, modelo, X_train, X_test, y_train, y_test):        
        cr = ClassificationReport(modelo, classes=[0, 1], support=True)
        cr.fit(X_train, y_train)
        cr.score(X_test, y_test)
        cr.poof()
        
    # plotando o resultado da acurácia do modelo 
    # de acordo com o número de árvores na floresta.
    def plotar_numero_arvores(self, grid_search):
        resultados = grid_search.cv_results_
        scores = resultados['mean_test_score']
        num_arvores = resultados['param_n_estimators'].data
        plt.figure(figsize=(10, 6))
        plt.plot(num_arvores, scores, marker='o', linestyle='-')
        plt.title('Acurácia em função do número de árvores na Floresta Aleatória')
        plt.xlabel('Número de Árvores na Floresta')
        plt.ylabel('Acurácia Média (Validação Cruzada)')
        plt.grid(True)
        plt.show()
        
    # buscando o numero de arvores com a validação cruzada
    def buscar_melhor_numero_arvores(self, X_train, y_train, num_arvores_range=range(10, 101, 10), cv=10):
        melhor_score = 0
        melhor_numero_arvores = None
        media_scores = []
        for n_arvore in num_arvores_range:
            model = self.criar(n_arvore)
            kfold = KFold(n_splits=cv, shuffle=True, random_state=0)
            scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
            media_score = scores.mean()
            media_scores.append(media_score)
            if media_score > melhor_score:
                melhor_score = media_score
                melhor_numero_arvores = n_arvore
        return media_scores, melhor_numero_arvores
    
    # plotando a acurácia versus o numero de arvores
    def plot_resultados_avaliacao_cruzada(self, scores, num_arvores_range=range(10, 101, 10)):
        plt.figure(figsize=(8, 6))
        plt.plot(num_arvores_range, scores, marker='o', linestyle='-')
        plt.title("Validação Cruzada para Random Forest")
        plt.xlabel("Número de Árvores na Floresta")
        plt.ylabel("Acurácia Média")
        plt.grid(True)
        plt.show()