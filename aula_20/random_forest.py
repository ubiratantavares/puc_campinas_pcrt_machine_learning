# Criando a classe para o modelo Random Forest

from modelo import Modelo

from sklearn.ensemble import RandomForestClassifier

class RandomForest(Modelo):
    
    # criando o modelo do classificador 
    # Ranfom Forest com 10 árvores como default
    def criar(self, parametro):
        return RandomForestClassifier(n_estimators=parametro, random_state=0) 