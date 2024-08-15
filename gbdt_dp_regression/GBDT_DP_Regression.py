from gbdt_dp_regression.HyperParameters import HyperParameters
import pandas as pd
import numpy as np
import math
from gbdt_dp_regression.Tree import Tree
from sklearn.metrics import mean_squared_error
from sklearn.base import RegressorMixin,BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from gbdt_dp_regression.Gain import Gain


class GBDT_DP_Regression(RegressorMixin,BaseEstimator):


    def __init__(self, 
                 max_depth, 
                 privacy_budget, 
                 regularization,
                 total_trees,
                 trees_in_ensemble,
                 learning_rate,
                 categorical_features=[],
                 type_gain=0,
                 min_node_support=2,
                 type_tree = 0):
        
        """
        max_depth -> Maximum depth allowed for each decision tree in the ensemble.
        privacy_budget -> Privacy budget for the GBDT.
        regularization -> Total number of decision trees that will be created.
        trees_in_ensemble -> Define the number of trees in an ensemble.
        learning_rate -> It controls how much the model's weights are updated in each iteration during the training process.
        categorical_features -> Containing the names or indices of the features in the data that are categorical.
        min_node_support -> Specifies the minimum number of samples that must be present at a node
        type_gain -> Define the gain function used to calculate the information gain of internal nodes.
        type_tree -> Define the type of tree create by model.

        """

        self.gl = 1
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.categorical_features=categorical_features
        self.min_node_support =min_node_support
        self.type_gain = type_gain
        
        self.data =None
        self.target =None
        self.privacy_budget = privacy_budget
        self.type_tree =type_tree

        self.hyperparameters = HyperParameters(regularization=regularization,
                                               learning_rate=learning_rate,
                                               max_depth=max_depth,
                                               min_node_support=min_node_support,
                                               type_gain=type_gain,
                                               type_tree = type_tree,
                                               )
        self.total_trees = total_trees
        self.trees_in_ensemble = trees_in_ensemble

        #Definição estrutura para os gradientes

        self.forest =[]
        
        self._normalize=None
        self.categorical_features=categorical_features
        self._order =None
                
    def preprocessing(self,data,target):

        #Normalize data
        self.data = data.copy().reset_index().drop(['index'],axis=1)
        self._normalize = MinMaxScaler((-1,1))
        self.target = pd.DataFrame(self._normalize.fit_transform(target.values).reshape(-1,1),index=self.data.index)
        self.gradient = self.target.apply(lambda x:-1*x)
        self.prediction = pd.DataFrame(np.zeros(data.shape[0]))

        self.hyperparameters.categoricalFeatures =[self.data.columns.get_loc(i)for i in self.categorical_features]
        self.data.columns=(range(data.shape[1]))

        #Ordenar todas as features por indices

        order = np.zeros((data.shape[1],data.shape[0]))

        data = self.data.copy()
        #time = getTime()
        for i,feat in enumerate(data.columns):
            data=data.sort_values(by=feat,axis=0)
            order[i] =data.index

        #print(f"Tempo de Ordenação: {getTime()-time}")
        self._order= order


    def fit(self,data,target):
        
        self.preprocessing(data,target)
        
        N_ensembles=math.ceil(self.total_trees/self.trees_in_ensemble)
        ensemble_budget = self.privacy_budget/N_ensembles
        gl =self.hyperparameters.gl
        learning_rate = self.hyperparameters.learningRate
        regularization = self.hyperparameters.regularization
        
        for tree in range(1,self.total_trees+1):
            
            #now = getTime()
            print(f"=========Treino Árvore {tree}=========")

            #Update Gradient
            if tree>1:                
                self.gradient =-1*(self.target- self.prediction)
                
            
            #Verify in which ensemble the tree be.
            t_e = tree % self.trees_in_ensemble

            #Caso seja a primeira árvore então é utilizado todo o dataset para o treinamento
            if t_e == 1:
                I=np.array(self.data.index)
                np.random.shuffle(I)
                begin_part = 0

                
            #Calcula a quantidade de amostras 
            n_samples  = self.data.shape[0]*learning_rate*((1-learning_rate)**t_e)
            n_samples = round(n_samples/(1-((1-learning_rate)**self.trees_in_ensemble)))

            #Fim do particionamento
            end_part = min(begin_part+n_samples,I.shape[0])

            select_index =I[begin_part:end_part]

            begin_part=end_part

            self.hyperparameters.Order= np.zeros(shape=(self.data.shape[1],select_index.shape[0]))

            for i in range(self.data.shape[1]):

                index =np.isin(self._order[i],select_index)
                self.hyperparameters.Order[i] = self._order[i][index]


            sub_xi = self.data.loc[select_index]
            sub_grad = self.gradient.loc[select_index]

            new_tree = Tree(sub_xi,
                            sub_grad.to_numpy(),
                            self.hyperparameters,
                            ensemble_budget,
                            3*(gl**2),
                            min(gl/(1+regularization), 2*gl*((1-learning_rate)**(tree-1))),
                            tree
                            )
            
            self.forest.append(new_tree)
            new_tree.fit()

            predict =new_tree.predict(self.data)
            
            self.prediction+= pd.DataFrame(predict)

            #print(f"Treinamento da arvore {tree} levou {getTime()-now}")

            #print("MSE:", mean_squared_error(self._normalize.inverse_transform([self.target]),self._normalize.inverse_transform([self.prediction])))
            #print("RMSE", mean_squared_error(self._normalize.inverse_transform([self.target]),self._normalize.inverse_transform([self.prediction]),squared=False))

        #print('Gradient:\n',self.gradient)
        self.__desaloc_data()

    def __desaloc_data(self):
        del self.data
        del self.target
        del self.gradient
        del self.prediction
        del self._order

    def predict(self,instances):
        
        instances = instances.copy()
        instances.columns = range(instances.shape[1])

        predictons = np.zeros(instances.shape[0])

        if len(self.forest) ==0:
            raise Exception("Não há árvores para realizar a predição.")

        for k,tree in enumerate(self.forest):
                
            #print(f'Iter: {k}')
            
            predictons = predictons+(tree.predict(instances))

        
        return self._normalize.inverse_transform([predictons]).ravel()
        
    def showTree(self,id_tree):
        
        def _traverseTree(raiz,nivel=0):
            print("Nivel do noh:",nivel)
            if raiz.classification is not None:
                print("-------------------------------------------------")
                print("Classificação folha:",raiz.classification)
                
                if raiz.parent is not None:
                    print("Feature do Pai da Folha:",raiz.parent.nodeFeature)
                else:
                    print("Raiz")
                print("-------------------------------------------------")

                return
            
            print("-------------------------------------------------")
            print("Nivel do noh:",nivel)
            print("Feature do Noh:",raiz.nodeFeature)
            print("Ramos do noh:",raiz.children.keys())
            #print(raiz.target.iloc[:,0].value_counts().to_frame().transpose())
            if raiz.parent is not None:
                print("Feature do Pai:",raiz.parent.nodeFeature)
            
            
            
            if raiz.nodeValue is not None:
                print("Valor noh:",raiz.nodeValue)
            print("-------------------------------------------------")
            
            for k in raiz.children.keys():
                print("Key",k)
                _traverseTree(raiz.children[k],nivel+1)

        _traverseTree(self.forest[id_tree].root)
           