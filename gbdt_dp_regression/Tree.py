import pandas as pd
import numpy as np
from gbdt_dp_regression.Node import Node
from gbdt_dp_regression.Gain import Gain

from utilities import *

class Tree:

    def __init__(self,
                 xi,
                 yi,
                 hyperparam,
                 privacy_budget,
                 sensitivite_g,
                 sensitivite_v,
                 t):
        
        self.data = xi
        self.target = yi
        self.hyperparameters = hyperparam
        self.privacy_budget = privacy_budget
        self.t = t
        self.depth =[0]

        self.hyperparameters.sensitivityG= sensitivite_g
        self.hyperparameters.sensitivityV = sensitivite_v
        self.root =None
     

    def fit(self):
        e_leaf = self.privacy_budget
        e_nleaf=0
        
        if self.hyperparameters.typeTree==0:
            e_leaf /=2
            e_nleaf = self.privacy_budget/(2*self.hyperparameters.maxDepth)

        #print(f"""Privacy Budget total { self.privacy_budget}""")
        #print(f"""Privacy Budget Noh Internos { e_nleaf}""")
        #print(f"""Privacy Budget Folhas { e_leaf}""")
        
        mask = np.abs(self.target) <= self.hyperparameters.gl

        #print("R",np.sum(mask))
                
        xi = self.data.loc[mask]
        yi = self.target[mask]        

        """ print("Threshold", self.hyperparameters.gl)
        print("Data PRE-GDF", self.data.shape)
        print("Max Value ",np.max(np.abs(self.target)))
        print("Data POS-GDF", xi.shape)
        print("Max Value ",np.max(np.abs(yi))) """
                
        gain_function,split_function = Gain.getGain(self.hyperparameters.typeGain)

        self.root = Node(xi=xi,yi=yi,hyperparam=self.hyperparameters,gain_function=gain_function,split_function=split_function)

        self.depth =self.root.fit(e_leaf,e_nleaf,self.t)

        #print(f"Arvore tem profundidade {self.hyperparameters.maxDepth -self.depth[0]}.")
        self.__desaloc_data()

    def __desaloc_data(self):

        del self.xi
        del self.yi
    
    def _predictInstances(self,instance):


        node = self.root

        while node.children:

            feature = node.nodeFeature

            value = instance[feature]

            split_value = node.nodeValue
            
            if node.is_cat: # É categorico e é feito analise por categoria?
                
                node =node.children.get(value,None)
                
                if node is None:
                    return 0
                
            else:
                if value>= split_value:

                    node = node.children[f'>={split_value}']
                else:
                    node = node.children[f'<{split_value}']

        return node.classification
    
    def predict(self,data):
        

        predictions =[]

        for _, instance in data.iterrows():

            predictions.append(self._predictInstances(instance))
            
        return np.array(predictions)
            




