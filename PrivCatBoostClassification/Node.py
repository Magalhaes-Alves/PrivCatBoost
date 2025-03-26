import numpy as np
from gbdt_dp_regression import Gain
from gbdt_dp_regression.Privacy import exponential, laplace_mech
from utilities import *
from gbdt_dp_regression.Gain import Gain
from math import fabs


class Node:

    def __init__(self,xi,yi,hyperparam,parent=None,gain_function=Gain.gainContinuous,split_function = Gain.splitContinuous):

        self.parent = parent
        self._hyperparameters = hyperparam
        self.nodeValue = None
        self.nodeFeature =None
        self.classification =None
        self.is_cat = False

        self.xi =self.removeUniformColumns(xi)
        self.yi =yi
        self.children={}
        self.gain_function = gain_function
        self.split_function = split_function
        
        
    @property
    def hyperparameters(self):
        return self._hyperparameters


    def calcLeafValue(self,gradient,tree,e_leaf,v,learning_rate):

        #Calcula o valor do noh folha baseado na equação 4 
        v_t = -1* (np.sum(gradient)/(gradient.shape[0]+ self.hyperparameters.regularization))

        #Geometric Leaf Cliping
        if v_t != 0.0:
            eval =(self.hyperparameters.gl*(1-self.hyperparameters.learningRate)**(tree))/fabs(v_t)
            v_t = v_t* min(1,eval)
        
        if gradient.shape[0] ==0:
            print("============VAZIO============")
        
        
        noise= np.random.laplace(0,v/e_leaf)
        v_t+=noise
        return learning_rate*v_t


    def removeUniformColumns(self,dataframe):
        
        if dataframe.shape[0]<=1:
            return dataframe
        
        contagem = dataframe.nunique()
        contagem = contagem[contagem<=1].index
        return dataframe.drop(columns=contagem)
    

    def fit(self,e_leaf,e_enleaf,tree,depth=0):
        
        # Acessando os hiperparametros
        g = self.hyperparameters.sensitivityG
        v = self.hyperparameters.sensitivityV
        min_sample_size = self.hyperparameters.minSampleSize
        max_depth =self.hyperparameters.maxDepth
        learning_rate = self.hyperparameters.learningRate
        
        xi = self.xi
        yi = self.yi
       
        if depth== max_depth or xi.shape[0]<min_sample_size:
            
            self.classification = self.calcLeafValue(yi,tree,e_leaf,v,learning_rate)
            return depth
                
        key =exponential(xi,
                         yi,
                         xi.columns.array,
                         self.gain_function,
                         g,
                         e_enleaf,
                         self.hyperparameters)
        
        ganho,feature,split = key

        # Particiona os filhos dependendo do tipo de dado
        
        if ganho<=0:
            self.classification = self.calcLeafValue(yi,tree,e_leaf,v,learning_rate)
            return depth
        
        childs = self.split_function(xi,yi,feature,split,feature in self.hyperparameters.categoricalFeatures)
        
        for k,i in childs.items():
            if i[0].empty:
                self.classification = self.calcLeafValue(yi,tree,e_leaf,v,learning_rate)
                return depth


        self.nodeFeature = feature

        self.nodeValue = split
        
        self.is_cat = self.nodeFeature in self.hyperparameters.categoricalFeatures
        
        for k,v in childs.items():
            self.addChildren(k,v[0],v[1])
            
        max_depth_child =0
        for key,child in self.children.items():
            
            max_depth_child= max(max_depth_child,child.fit(e_leaf,e_enleaf,tree,depth+1))

        return max_depth_child
    
    def addChildren(self,key,xi,yi):
        
        self.children[key]=Node(xi=xi,yi=yi,hyperparam=self.hyperparameters,parent=self,gain_function=self.gain_function,split_function=self.split_function)