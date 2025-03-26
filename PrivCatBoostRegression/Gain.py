import pandas as pd
import numpy as np
import numba
from numba import jit

class Gain:
    
    CONTINUOUS_ONLY=0
    CONTINUOUS_AND_CATEGORICAL=1


    @staticmethod
    def getGain(type_gain):
        match type_gain:
            case Gain.CONTINUOUS_ONLY:
                return [Gain.gainContinuous,Gain.splitContinuous]
            
            case Gain.CONTINUOUS_AND_CATEGORICAL:
                return [Gain.gainContinuousCategorical,Gain.splitContinuousCategorical]
            
            case _: 
                raise ValueError("Esse não é o código de uma função de ganho válida.")

    @staticmethod
    def gainContinuous(data,target,feature_name,hyperparam):        
        acc=[]
        regularization =hyperparam.regularization
        index =np.isin(hyperparam.Order[feature_name],data.index.to_numpy())
        index = hyperparam.Order[feature_name][index]
        
        new_target = pd.DataFrame(target.reshape(-1,1),columns=[-1],index= data.index)

        new_target = pd.concat([data,new_target],axis=1)

        new_target =new_target.loc[index]

        split_points =np.unique(new_target[feature_name].to_numpy())
        split_points=(split_points[1:] + split_points[:-1])/2
                
        split_sd =Gain.calcGain(split_points,new_target.to_numpy(),regularization)
            
        acc= tuple(zip(split_sd,[feature_name]*split_points.shape[0],list(split_points)))
            

        return acc

    @staticmethod
    @jit
    def calcGain(split_points,matrix,regularization):

        split_sd=[]     
        for s in split_points:

            index = matrix[:,0]>= s
            
            sum_right_target = np.sum(matrix[index,1])
            sum_right_target *=sum_right_target
            
            count_right_target =np.sum(index)

            sum_left_target = np.sum(matrix[~index,1])
            sum_left_target *=sum_left_target

            count_left_target =np.sum(~index)

            split_sd.append( (sum_right_target/(count_right_target+regularization)) \
                        +(sum_left_target/(count_left_target+regularization)))
            
        return split_sd
            

    @staticmethod
    def splitValueCalc(target,regularization):
        return (np.sum(target)**2)/(target.shape[0]+ regularization)
    
    @staticmethod
    def _gainCategorical(data,target,feature_analise_name,hyperparam):

        acc =0
        values = data[feature_analise_name].unique()

        for v in values:
            index = data[feature_analise_name]==v
            new_target = np.array(target[index])
            
            acc += (np.sum(new_target)**2)/(new_target.shape[0]+ hyperparam.regularization)

        
        return [[acc,feature_analise_name,None]]
        
    
    def gainContinuousCategorical(data,target,feature_analise_name,hyperparam):
        
        gain = Gain._gainCategorical if feature_analise_name in hyperparam.categoricalFeatures else Gain.gainContinuous
        
        return gain(data,target,feature_analise_name,hyperparam)
    
    
    def splitCategorical(data,target,feature_split,split_value=None,):
        
        partitions = {}
        
        for v in data[feature_split].unique():
            
            index =data[feature_split]==v
            
            partitions[v] = [data[index],target[index]]
        
        return partitions
    
    def splitContinuous(data,target,feature_split,split_value,is_categorical=None):
        
        idx_1 = data[feature_split]>=split_value
        idx_2 = ~idx_1

        partitions ={}
            
        partitions[f'>={split_value}']=[data[idx_1],target[idx_1]]
        partitions[f'<{split_value}']=[data[idx_2],target[idx_2]]
        
        return partitions
            
    def splitContinuousCategorical(data,target,feature_analise_name,split_value,is_categorical=False):
        
        split = Gain.splitCategorical if is_categorical else Gain.splitContinuous
        
        return split(data,target,feature_analise_name,split_value)