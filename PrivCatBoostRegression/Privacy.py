import numpy as np


def laplace_mech(value, epsilon, sensitivity):

    return value + np.random.laplace(0,sensitivity/epsilon)

def exponential(xi,yi,R,u,sensitivity,epsilon,hyperparam):
    
    #Para cada elemento em R será definido seu respectivo score usando a função custo u
    scores = [u(xi[[r]],yi,r,hyperparam) for r in R]

    scores = [s for scor in scores for s in scor]
        
    probabilities = np.clip(a=np.array([gain[0] for gain in scores]),a_min=-700,a_max=700)            
    #print(f'{epsilon=}\n{sensitivity=}\n{max_score=}\n{epsilon*max_score[0]/(2*sensitivity)=}')
    #Para cada score será aplicado a distribuição de probabilidade abaixo
    probabilities=np.exp((epsilon*probabilities)/(2*sensitivity))

    #Normaliza as probabilidades usando a norma L1
    probabilities = probabilities/ np.linalg.norm(probabilities,ord=1)
    
    #Seleciona um dos elementos do conjunto R baseados nas probabilidades construidas
    return scores[np.random.choice(range(len(scores)),1,p=probabilities)[0]]

