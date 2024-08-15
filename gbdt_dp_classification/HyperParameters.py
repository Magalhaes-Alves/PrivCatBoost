class HyperParameters:

    def __init__(self,
                 regularization=None,
                 learning_rate=None,
                 sensitivity_g=None,
                 sensitivity_v=None,
                 max_depth=None,
                 categorical_features=None,
                 min_node_support =None,
                 type_gain=None,
                 type_tree=None,
                 order=None):
        
        self._gl = 1
        self._regularization = regularization
        self._sensitivity_g = sensitivity_g
        self._sensitivity_v = sensitivity_v
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._categorical_features=categorical_features
        self._min_node_support =min_node_support
        self._type_gain = type_gain
        self._type_tree = type_tree
        self._order =order
        

    @property
    def Order(self):
        return self._order
    
    @Order.setter
    def Order(self,order):
        self._order = order

    @property
    def typeTree(self):
        return self._type_tree

    @property
    def typeGain(self):
        return self._type_gain
    
    @property
    def partionStrategy(self):
        return self._partion_strategy
    
    @property
    def typeGain(self):
        return self._type_gain

    @typeGain.setter
    def typeGain(self,g):
        self._type_gain=g

    @property
    def minSampleSize(self):
        return self._min_node_support
    
    @minSampleSize.setter
    def minSampleSize(self,s):
        self._min_node_support=s

    @property
    def categoricalFeatures(self):
        return self._categorical_features
    
    @categoricalFeatures.setter
    def categoricalFeatures(self,cat):
        self._categorical_features=cat

    @property
    def maxDepth(self):
        return self._max_depth
    
    @maxDepth.setter
    def maxDepth(self,m):
        self._max_depth=m

    @property
    def regularization(self):
        return self._regularization
    
    @regularization.setter
    def regularization(self,r):
        self._regularization=r

    @property
    def gl(self):
        return self._gl
    
    @gl.setter
    def gl(self,gl):
        self._gl =gl

    @property
    def sensitivityG(self):
        return self._sensitivity_g
    
    @sensitivityG.setter
    def sensitivityG(self,g):
        self._sensitivity_g=g

    @property
    def sensitivityV(self):
        return self._sensitivity_v
    
    @sensitivityV.setter
    def sensitivityV(self,v):
        self._sensitivity_v=v

    @property
    def learningRate(self):
        return self._learning_rate
    
    @learningRate.setter
    def learningRate(self,lr):
        self._learning_rate=lr

        