import random 
import numpy as np

class SumTree(object):
    data_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.length = 0
        
    def update(self, tree_index, priority):
        change = priority-self.tree[tree_index]
        self.tree[tree_index] = priority
        
        while tree_index>0:
            tree_index = (tree_index-1)//2
            self.tree[(tree_index)] += change
    
    def add(self, priority, data):
        tree_index = self.data_pointer+self.capacity-1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer+=1
        if(self.data_pointer>=self.capacity):
            self.data_pointer=0
        else: 
            self.length+=1
    
    def get_leaf(self, v):
        pi = 0
        while True:
            lci = pi*2+1
            rci = lci+1
            
            if lci>=self.capacity:
                leaf_index = pi
                break
            else:
                if v<=self.tree[lci]:
                    pi = lci
                else:
                    v-=self.tree[lci]
                    pi = rci
        data_index = leaf_index-self.capacity+1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]
    

class Memory(object):
    e = 0.04
    a = 0.6
    b = 0.4
    b_increment = 0.001
    absolute_error_upper = 1   
    
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity=capacity
        self.memories_n = 0
    
    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)
        self.memories_n+=1
    
    def sample(self, n):
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)
        priority_segment = self.tree.total_priority/n
        for i in range(n):
            A,B = priority_segment*i, priority_segment*(i+1)
            value = np.random.uniform(A,B)
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i] = index
            minibatch.append(list(data))
        return b_idx, minibatch
    
    def batch_update(self, tree_idx, abs_error):
        abs_error+=self.e
        clipped_error = np.minimum(abs_error, self.absolute_error_upper)
        ps = np.power(clipped_error, self.a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)