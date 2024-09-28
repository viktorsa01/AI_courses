import numpy as np
from pathlib import Path
from typing import Tuple



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value

def pos_exmaples(examples):
    if examples.size == 0:
        return 0
    return np.count_nonzero(examples[:, -1] == 2)

def entropy(p) :
    #calculates entropy for a boolean with probability p of being positive
    if p == 0 or p ==1:
        return 0
    
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def gain(attribute, examples):
    
    remainder = 0
    for label in range(1,3):
        has_label = np.array([ ex for ex in examples if ex[attribute] == label])
        if has_label.size == 0:
            return 0
        
        p_k = pos_exmaples(has_label)
        #calculate the remaining entropy of the subtree at the current node of the current split
        remainder += (has_label.size)/(examples.size) * entropy(p_k/has_label.size)
    
    pos = pos_exmaples(examples)
    return entropy(pos/examples.size) - remainder

def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    # TODO implement the importance function for both measure = "random" and measure = "information_gain"
    
    #find the attribute a that maximize importance
    A = None 
    max_importance = -1
    for a in attributes:
        
        #calculate importance based on give measure
        if measure == "random":
            imp = np.random.random()
        else:
            imp = gain(a, examples)  
        if imp > max_importance:
            A = a
            max_importance = imp
    
    return A


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678
    if examples.size == 0: 
        node.value = plurality_value(parent_examples)
        return node.value
    elif len(set(examples[:, -1]))==0:
        node.value = plurality_value(examples)
        return node.value
    elif attributes.size == 0:
        node.value = plurality_value(examples)
        return node.value
    
    a = importance(attributes, examples, measure)
    node.attribute = a
    
    for label in range(1,3):
        #create a list of new examples based on the current split
        new_exs = np.array([ ex for ex in examples if ex[a] == label]) 
        #create subtree from current split
        child = learn_decision_tree(new_exs, np.delete(attributes, np.where(attributes == a)), examples, node, label, measure)
        
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':
    
    train, test = load_data()

    # information_gain or random
    measure = "information_gain"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy with information gain: {accuracy(tree, train)}")
    print(f"Test Accuracy with information gain: {accuracy(tree, test)}")
    
    N=1000
    train_accuracy_random = np.zeros(N)
    test_accuracy_random = np.zeros(N)
    for i in range(N):
        measure = "random"

        tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)
     
        train_accuracy_random[i] = accuracy(tree, train)  
        test_accuracy_random[i] = accuracy(tree, test)  
    
    print(f"Mean Training Accuracy over {N} trees using random importance: {np.mean(train_accuracy_random)}")
    print(f"Mean Test Accuracy over {N} trees using random importance: {np.mean(test_accuracy_random)}")  