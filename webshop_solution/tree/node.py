class Node:
    def __init__(self, depth: int):
        self.depth = depth
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
