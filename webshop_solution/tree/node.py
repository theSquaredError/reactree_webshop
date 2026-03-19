class Node:
    _id_counter = 0

    def __init__(self, depth: int):
        self.depth = depth
        self.children = []
        self.parent = None
        self.node_id = Node._id_counter
        Node._id_counter += 1

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
