from backend.dataflow.basicblock import BasicBlock

"""
CFG: Control Flow Graph

nodes: sequence of basicblock
edges: sequence of edge(u,v), which represents after block u is executed, block v may be executed
links: links[u][0] represent the Prev of u, links[u][1] represent the Succ of u,
"""


class CFG:
    def __init__(self, nodes: list[BasicBlock], edges: list[(int, int)]) -> None:
        self.nodes = nodes
        self.edges = edges

        self.links = []

        for i in range(len(nodes)):
            self.links.append((set(), set()))

        for u, v in edges:
            self.links[u][1].add(v)
            self.links[v][0].add(u)

        """
        You can start from basic block 0 and do a DFS traversal of the CFG
        to find all the reachable basic blocks.
        """
        stack = [0]
        self.nodes[0].reach = True
        while len(stack) > 0:
            node_index = stack.pop()
            succ_indices = self.links[node_index][1]
            for succ_index in succ_indices:
                if not self.nodes[succ_index].reach:
                    self.nodes[succ_index].reach = True
                    stack.append(succ_index)

    def getBlock(self, id):
        return self.nodes[id]

    def getPrev(self, id):
        return self.links[id][0]

    def getSucc(self, id):
        return self.links[id][1]

    def getInDegree(self, id):
        return len(self.links[id][0])

    def getOutDegree(self, id):
        return len(self.links[id][1])

    def iterator(self):
        return iter(self.nodes)
