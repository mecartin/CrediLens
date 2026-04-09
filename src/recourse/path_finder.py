import networkx as nx

class PathFinder:
    """Find paths through recourse graph."""
    
    def find_all_paths(self, graph, start_node, max_paths=10):
        # End nodes are those with prediction == 0 (Approved)
        approved_nodes = [n for n, data in graph.nodes(data=True) if data['prediction'] == 0]
        
        all_paths = []
        for target in approved_nodes:
            try:
                paths = list(nx.all_simple_paths(graph, start_node, target, cutoff=5))
                all_paths.extend(paths[:max_paths])
            except nx.NetworkXNoPath:
                continue
                
        # Format paths returning sequential edges with action payloads
        formatted_paths = []
        for p in all_paths:
             seq = []
             for i in range(len(p)-1):
                  edge = graph.edges[p[i], p[i+1]]
                  seq.append(edge['action'])
             formatted_paths.append(seq)
             
        return formatted_paths
