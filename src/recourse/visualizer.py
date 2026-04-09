class RecourseVisualizer:
    """Create interactive recourse graph visualizations."""
    
    def create_interactive_graph(self, graph, paths=None, highlight_path=None):
        """Use pyvis to create interactive HTML graph."""
        from pyvis.network import Network
        
        net = Network(height='600px', width='100%', directed=True, bgcolor="#ffffff", font_color="#000000")
        
        # Extract node IDs from highlight path if it's a list of steps
        highlight_nodes = set()
        highlight_edges = set()
        if highlight_path:
            # Assume highlight_path is a list of node IDs from start to target
            highlight_nodes = set(highlight_path)
            for i in range(len(highlight_path) - 1):
                highlight_edges.add((highlight_path[i], highlight_path[i+1]))

        # Add nodes
        for node, data in graph.nodes(data=True):
            is_highlighted = node in highlight_nodes
            base_color = 'green' if data.get('prediction', 1) == 0 else 'red'
            color = "#FFD700" if is_highlighted else base_color # Gold highlight
            
            size = 25 if is_highlighted else (20 + (3 - data.get('depth', 0)) * 5)
            border_width = 4 if is_highlighted else 1
            
            features_summary = "<br>".join([f"<b>{k}</b>: {v}" for k,v in data.get('features', {}).items() if v is not None])
            
            net.add_node(
                node,
                label=f"State {str(node)[:6]}...",
                color={'background': color, 'border': '#000000'},
                size=size,
                borderWidth=border_width,
                title=f"<strong>Prediction:</strong> {'Approved' if data.get('prediction')==0 else 'Denied'}<br>{features_summary}"
            )
            
        # Add edges
        for source, target, data in graph.edges(data=True):
            is_highlighted = (source, target) in highlight_edges
            color = "#FFD700" if is_highlighted else "#808080"
            width = 5 if is_highlighted else 1
            
            action = data.get('action')
            action_desc = action.description if hasattr(action, 'description') else str(action)
            cost = data.get('cost', 0)
            time = data.get('time', 0)
            
            net.add_edge(
                source, target,
                label=f"${cost:.0f}",
                width=width,
                color=color,
                title=f"{action_desc}<br>Time: {time} months"
            )
            
        # Enable physics for better layout
        net.toggle_physics(True)
        return net
