import networkx as nx
import heapq
import time
import pandas as pd
from ..core.logger import logger

class RecourseGraphBuilder:
    """Build graph of possible state transitions from rejection to approval."""
    
    def __init__(self, model, action_space, cost_model, preprocessor=None):
        self.model = model
        self.action_space = action_space
        self.cost_model = cost_model
        self.preprocessor = preprocessor
        self.graph = nx.DiGraph()
        self.prediction_cache = {}
        
    def _state_to_id(self, state):
        # Optimized hashing for constant-key dictionaries
        return hash(frozenset(state.items()))
        
    def _predict_batch(self, states):
        """Predict multiple states at once for efficiency."""
        if not states:
            return []
        df = pd.DataFrame.from_records(states)
        if self.preprocessor:
             processed = self.preprocessor.transform(df)
        else:
             processed = df
        return self.model.predict(processed).tolist()

    def _predict_proba_batch(self, states):
        """Get probabilities for multiple states for greedy search."""
        if not states:
            return []
        df = pd.DataFrame.from_records(states)
        if self.preprocessor:
             processed = self.preprocessor.transform(df)
        else:
             processed = df
        return self.model.predict_proba(processed)[:, 1].tolist()

    def find_greedy_path(self, initial_state: dict, max_depth=10):
        """Fastest path-finding: always pick the action that most reduces default probability."""
        path = []
        current_state = initial_state
        
        # Check initial state
        initial_pred = self._predict_batch([current_state])[0]
        
        if initial_pred == 0:
            return [] # Already approved
            
        for d in range(max_depth):
            actions = self.action_space.get_actions(current_state)
            if not actions:
                break
                
            next_states = [a.apply(current_state) for a in actions]
            probs = self._predict_proba_batch(next_states)
            
            # Find action that minimizes probability of default (Target=1)
            best_idx = probs.index(min(probs))
            best_action = actions[best_idx]
            best_prob = probs[best_idx]
            best_state = next_states[best_idx]
            
            path.append({
                'action': best_action,
                'state': best_state,
                'prob': best_prob
            })
            
            # Check if this new state is approved
            # Pred 0 is Approved
            if self._predict_batch([best_state])[0] == 0:
                return path # Success!
                
            current_state = best_state
            
        return path # Failed to find approval or just returned the sequence reached

    def build_graph(self, initial_state: dict, max_depth=10, max_states=1000):
        start_time_perf = time.time()
        timeout = 30.0 # Increased for deeper Dijkstra search and better state exploration
        
        start_id = self._state_to_id(initial_state)
        # Prediction caching to avoid redundant work
        self.prediction_cache = {start_id: self._predict_batch([initial_state])[0]}
        
        self.graph.add_node(start_id, features=initial_state, prediction=self.prediction_cache[start_id], depth=0)
        
        # Priority Queue: (cumulative_cost, state_id)
        pq = [(0, start_id)]
        costs = {start_id: 0}
        
        nodes_expanded = 0
        
        # To avoid redundant batch predictions in Dijkstra, we'll still use a small batching buffer
        pending_evals = [] # (next_id, next_state)
        pending_transitions = [] # (parent_id, next_id, action, time_cost, total_cost)

        while pq and nodes_expanded < max_states:
            if time.time() - start_time_perf > timeout:
                logger.warning("Recourse search timed out during Dijkstra expansion.")
                break
                
            curr_cost, curr_id = heapq.heappop(pq)
            
            # Skip if we already found a cheaper path to this node
            if curr_cost > costs.get(curr_id, float('inf')):
                continue
                
            node_data = self.graph.nodes[curr_id]
            # Goal: find pathways to approval. If node is already approved, don't expand further from here.
            if node_data['prediction'] == 0:
                continue
                
            if node_data['depth'] >= max_depth:
                continue

            nodes_expanded += 1
            curr_state = node_data['features']
            
            # 1. Expand node
            for action in self.action_space.get_actions(curr_state):
                next_state = action.apply(curr_state)
                next_id = self._state_to_id(next_state)
                action_cost = self.cost_model.calculate_cost(action)
                new_total_cost = curr_cost + action_cost
                
                # If we found a cheaper way or a new state
                if new_total_cost < costs.get(next_id, float('inf')):
                    costs[next_id] = new_total_cost
                    time_cost = self.cost_model.calculate_time(action)
                    
                    # Need to check if we know the prediction for this state
                    if next_id not in self.prediction_cache:
                        # Batch those for later (we'll process them in chunks)
                        pending_evals.append((next_id, next_state))
                        pending_transitions.append((curr_id, next_id, action, time_cost, new_total_cost))
                    else:
                        # Already predicted, add to graph and queue immediately
                        pred = self.prediction_cache[next_id]
                        if next_id not in self.graph:
                            self.graph.add_node(next_id, features=next_state, prediction=pred, depth=node_data['depth'] + 1)
                        self.graph.add_edge(curr_id, next_id, action=action, cost=action_cost, time=time_cost)
                        heapq.heappush(pq, (new_total_cost, next_id))

            # 2. Performance: Periodically process batch predictions
            # If buffer is full or we are out of nodes to expand in this 'level'
            if len(pending_evals) >= 32 or (not pq and pending_evals):
                batch_states = [s for _, s in pending_evals]
                batch_preds = self._predict_batch(batch_states)
                
                for (nid, nstate), (pid, nid2, act, t_cost, tot_cost), pred in zip(pending_evals, pending_transitions, batch_preds):
                    self.prediction_cache[nid] = pred
                    if nid not in self.graph:
                         self.graph.add_node(nid, features=nstate, prediction=pred, depth=self.graph.nodes[pid]['depth'] + 1)
                    self.graph.add_edge(pid, nid, action=act, cost=self.cost_model.calculate_cost(act), time=t_cost)
                    heapq.heappush(pq, (tot_cost, nid))
                
                pending_evals = []
                pending_transitions = []
                
        logger.info(f"Graph built (Dijkstra): {self.graph.number_of_nodes()} states, {nodes_expanded} expanded in {time.time() - start_time_perf:.2f}s.")
        return start_id
