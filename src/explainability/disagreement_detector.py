from scipy.stats import kendalltau

class ExplanationDisagreementDetector:
    """Detect when different XAI methods disagree."""
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        
    def detect_disagreement(self, shap_exp, lime_exp, model_exp):
        """Compare feature rankings from different methods."""
        shap_ranking = self._get_ranking(shap_exp['feature_importance'])
        lime_ranking = self._get_ranking(lime_exp['feature_weights'])
        model_ranking = self._get_ranking(model_exp['feature_importance'])
        
        # We need a shared list of features to rank sequentially
        features = list(set(shap_ranking) | set(lime_ranking) | set(model_ranking))
        
        s_ranks = [shap_ranking.get(f, len(features)) for f in features]
        l_ranks = [lime_ranking.get(f, len(features)) for f in features]
        m_ranks = [model_ranking.get(f, len(features)) for f in features]
        
        tau_shap_lime, _ = kendalltau(s_ranks, l_ranks)
        tau_shap_model, _ = kendalltau(s_ranks, m_ranks)
        tau_lime_model, _ = kendalltau(l_ranks, m_ranks)
        
        avg_correlation = (tau_shap_lime + tau_shap_model + tau_lime_model) / 3
        
        disagreement = 1 - avg_correlation
        conflicts = self._find_conflicts(shap_ranking, lime_ranking, model_ranking)
        warn = disagreement > self.threshold
        
        return {
            'disagreement_score': disagreement,
            'correlations': {
                'shap_lime': tau_shap_lime,
                'shap_model': tau_shap_model,
                'lime_model': tau_lime_model
            },
            'conflicting_features': conflicts,
            'warn_user': warn,
            'reliability': 'low' if warn else 'high'
        }
        
    def _get_ranking(self, importance_dict):
        """Return dict of {feature: rank} where rank 1 is top feature."""
        sorted_feats = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return {k: rank for rank, (k, _) in enumerate(sorted_feats, 1)}

    def _find_conflicts(self, shap_rank, lime_rank, model_rank):
        """Find features where methods strongly disagree (e.g., top 5 in one but > 10 in another)."""
        conflicts = []
        features = list(set(shap_rank) | set(lime_rank) | set(model_rank))
        
        for f in features:
            s = shap_rank.get(f, 99)
            l = lime_rank.get(f, 99)
            m = model_rank.get(f, 99)
            
            ranks = [s, l, m]
            if min(ranks) <= 5 and max(ranks) > 10:
                conflicts.append({
                    'feature': f,
                    'ranks': {'shap': s, 'lime': l, 'model': m}
                })
        return conflicts
