"""
Unit tests for multiworld-causal-subsidy project.

Run with:  pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest


# ============================================================
#  Tests for src/evaluation/metrics.py
# ============================================================

class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_returns_tuple_of_two(self):
        from src.evaluation.metrics import bootstrap_ci

        data = np.random.normal(0, 1, 100)
        ci = bootstrap_ci(data, n_bootstrap=100, ci=0.95)
        assert isinstance(ci, tuple), "bootstrap_ci should return a tuple"
        assert len(ci) == 2, "bootstrap_ci should return (lower, upper)"

    def test_ci_contains_true_mean(self):
        from src.evaluation.metrics import bootstrap_ci

        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, 500)
        true_mean = float(np.mean(data))
        ci = bootstrap_ci(data, n_bootstrap=1000, ci=0.95)
        assert ci[0] < true_mean < ci[1], "95% CI should contain the sample mean"

    def test_ci_wider_for_higher_confidence(self):
        from src.evaluation.metrics import bootstrap_ci

        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        ci_90 = bootstrap_ci(data, n_bootstrap=500, ci=0.90)
        ci_95 = bootstrap_ci(data, n_bootstrap=500, ci=0.95)
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        assert width_95 > width_90, "95% CI should be wider than 90% CI"


class TestComputeROI:
    """Tests for compute_roi function."""

    def test_positive_roi(self):
        from src.evaluation.metrics import compute_roi

        # ROI = (GTV - cost) / cost = (300 - 100) / 100 = 2.0
        roi = compute_roi(subsidy_cost=100.0, incremental_gtv=300.0)
        assert roi == pytest.approx(2.0, rel=1e-6), "ROI = (GTV - cost) / cost"

    def test_zero_cost_returns_zero(self):
        from src.evaluation.metrics import compute_roi

        # Implementation returns 0.0 for zero/negative cost (not raise)
        roi = compute_roi(subsidy_cost=0.0, incremental_gtv=100.0)
        assert roi == 0.0, "ROI for zero cost should return 0.0"

    def test_negative_roi(self):
        from src.evaluation.metrics import compute_roi

        # ROI = (50 - 100) / 100 = -0.5
        roi = compute_roi(subsidy_cost=100.0, incremental_gtv=50.0)
        assert roi == pytest.approx(-0.5, rel=1e-6)


class TestEValue:
    """Tests for e_value function."""

    def test_returns_float(self):
        from src.evaluation.metrics import e_value

        ev = e_value(rr=2.0)
        assert isinstance(ev, float), "e_value should return a float"

    def test_monotonic_with_rr(self):
        from src.evaluation.metrics import e_value

        ev_2 = e_value(rr=2.0)
        ev_3 = e_value(rr=3.0)
        assert ev_3 > ev_2, "E-value should increase with RR"

    def test_e_value_at_rr_1(self):
        from src.evaluation.metrics import e_value

        ev = e_value(rr=1.0)
        assert ev == pytest.approx(1.0, rel=1e-6), "E-value for RR=1 should be 1.0"


# ============================================================
#  Tests for src/simulation/cognitive_agent_theory.py
# ============================================================

class TestProspectValue:
    """Tests for prospect_value function."""

    def test_value_at_zero(self):
        from src.simulation.cognitive_agent_theory import prospect_value

        v = prospect_value(0.0)
        assert v == pytest.approx(0.0, abs=1e-10), "v(0) = 0"

    def test_loss_aversion(self):
        """Losses should feel steeper than equivalent gains (lambda > 1)."""
        from src.simulation.cognitive_agent_theory import prospect_value

        gain = prospect_value(10.0)
        loss = prospect_value(-10.0)
        assert abs(loss) > gain, "Loss aversion: |v(-x)| > v(x) for same |x|"

    def test_concavity_for_gains(self):
        """Value function should be concave for gains (diminishing sensitivity)."""
        from src.simulation.cognitive_agent_theory import prospect_value

        v_10 = prospect_value(10.0)
        v_20 = prospect_value(20.0)
        assert v_20 < 2 * v_10, "Concavity: v(2x) < 2*v(x) for gains"

    def test_convexity_for_losses(self):
        """Value function should be convex for losses (diminishing sensitivity)."""
        from src.simulation.cognitive_agent_theory import prospect_value

        v_neg10 = abs(prospect_value(-10.0))
        v_neg20 = abs(prospect_value(-20.0))
        assert v_neg20 < 2 * v_neg10, "Convexity: |v(-2x)| < 2*|v(-x)| for losses"


class TestTheoreticalCognitiveAgent:
    """Tests for TheoreticalCognitiveAgent class."""

    def test_initialization(self):
        from src.simulation.cognitive_agent_theory import (
            TheoreticalCognitiveAgent, MentalAccountType,
        )

        agent = TheoreticalCognitiveAgent(
            agent_id="test_001",
            mental_account=MentalAccountType.WINDFALL_SPENDER,
        )
        assert agent.agent_id == "test_001"
        assert agent.reference_point == 0.0
        assert len(agent.history) == 0

    def test_decide_returns_bool(self):
        from src.simulation.cognitive_agent_theory import (
            TheoreticalCognitiveAgent, MentalAccountType,
        )

        agent = TheoreticalCognitiveAgent(
            agent_id="test_002",
            mental_account=MentalAccountType.WINDFALL_SPENDER,
        )
        decision = agent.decide(subsidy_amount=10.0)
        assert isinstance(decision, bool), "decide() should return a bool"

    def test_update_state_increments_redemptions(self):
        from src.simulation.cognitive_agent_theory import (
            TheoreticalCognitiveAgent, MentalAccountType,
        )

        agent = TheoreticalCognitiveAgent(
            agent_id="test_003",
            mental_account=MentalAccountType.WINDFALL_SPENDER,
        )
        old_rp = agent.reference_point
        agent.update_state(was_subsidized=True, redeemed=True)
        assert len(agent.history) == 1
        assert agent.history[0].get('redeemed', False) or agent.reference_point != old_rp

    def test_all_mental_account_types_run(self):
        from src.simulation.cognitive_agent_theory import (
            TheoreticalCognitiveAgent, MentalAccountType,
        )

        for account_type in MentalAccountType:
            agent = TheoreticalCognitiveAgent(
                agent_id=f"test_{account_type.value}",
                mental_account=account_type,
            )
            decision = agent.decide(subsidy_amount=10.0)
            assert isinstance(decision, bool), f"Failed for {account_type.value}"


# ============================================================
#  Tests for src/modeling/psm_matcher.py
# ============================================================

class TestPSMMatcher:
    """Tests for PSM matcher module."""

    @pytest.fixture
    def psm_data(self):
        """Generate synthetic data for PSM testing."""
        np.random.seed(42)
        n = 500
        X1 = np.random.randn(n)
        X2 = np.random.binomial(1, 0.5, n)
        propensity = 1 / (1 + np.exp(-0.5 * X1 + 0.3 * X2))
        treatment = np.random.binomial(1, propensity, n)
        outcome = X1 + 0.5 * X2 + treatment * 2.0 + np.random.randn(n)
        return pd.DataFrame({
            'X1': X1, 'X2': X2, 'treatment': treatment, 'outcome': outcome
        })

    def test_compute_propensity_scores(self, psm_data):
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        matcher = PSMMatcher(PSMConfig(method='nearest'))
        matcher.compute_propensity_scores(psm_data, 'treatment', ['X1', 'X2'])
        assert matcher.propensity_scores is not None
        assert len(matcher.propensity_scores) == len(psm_data)

    def test_match_returns_dataframe(self, psm_data):
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        matcher = PSMMatcher(PSMConfig(method='nearest'))
        matcher.compute_propensity_scores(psm_data, 'treatment', ['X1', 'X2'])
        matched = matcher.match(psm_data, 'treatment')
        assert isinstance(matched, pd.DataFrame)
        # Check that matched data has treatment and outcome columns
        assert 'treatment' in matched.columns

    def test_smd_decreases_after_matching(self, psm_data):
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        matcher = PSMMatcher(PSMConfig(method='nearest'))
        matcher.compute_propensity_scores(psm_data, 'treatment', ['X1', 'X2'])
        matched = matcher.match(psm_data, 'treatment')
        quality = matcher.evaluate_match_quality(psm_data, matched, 'treatment')
        assert 'smd_matched_max' in quality
        assert quality['smd_matched_max'] < quality['smd_original_max'], \
            "Matching should reduce max SMD"

    def test_optimal_matching_runs(self, psm_data):
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        matcher = PSMMatcher(PSMConfig(method='optimal', caliper=0.2))
        matcher.compute_propensity_scores(psm_data, 'treatment', ['X1', 'X2'])
        matched = matcher.match(psm_data, 'treatment')
        assert isinstance(matched, pd.DataFrame)

    def test_caliper_matching_runs(self, psm_data):
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        matcher = PSMMatcher(PSMConfig(method='caliper', caliper=0.05))
        matcher.compute_propensity_scores(psm_data, 'treatment', ['X1', 'X2'])
        matched = matcher.match(psm_data, 'treatment')
        assert isinstance(matched, pd.DataFrame)


# ============================================================
#  Tests for src/features/data_generator.py
# ============================================================

class TestDataGenerator:
    """Tests for synthetic data generator."""

    def test_generate_returns_dict(self):
        from src.features.data_generator import generate_all_data, SyntheticDataConfig

        config = SyntheticDataConfig(n_users=200, n_orders=500)
        data = generate_all_data(config)
        assert isinstance(data, dict)
        assert 'user_profiles' in data
        assert 'orders' in data
        assert 'causal_data' in data

    def test_causal_data_has_true_cate(self):
        from src.features.data_generator import generate_all_data, SyntheticDataConfig

        config = SyntheticDataConfig(n_users=200, n_orders=500)
        data = generate_all_data(config)
        assert 'true_cate' in data['causal_data'].columns

    def test_reproducibility_with_same_seed(self):
        from src.features.data_generator import generate_all_data, SyntheticDataConfig

        c1 = SyntheticDataConfig(n_users=100, n_orders=200, random_state=42)
        c2 = SyntheticDataConfig(n_users=100, n_orders=200, random_state=42)
        d1 = generate_all_data(c1)
        d2 = generate_all_data(c2)
        pd.testing.assert_frame_equal(d1['user_profiles'], d2['user_profiles'])


# ============================================================
#  Tests for src/simulation/network_contagion.py
# ============================================================

class TestSocialNetwork:
    """Tests for SocialNetwork class."""

    def test_build_barabasi_albert(self):
        from src.simulation.network_contagion import SocialNetwork

        sn = SocialNetwork()
        G = sn.build_barabasi_albert(n=100, m=3, seed=42)
        assert G.number_of_nodes() == 100
        assert G.number_of_edges() > 0

    def test_build_watts_strogatz(self):
        from src.simulation.network_contagion import SocialNetwork

        sn = SocialNetwork()
        G = sn.build_watts_strogatz(n=100, k=4, p=0.3, seed=42)
        assert G.number_of_nodes() == 100

    def test_build_from_cooccurrence(self):
        from src.simulation.network_contagion import SocialNetwork

        sn = SocialNetwork()
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'poi_id': ['A', 'B', 'A', 'C', 'B']
        })
        G = sn.build_from_cooccurrence(df, 'user_id', 'poi_id')
        assert G.number_of_nodes() > 0


class TestSocialContagion:
    """Tests for SocialContagion class."""

    def test_propagate_returns_dict(self):
        from src.simulation.network_contagion import SocialNetwork, SocialContagion

        sn = SocialNetwork()
        G = sn.build_barabasi_albert(n=50, m=3, seed=42)
        sc = SocialContagion()
        result = sc.propagate(G, seed_nodes=[0, 1], contagion_rate=0.2, n_steps=10, seed=42)
        assert isinstance(result, dict)
        assert 'time_series' in result
        assert 'cascade_size' in result
        assert 'states' in result

    def test_propagate_time_series_has_sir(self):
        from src.simulation.network_contagion import SocialNetwork, SocialContagion

        sn = SocialNetwork()
        G = sn.build_barabasi_albert(n=50, m=3, seed=42)
        sc = SocialContagion()
        result = sc.propagate(G, seed_nodes=[0, 1], contagion_rate=0.2, n_steps=10, seed=42)
        ts = result['time_series']
        assert len(ts) > 0
        assert 'S' in ts[0] and 'I' in ts[0] and 'R' in ts[0]

    def test_estimate_social_effect(self):
        from src.simulation.network_contagion import SocialNetwork, SocialContagion

        sn = SocialNetwork()
        G = sn.build_barabasi_albert(n=50, m=3, seed=42)
        sc = SocialContagion()
        result = sc.estimate_social_effect(G, treatment_nodes=[0, 1, 2], n_simulations=10, seed=42)
        assert 'social_effect' in result
        assert 'social_effect_ratio' in result


# ============================================================
#  Integration test: data → CausalML
# ============================================================

class TestCausalMLIntegration:
    """Integration test: data generation → CausalML."""

    def test_tlearner_pipeline(self):
        from src.features.data_generator import generate_all_data, SyntheticDataConfig
        from src.modeling.causalml_wrapper import CausalMLWrapper, CausalMLConfig

        config = SyntheticDataConfig(n_users=500, n_orders=2000, random_state=42)
        data = generate_all_data(config)
        causal_data = data['causal_data']
        feature_cols = [c for c in causal_data.columns
                        if c not in ('treatment', 'outcome', 'true_cate')]

        wrapper = CausalMLWrapper(CausalMLConfig(learner_type='tlearner'))
        result = wrapper.fit_predict(causal_data, feature_cols, 'treatment', 'outcome')
        assert 'cate_causalml' in result.columns
        cate = result['cate_causalml'].values.flatten()
        true_cate = causal_data['true_cate'].values
        corr = np.corrcoef(cate, true_cate)[0, 1]
        assert corr > 0.5, f"CATE correlation {corr:.3f} too low"
