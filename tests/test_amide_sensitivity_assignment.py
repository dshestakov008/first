"""Sensitivity compares one reported estimator and retains real area changes."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import savgol_filter

from amide_analysis.config import default_config
from amide_analysis.fitting import STRUCTURES, _percentages, analyze_band, gaussian


class SharedSensitivityAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config['resolution_confirmed'] = True
        self.config['fitting']['bootstrap_iterations'] = 12
        self.x = np.linspace(1600., 1700., 53)

    def model(self, centers, amplitudes=None, widths=None):
        count = len(centers)
        if amplitudes is None:
            amplitudes = np.full(count, .8)
        if widths is None:
            widths = np.full(count, 12.)
        params = np.column_stack((amplitudes, centers, widths))
        curve = np.sum([gaussian(self.x, *row) for row in params], axis=0)
        return dict(params=params, fit=curve, sse=1e-7, aicc=-100., bic=-90.,
                    count=count, condition_number=100., bound_hits=False,
                    initial_centers=np.asarray(centers), admissible=True,
                    admissibility_reasons=[], nfev=10)

    def processed(self, model):
        residual = np.random.default_rng(791).normal(0, 1e-5, len(self.x))
        return dict(x=self.x, corrected=model['fit'] + residual,
                    noise_sd=1e-5, step=self.x[1]-self.x[0], source_step=1.92905,
                    derivatives={5: np.zeros(len(self.x)), 7: np.zeros(len(self.x))})

    def execute(self, primary, alternatives, ci=None, bootstrap_percentages=None):
        processed = self.processed(primary)
        processed['sensitivity_variants'] = [dict(label=label, processed=self.processed(model))
                                            for label, model in alternatives]
        params = np.repeat(primary['params'][None, :, :], 12, axis=0)
        if ci is not None:
            params[:6, :, 1] = np.asarray(ci)[:, 0]
            params[6:, :, 1] = np.asarray(ci)[:, 1]
        relative = _percentages(primary, self.config)[1]
        percentages = (np.repeat(relative[None, :], 12, axis=0)
                       if bootstrap_percentages is None else bootstrap_percentages)
        boot = dict(parameters=params, percentages=percentages, success_count=12,
                    success_fraction=1., count_support=1., count_comparison_fraction=1.,
                    block_length=3, diagnostics=[])
        selections = [(primary, [primary], [{'stable': True}])] + [
            (model, [model], [{'stable': True}]) for _, model in alternatives]
        with patch('amide_analysis.fitting._select_model', side_effect=selections), \
                patch('amide_analysis.fitting._bootstrap', return_value=boot):
            return analyze_band(processed, self.config)

    def test_single_peak_boundary_shift_changes_assignment_without_false_area_flag(self):
        primary = self.model([1641.342])
        alternate = self.model([1640.507])
        result = self.execute(primary, [('endpoints', alternate)])
        self.assertTrue(result['quantitative_accepted'], result['warnings'])
        self.assertEqual(result['components'][0]['structure'], 'unassigned')
        self.assertEqual(result['percentages']['unassigned'], 100.)
        row = result['sensitivity'][0]
        self.assertAlmostEqual(row['nominal_center_assignment_delta_pct'], 100.)
        self.assertAlmostEqual(row['max_component_area_delta_pct'], 0.)
        self.assertAlmostEqual(row['max_structure_delta_pct'], 0.)
        self.assertAlmostEqual(row['unassigned'], 100.)
        self.assertEqual(row['nominal_assignment_change_count'], 1)
        self.assertNotIn('PREPROCESSING_AREA_OR_COUNT_INSTABILITY', result['fit_rejection_codes'])
        component = result['components'][0]
        self.assertAlmostEqual(component['center_ci_low'], 1641.342)
        self.assertAlmostEqual(component['assignment_interval_low'], 1639.507)

    def test_component_area_tradeoff_is_rejected_even_when_both_unassigned(self):
        primary = self.model([1640., 1650.], amplitudes=[.8, .2])
        alternate = self.model([1640., 1650.], amplitudes=[.6, .4])
        result = self.execute(primary, [('baseline', alternate)], ci=[[1638,1642],[1648,1652]])
        self.assertEqual(result['diagnostic_percentages']['unassigned'], 100.)
        row = result['sensitivity'][0]
        self.assertAlmostEqual(row['max_structure_delta_pct'], 0.)
        self.assertGreater(row['max_component_area_delta_pct'], 19.9)
        self.assertGreater(row['max_delta_pct'], 19.9)
        self.assertIn('PREPROCESSING_AREA_OR_COUNT_INSTABILITY', result['fit_rejection_codes'])

    def test_every_variant_uses_final_common_assignment_in_any_order(self):
        primary = self.model([1655.])
        inside = self.model([1656.])
        crossing = self.model([1659.6])
        first = self.execute(primary, [('inside', inside), ('crossing', crossing)])
        second = self.execute(primary, [('crossing', crossing), ('inside', inside)])
        for result in [first, second]:
            self.assertEqual(result['diagnostic_percentages']['unassigned'], 100.)
            self.assertEqual(result['components'][0]['structure'], 'unassigned')
            for row in result['sensitivity']:
                self.assertEqual(row['unassigned'], 100.)
                self.assertEqual(row['alpha_helix'], 0.)
                self.assertAlmostEqual(row['max_delta_pct'], 0.)
        self.assertEqual(first['components'], second['components'])

    def test_bootstrap_structure_sd_uses_same_envelope_as_reported_fractions(self):
        primary = self.model([1640., 1655.], amplitudes=[.5, .5])
        alternate = self.model([1640., 1659.6], amplitudes=[.5, .5])
        samples = np.tile([49., 51.], (12, 1))
        samples[6:] = [51., 49.]
        result = self.execute(primary, [('boundary', alternate)],
                              ci=[[1638.,1642.],[1654.9,1655.1]],
                              bootstrap_percentages=samples)
        self.assertEqual(result['diagnostic_percentages']['unassigned'], 100.)
        self.assertAlmostEqual(result['metrics']['max_structure_pct_sd'], 0.)
        self.assertGreater(result['metrics']['max_area_pct_sd'], 1.)

    def test_changed_count_has_no_invented_aligned_percentage_delta(self):
        primary = self.model([1655.])
        alternate = self.model([1630., 1655.])
        result = self.execute(primary, [('new_count', alternate)])
        row = result['sensitivity'][0]
        self.assertTrue(row['component_count_changed'])
        self.assertFalse(row['aligned_component_comparison'])
        self.assertTrue(np.isnan(row['max_component_area_delta_pct']))
        self.assertTrue(np.isnan(row['max_structure_delta_pct']))
        self.assertTrue(np.isnan(row['max_delta_pct']))
        self.assertIn('PREPROCESSING_AREA_OR_COUNT_INSTABILITY', result['fit_rejection_codes'])

    def test_inadmissible_variant_does_not_expand_shared_assignment(self):
        primary = self.model([1655.])
        alternate = self.model([1659.6])
        alternate.update(admissible=False, bound_hits=True,
                         admissibility_reasons=['WIDTH_LOWER_BOUND'])
        result = self.execute(primary, [('invalid', alternate)])
        self.assertEqual(result['diagnostic_percentages']['alpha_helix'], 100.)
        self.assertIn('PREPROCESSING_VARIANT_QC_FAILED', result['fit_rejection_codes'])

    def test_real_gaussian_fit_near_boundary_keeps_area_stability(self):
        """Exercise detection, constrained fitting and bootstrap on sampled curves."""
        self.config['fitting']['multistarts'] = 2
        primary = self.model([1641.342])
        alternate = self.model([1640.507])
        processed = self.processed(primary)
        variant = self.processed(alternate)
        for data in [processed, variant]:
            data['derivatives'] = {window: savgol_filter(
                data['corrected'], window, 3, deriv=2,
                delta=data['step'], mode='interp') for window in [5,7,9]}
        processed['sensitivity_variants'] = [dict(label='analytic_shift', processed=variant)]
        result = analyze_band(processed, self.config, seed=92)
        self.assertTrue(result['quantitative_accepted'], result['warnings'])
        self.assertEqual(result['metrics']['components'], 1)
        self.assertEqual(result['components'][0]['structure'], 'unassigned')
        row = result['sensitivity'][0]
        self.assertGreater(row['nominal_center_assignment_delta_pct'], 99.)
        self.assertLess(row['max_delta_pct'], 1e-8)
        self.assertAlmostEqual(result['percentages']['unassigned'], 100.)


if __name__ == '__main__':
    unittest.main()
