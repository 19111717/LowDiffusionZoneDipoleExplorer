"""Self-contained smoke tests for the GitHub-ready explorer copy."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from LowDiffusionDipoleExplorer import (
    ExplorerState,
    LowDiffusionDipoleExplorer,
    calculate_state,
)
from cygbubble import config
from cygbubble.physics.diffusion import low_parallel_ratio_to_energy


class ExplorerSmokeTests(unittest.TestCase):
    def test_default_state_returns_finite_physical_outputs(self):
        result = calculate_state(ExplorerState())
        self.assertGreater(result["affected_dipole_amplitude"], 0.0)
        self.assertTrue(np.isfinite(
            result["affected_right_ascension_amplitude"]
        ))
        self.assertAlmostEqual(
            low_parallel_ratio_to_energy(config.A_at_10TeV),
            10.0,
        )

    def test_observations_load_only_on_energy_scan(self):
        import matplotlib.pyplot as plt

        explorer = LowDiffusionDipoleExplorer()
        try:
            explorer.set_li2024_data_visible(True)
            self.assertEqual(len(explorer.li2024_data_artists), 0)

            explorer.set_curve_mode("parameter scan")
            explorer.set_scan_parameter("d_low_over_d_parallel")

            self.assertIsNone(explorer.li2024_data_load_error)
            self.assertEqual(
                sum(len(handle.get_xdata())
                    for handle in explorer.li2024_amplitude_handles),
                44,
            )
            self.assertEqual(
                sum(len(handle.get_xdata())
                    for handle in explorer.li2024_phase_handles),
                43,
            )
            self.assertTrue(all(
                artist.get_visible()
                for artist in explorer.li2024_data_artists
            ))

            explorer.set_scan_parameter("ma")
            self.assertTrue(all(
                not artist.get_visible()
                for artist in explorer.li2024_data_artists
            ))
        finally:
            plt.close(explorer.figure)


if __name__ == "__main__":
    unittest.main()

