from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, min_max_inverse_scale, storm_max_value, get_meta_scalars
import unittest
from os.path import exists


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.patch_path = "../testdata/track_data_ncarstorm_3km_REFL_COM_ws_nc_small/"
        if not exists(self.patch_path):
            self.patch_path = "testdata/track_data_ncarstorm_3km_REFL_COM_ws_nc_small/"
        self.start_date = "2011-04-25"
        self.end_date = "2011-04-28"
        self.run_freq = "daily"
        self.input_variables = ["REFL_1KM_AGL_curr", "U10_curr", "V10_curr"]
        self.output_variable = ["UP_HELI_MAX_curr"]
        self.meta_variables = ["masks", "i", "j", "time", "centroid_lon", "centroid_lat",
                 "centroid_i", "centroid_j", "track_id", "track_step"]
        self.patch_radius = 16
        self.input_data, self.output_data, self.meta_data = load_patch_files(self.start_date,
                                                                             self.end_date,
                                                                             self.run_freq,
                                                                             self.patch_path,
                                                                             self.input_variables,
                                                                             self.output_variable,
                                                                             self.meta_variables,
                                                                             self.patch_radius)

    def test_load_patch_files(self):
        assert len(self.input_variables) == len(list(self.input_data.data_vars.keys()))
        assert len(self.output_variable) == len(list(self.output_data.data_vars.keys()))
        assert self.output_data[self.output_variable[0]].shape[0] == self.input_data[self.input_variables[0]].shape[0]
        return

    def test_combine_patch_data(self):
        combined = combine_patch_data(self.input_data, self.input_variables)
        self.assertListEqual(combined["var_name"].values.tolist(), self.input_variables)
        self.assertListEqual(list(combined.dims), ["p", "row", "col", "var_name"])

    def test_get_meta_scalars(self):
        meta_df = get_meta_scalars(self.meta_data)
        assert "masks" not in meta_df.columns
        assert meta_df.index[-1] == meta_df.shape[0] - 1
        return

    def test_min_max_scale(self):
        combined = combine_patch_data(self.input_data, self.input_variables)
        scaled_data, scale_values = min_max_scale(combined)
        inverse_data = min_max_inverse_scale(scaled_data, scale_values)
        self.assertEqual(combined.max(), inverse_data.max())
        self.assertEqual(combined.min(), inverse_data.min())
        self.assertEqual(scaled_data.max(), 1)
        self.assertEqual(scaled_data.min(), 0)

    def test_storm_max_value(self):
        max_values = storm_max_value(self.output_data[self.output_variable[0]], self.meta_data["masks"])
        assert len(max_values.shape) == 1
        assert max_values.max() > 0
