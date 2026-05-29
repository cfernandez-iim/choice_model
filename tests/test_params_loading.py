import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from IIM_core_model.utils.params_loading import expand_beta_by_year, initialize_seg_params

BETA_INITIAL = {
    "Period": ["AM", "MD", "PM", "NT"],
    1: [3.3499, 3.3532, 3.5185, 3.5218],
    2: [0.715, 0.708, 0.592, 0.708],
}
YEARS = [2025, 2032, 2040, 2050]
GROWTH = 1.02


class TestExpandBetaByYear:
    def test_returns_all_years(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, YEARS)
        assert set(result.keys()) == set(YEARS)

    def test_first_year_is_copy_of_base(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, YEARS)
        assert result[2025] == list(BETA_INITIAL[1])

    def test_first_year_not_mutated_by_base(self):
        template = {"Period": ["AM"], 1: [1.0, 2.0], 2: [0.5]}
        result = expand_beta_by_year(template, 1.05, [2025, 2030])
        original = list(template[1])
        result[2025][0] = 999
        assert template[1] == original

    def test_subsequent_year_formula(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, YEARS)
        log_g = math.log(GROWTH)
        for year in YEARS[1:]:
            expected = [v + log_g * (year - 2025) for v in BETA_INITIAL[1]]
            assert result[year] == pytest.approx(expected)

    def test_growth_1_leaves_values_unchanged(self):
        result = expand_beta_by_year(BETA_INITIAL, 1.0, YEARS)
        base = list(BETA_INITIAL[1])
        for year in YEARS:
            assert result[year] == pytest.approx(base)

    def test_single_year_list(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, [2025])
        assert result == {2025: list(BETA_INITIAL[1])}

    def test_output_length_matches_base_values(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, YEARS)
        for year, values in result.items():
            assert len(values) == len(BETA_INITIAL[1])

    def test_key_1_used_not_key_2(self):
        result = expand_beta_by_year(BETA_INITIAL, GROWTH, [2025])
        assert result[2025] == list(BETA_INITIAL[1])
        assert result[2025] != list(BETA_INITIAL[2])


class TestInitializeSegParams:
    def _make_csv(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_returns_dataframe(self):
        path = self._make_csv("a,b\n1,2\n3,4\n")
        try:
            result = initialize_seg_params(path)
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(path)

    def test_correct_shape(self):
        path = self._make_csv("a,b,c\n1,2,3\n4,5,6\n")
        try:
            result = initialize_seg_params(path)
            assert result.shape == (2, 3)
        finally:
            os.unlink(path)

    def test_correct_columns(self):
        path = self._make_csv("seg,value\n1,0.5\n")
        try:
            result = initialize_seg_params(path)
            assert list(result.columns) == ["seg", "value"]
        finally:
            os.unlink(path)

    def test_correct_values(self):
        path = self._make_csv("x,y\n10,20\n30,40\n")
        try:
            result = initialize_seg_params(path)
            assert result["x"].tolist() == [10, 30]
            assert result["y"].tolist() == [20, 40]
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            initialize_seg_params("nonexistent_file.csv")
