import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("IIM_model")


def expand_beta_by_year(beta_initial: dict, growth: float, years: list) -> dict:
    first_year = years[0]
    base_values = beta_initial[1]
    return {
        year: (
            list(base_values)
            if year == first_year
            else [v + np.log(growth) * (year - first_year) for v in base_values]
        )
        for year in years
    }

def get_tolls(initial_toll: float, growth: float, years: list) -> dict:

    first_year = years[0]

    return {
        year: (
            initial_toll
            if year == first_year
            else first_year * (1 + growth) ^ (year - first_year)
        )
        for year in years
    }

def initialize_seg_params(path: str) -> pd.DataFrame:
    logger.info(f"Loading segment params from: {path}")
    df = pd.read_excel(path, index_col=False)
    logger.debug(f"Loaded {len(df)} rows from {path}")
    return df

def initialize_model_df(seg_params: pd.DataFrame, ):
    pass
