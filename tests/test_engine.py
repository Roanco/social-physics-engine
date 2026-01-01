import pytest
import pandas as pd
import numpy as np
from social_physics_engine.generate_data import GoogleJules

def test_interaction_calculation():
    # Setup small test case
    engine = GoogleJules(num_users=3)
    # Manually set user data to predictable values
    engine.users = pd.DataFrame({
        'id': [0, 1, 2],
        'x': [0.0, 1.0, 0.0],
        'y': [0.0, 0.0, 1.0],
        'influence': [1.0, 1.0, 1.0]
    })

    result = engine.calculate_interactions()

    # Expected interactions:
    # 0-1: dist=1.0, score=1.0
    # 0-2: dist=1.0, score=1.0
    # 1-0: dist=1.0, score=1.0
    # 1-2: dist=sqrt(2), score=1/sqrt(2) approx 0.707
    # 2-0: dist=1.0, score=1.0
    # 2-1: dist=sqrt(2), score=0.707

    # Total pairs: 3*3 - 3(self) = 6
    assert len(result) == 6

    # Check specific values
    row_0_1 = result[(result['user_a'] == 0) & (result['user_b'] == 1)].iloc[0]
    assert np.isclose(row_0_1['score'], 1.0)

    row_1_2 = result[(result['user_a'] == 1) & (result['user_b'] == 2)].iloc[0]
    assert np.isclose(row_1_2['score'], 1.0 / np.sqrt(2))

def test_large_dataset_runs():
    # Just check it doesn't crash on slightly larger data
    engine = GoogleJules(num_users=10)
    result = engine.calculate_interactions()
    assert len(result) == 10 * 9
