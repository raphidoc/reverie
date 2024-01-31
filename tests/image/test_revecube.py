import pytest
from reverie.image.revecube import ReveCube
import pandas as pd
import os


@pytest.fixture
def reve_cube():
    # Assuming the ReveCube class has a default constructor
    return ReveCube()


def test_extract_pixel(reve_cube):
    # Assuming you have a matchup_file for testing
    matchup_file = "path_to_your_test_matchup_file.csv"
    var_name = ["var1", "var2"]  # replace with your variable names
    window_size = 1  # replace with your window size

    # Ensure the matchup file exists
    assert os.path.exists(matchup_file)

    # Call the extract_pixel method
    result = reve_cube.extract_pixel(matchup_file, var_name, window_size)

    # Check the type of the result
    assert isinstance(result, pd.DataFrame)

    # Add more assertions here based on what you expect the output to be
