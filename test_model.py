import numpy as np,pytest
from model import *
def test_seed():assert np.array_equal(generate(10)[0],generate(10)[0])
def test_invalid():
 with pytest.raises(ValueError):features(dict(amount=-1,days_since_start=1,injury=0,description_length=20))
