import pytest

from project2.fcn_model import fcn_32s


def test_fcn32s():
    input_shape = (3, 400, 400)
    model = fcn_32s(input_shape, 2)

    model.summary()

    assert 0



if __name__ == '__main__':
    pytest.main([__file__])