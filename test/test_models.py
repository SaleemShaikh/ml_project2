import pytest

from project2.cnn_v1 import leNet


def test_lenet():
    model = leNet((3,24,24))
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()


if __name__ == '__main__':
    pytest.main([__file__])