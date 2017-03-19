"""
The :mod:`imblearn.exceptions` module includes all custom warnings and error
classes and functions used across imbalanced-learn.
"""


def raise_isinstance_error(variable_name, possible_type, variable):
    raise ValueError("{} has to be one of {}. Got {} instead.".format(
        variable_name, possible_type, type(variable)))
