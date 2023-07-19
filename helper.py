from enum import Enum


class NormOptions(Enum):

    NO_ACTION = 1
    STD_NORM = 2
    MINMAX_NORM = 3


def invoke_normalization(value, statistics, normalize):
  if normalize == NormOptions.NO_ACTION:
      return value
  elif normalize == NormOptions.STD_NORM:
    return (value - statistics['avg']) / statistics['std']
  elif normalize == NormOptions.MINMAX_NORM:
    return (value - statistics['min']) / (statistics['max'] - statistics['min'])