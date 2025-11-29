from typing import Any, Tuple, Iterable, Callable
from functools import reduce

from torch.func import grad_and_value

from .functional import FunctionalModel
from .loss_criterion import LossCriterion
from .utils import D

from src.datasets import Examples

Array = Any

"""Code for representing loss functions.

The central flows are given access to the training objective in the form
of a LossFunction, which models the training loss as a black-box function L(w)
which one can take arbitrary derivatives of, but hides everything else.
"""


class LossFunction:
    """A generic loss function."""
    
    def __init__(self, function):
        self.function = function

    def __call__(self, w) -> float:
        """Evaluate the loss function at weights w."""
        return self.function(w)

    def grad_and_value(self, w) -> Tuple[Array, float]:
        """Return the gradient and loss value at weights w.
        
        Returns:
          gradient: Array
          loss: float
        """
        return grad_and_value(self.function)(w)

    def D(self, w, order=1, *vs) -> Array:
        """Compute arbitrary higher-order derivatives.
        
        See documentation for D() in utils.py.
        """
        return D(self.function, w, order, *vs)

    def D_masked(self, w, mask, order=1, *vs) -> Array:
        """Compute arbitrary higher-order derivatives with respect to masked subset of weights.

        See documentation for D() in utils.py.
        """
        raise NotImplementedError


class SupervisedLossFunction(LossFunction):
    """A loss function of the form E[l(f(x), y)].
    
    where
      f is the model function,
      l is the loss criterion, and
      (x, y) are the data.
    """
    
    def __init__(self, model_fn: FunctionalModel, criterion: LossCriterion, batches: Iterable[Examples]):
        """Create the loss function.
        
        Args:
          model_fn (FunctionalModel): the model function f
          criterion (LossCriterion): the loss criterion l
          batches (Iterable[Examples]):  the training dataset, in batches
          
        """
        def batch_loss(w, batch: Examples):
            output = model_fn.apply(w, batch.inputs)
            return criterion(output, batch.labels)

        self.batch_loss = batch_loss
        self.model_fn = model_fn
        self.batches = batches

    def __call__(self, w) -> float:
        """Evaluate the loss function at weights w."""
        def over_batch(batch): # loss over a batch
          return self.batch_loss(w, batch)
        return dataloop(over_batch, self.batches)

    def grad_and_value(self, w) -> Tuple[Array, float]: 
        """Return the gradient and loss value at weights w.
        
        Returns:
          gradient: Array
          loss: float
        """
        def over_batch(batch): # grad_and_value over a batch
          return grad_and_value(self.batch_loss)(w, batch)
        return dataloop(over_batch, self.batches)

    def D(self, w, order=1, *vs) -> Array:
        """Compute arbitrary higher-order derivatives.
        
        See documentation for D() in utils.py.
        """
        def over_batch(batch): # D over a batch
          return D(lambda w_: self.batch_loss(w_, batch), w, order, *vs)
        return dataloop(over_batch, self.batches)

    def D_masked(self, w, mask, order=1, *vs) -> Array:
        """Compute arbitrary higher-order derivatives with respect to masked subset of weights.

        See documentation for D() in utils.py.
        """
        w_masked = w[mask]  # Extract only masked weights

        def masked_loss(w_masked_, batch: Examples):
            w_full = w.clone()
            w_full[mask] = w_masked_
            return self.batch_loss(w_full, batch)

        def over_batch(batch):
            return D(lambda w_: masked_loss(w_, batch), w_masked, order, *vs)

        return dataloop(over_batch, self.batches)


def dataloop(avg_over_batch: Callable[[Examples], Any], batches: Iterable[Examples]):
    """Given averages over batches, compute average over dataset.
    
    Given a function that computes an average of f over an
    arbitrary minbatch, return the average of f over a full dataset.
        
    Args:
      avg_over_batch: a function which computes the average of some function f
        over an arbitrary minibatch.  this could have multiple outputs
      batches: a dataset in batches
    
    Returns:
      the average of f over the dataset
    """
    n_samples = sum(len(batch) for batch in batches)
    return _sum([_scale(avg_over_batch(batch), len(batch) / n_samples) for batch in batches])


# addition and multiplication with tuples

def _scale(x, c):
  """Multiply x by c, handling the case where x is a tuple."""
  return tuple(x_i * c for x_i in x) if isinstance(x, tuple) else x * c
  
def _add(x, y):
  """Add x and y, handling the case where both are tuples."""
  return tuple(x_i + y_i for (x_i, y_i) in zip(x, y)) if isinstance(x, tuple) else x+y

def _sum(xs):
  """Sum x's, handling the case where they are tuples."""
  return reduce(_add, xs)
