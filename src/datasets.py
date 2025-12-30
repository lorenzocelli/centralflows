from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Tuple, Dict, NamedTuple, Literal, Callable, Iterable

import numpy as np
import torch

from datasets import load_dataset as hf_load_dataset
from src.loss_criterion import ce_categorical_loss, ce_binary_loss, mse_categorical_loss, \
    mse_loss, mse_binary_loss, categorical_accuracy, binary_accuracy

"""Datasets that can be used."""

Array = Any


class Examples(NamedTuple):
    """A set of examples.
    
    This could be a whole dataset, or just a batch.
    """
    inputs: Array
    labels: Array
    
    def __len__(self):
        return len(self.inputs)


class Dataset(NamedTuple):
    """A neural network training problem packaged together."""
    
    trainset: Iterable[Examples]
    """Training dataset, as an iterable of batches of the form (inputs, labels)"""
    
    testset: Iterable[Examples]
    """Testing dataset, as an iterable of batches of the form (inputs, labels)"""

    criterion_fn: Callable[[Array, Array], Array]
    """Compute loss over a batch.  Signature: `(preds, labels) -> scalar`"""

    accuracy_fn:  Callable[[Array, Array], Array] | None
    """Compute accuracy over a batch.  Signature: `(preds, labels) -> scalar`"""

    output_dim: int
    """Output dimension for the neural network."""
    
    input_shape: Tuple[int, ...]
    """Shape of the input data."""

    

@dataclass(kw_only=True)
class DatasetBuilder:
    """Abstract parent class for datasets.
    
    Example usage:
      dataset = CIFAR10(**kwargs).load(device=device)
    
    Subclasses should override the following functions:
     - criterion_fn: compute loss criterion over a batch
     - accuracy_fn: compute accuracy over a batch
     - download: optionally download any raw data, which will
         be cached on the system after the first time
     - make: make the dataset
    """
    
    # even though training is full-batch, for memory
    # reasons we can use a smaller, 'ghost' batch size
    # when looping over the data (-1 = full-batch)
    ghost_batch_size: int = -1
    
    # the directory where raw data will be cached
    cache_dir: str = "data_cache"
    
    def get_output_dim(self) -> int:
        pass
    
    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        pass

    def download(self) -> Optional[Dict]:
        """Optionally download 'raw data', which will then be cached.
        
        If this method is implemented, the resulting data will be cached
        on the system so that in the future it can be accessed quickly.
        
        Returns:
          dictionary mapping string to numpy array, holding raw data
        """
        pass


    def make(self, raw_data:Optional[Dict] = None) -> Tuple[Examples, Examples]:
        """Make the dataset.
        
        Args:
          raw_data (optional dict): raw data that has been downloaded or uncached
          device ('cuda'|'cpu'): the device to put the dataset on
        
        Returns:
          the dataset, as (training examples, test examples)
        """
        pass


    def load(self, device:str = None):
        """Load the dataset."""
        # download or uncache the raw data, if there is any
        raw_data = self._get_raw_data() # returns None if there is no raw data
        
        # make the dataset, possibly using the downloaded/uncached raw data
        train, test = self.make(raw_data=raw_data)
        
        # shuffle the data
        train, test = self._shuffle(train), self._shuffle(test)
        
        # convert data from numpy to pytorch
        train, test = self._to_torch(train, device=device), self._to_torch(test, device=device)
        
        # divide data into (ghost) batches
        trainset, testset = self._batch(train), self._batch(test)
        
        # get loss criterion function and accuracy function
        criterion_fn, accuracy_fn = self.get_loss_and_acc()
        
        return Dataset(
            trainset=trainset,
            testset=testset,
            criterion_fn=criterion_fn,
            accuracy_fn=accuracy_fn,
            output_dim=self.get_output_dim(),
            input_shape=trainset[0].inputs.shape[1:]
        )

    def _get_raw_data(self):
        """If cache exists, return it; otherwise try to download.
        
        Returns:
          a Dict[string, np array], if one exists, or else None,
          if this dataset does not have anything downloadable
        """
        
        filename = self.__class__.__name__ + ".npz"
        cache = Path(self.cache_dir) 
        cache.mkdir(parents=True, exist_ok=True)
        file = cache / filename
        if file.exists():
            print("loading data from cache: ", file)
            dataset = dict(np.load(file))
        else:
            dataset = self.download()
            if dataset is not None:
                # save cache
                np.savez(file, **dataset) 
        return dataset

    def _batch(self, examples: Examples) -> List[Examples]:
        """Divide examples into (ghost) batches. """
        batch_sizes = _batch_sizes(len(examples), self.ghost_batch_size)
        input_batches = torch.split(examples.inputs, batch_sizes)
        label_batches = torch.split(examples.labels, batch_sizes)
        return [Examples(inputs_batch, labels_batch) for 
                (inputs_batch, labels_batch) in zip(input_batches, label_batches)]

    def _shuffle(self, examples: Examples) -> Examples:
        """Shuffle examples."""
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(examples))
        return Examples(examples.inputs[idx], examples.labels[idx])
    
    def _to_torch(self, examples: Examples, device=None) -> Examples:
        """Convert from numpy to pytorch."""
        return Examples(
            torch.from_numpy(examples.inputs).to(device),
            torch.from_numpy(examples.labels).to(device)
        )


def _batch_sizes(n: int, batch_size: int) -> List[int]:
    """Divide n examples into `batch_size` groups.
    
    Args:
      n (int): the number of examples
      batch_size (int) the maximum size of each batch
      
    Returns:
      (list of ints): the number of elements in each batch
    """
    if batch_size <= 0:
        return [n]
    n_batch, remainder = divmod(n, batch_size)
    batch_sizes = [batch_size] * n_batch
    if remainder > 0:
        batch_sizes.append(remainder)
    return batch_sizes


CIFAR_MEAN = 255 * np.array([0.49139968, 0.48215827, 0.44653124])
CIFAR_STD = 255 * np.array([0.24703233, 0.24348505, 0.26158768])

@dataclass(kw_only=True)
class CIFAR10(DatasetBuilder):
    """The CIFAR-10 dataset.
    
    This implementation lets you select only a subset of 
    classes and/or examples.
    """
    
    # loss criterion
    criterion: Literal["ce", "mse"]
    
    # overall number of training examples to keep
    n: int
    
    # number of classes to keep
    classes: Optional[int | Literal["binary"]] = 10

    def get_output_dim(self) -> int:
        if self.classes == "binary":
            return 1
        else:
            return self.classes

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        if self.classes == "binary":
            if self.criterion == "ce":
                criterion_fn = ce_binary_loss
            elif self.criterion == "mse":
                criterion_fn = mse_binary_loss
            accuracy_fn = binary_accuracy
        else:
            if self.criterion == "ce":
                criterion_fn = ce_categorical_loss
            elif self.criterion == "mse":
                criterion_fn = mse_categorical_loss
            accuracy_fn = categorical_accuracy
        return criterion_fn, accuracy_fn
            
    def download(self):
        dataset = hf_load_dataset("cifar10") # download from huggingface
        return dict(
            train_x=np.array(dataset["train"]["img"], dtype=np.float32),
            train_y=np.array(dataset["train"]["label"], dtype=np.int32),
            test_x=np.array(dataset["test"]["img"], dtype=np.float32),
            test_y=np.array(dataset["test"]["label"], dtype=np.int32)
        )
        
    def make(self, raw_data):
        classes = 2 if self.classes == 'binary' else self.classes
        
        # code to make a split
        def _make_split(x, y, n_per_class=None):
            # select the desired examples
            idx = []
            for i in range(classes):
                class_indices = np.nonzero(y == i)[0][:n_per_class]
                idx.append(class_indices)
            idx = np.concatenate(idx)
            x, y = x[idx], y[idx]
            
            # normalize channels
            x = (x - CIFAR_MEAN) / CIFAR_STD
            
            # rearrange dimensions according to pytorch convention
            x = x.transpose(0, 3, 1, 2)
            
            # cast to proper dtype
            x, y = x.astype(np.float32), y.astype(np.int64)
            
            return Examples(x, y)
        
        # make the train split
        train_x, train_y = raw_data['train_x'], raw_data['train_y']
        train = _make_split(train_x, train_y, n_per_class=self.n // classes)
        
        # make the test split
        test_x, test_y = raw_data['test_x'], raw_data['test_y'] # for now, keep all test examples
        test = _make_split(test_x, test_y)

        return train, test


@dataclass(kw_only=True)
class SST2(DatasetBuilder):
    criterion: Literal["ce", "mse"]
    n: int
    vocab_size: ClassVar[int] = 30522
    seq_len: ClassVar[int] = 66

    def get_output_dim(self) -> int:
        return 1

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        if self.criterion == "ce":
            criterion_fn = ce_binary_loss
        elif self.criterion == "mse":
            criterion_fn = mse_binary_loss
        accuracy_fn = binary_accuracy
        return criterion_fn, accuracy_fn

    def download(self):
        from transformers import AutoTokenizer

        assert self.n % 2 == 0
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = hf_load_dataset("glue", "sst2")
        train_x = dataset["train"]["sentence"]
        test_x = dataset["test"]["sentence"]
        seqs = tokenizer(
            train_x + test_x,
            padding=True,
            return_tensors="np",
        )["input_ids"]
        train_x = seqs[: len(train_x)]
        test_x = seqs[len(train_x) :]
        train_y = np.array(dataset["train"]["label"])
        test_y = np.array(dataset["test"]["label"])
        
        return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    def make(self, raw_data=None):
        train_x, train_y = raw_data['train_x'], raw_data['train_y']
        test_x, test_y = raw_data['test_x'], raw_data['test_y']

        # TODO comment this - do something to train set
        neg_idx = np.argwhere(train_y == 0)[: self.n // 2, 0]
        pos_idx = np.argwhere(train_y == 1)[: self.n // 2, 0]
        idx = np.concatenate([neg_idx, pos_idx])
        train_x = train_x[idx]
        
        train_x, train_y = train_x.astype(np.int64), train_y.astype(np.int64)
        test_x, test_y = test_x.astype(np.int64), test_y.astype(np.int64)
        
        return Examples(train_x, train_y), Examples(test_x, test_y)
    
    
@dataclass(kw_only=True)
class Classification(DatasetBuilder):
    """SKLearn synthetic classification dataset."""
    
    # number of training examples
    n: int = 100
    
    # number of test examples
    n_test: int = 100
    
    # number of classes
    classes: int = 2
    
    # number of features
    n_features: int = 20
    
    # random seed
    seed: int = 0
     
    def get_output_dim(self) -> int:
        return self.classes

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        return mse_categorical_loss, categorical_accuracy
    
    def make(self, raw_data=None):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=self.n+self.n_test, n_features=self.n_features, n_classes=self.classes, n_informative=5, random_state=self.seed)
        X_train, y_train = X[0:self.n], y[0:self.n]
        X_test, y_test = X[self.n:], y[self.n:]
        return Examples(X_train.astype(np.float32), y_train.astype(np.int64)), Examples(X_test.astype(np.float32), y_test.astype(np.int64))
    
    

@dataclass(kw_only=True)
class Moons(DatasetBuilder):
    """Two moons toy dataset."""
    
    # number of training examples
    n: int = 200
    
    # number of test examples
    n_test: int = 100
    
    # amount of noise
    noise: float = 0.2
    
    # random seed
    seed: int = 10
     
    def get_output_dim(self) -> int:
        return 1

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        return mse_binary_loss, binary_accuracy
    
    def make(self, raw_data=None):
        from sklearn.datasets import make_moons
        X_train, y_train = make_moons(n_samples=self.n, noise=self.noise, random_state=self.seed)
        X_test, y_test = make_moons(n_samples=self.n_test, noise=self.noise, random_state=1000)
        return Examples(X_train.astype(np.float32), y_train.astype(np.int64)), Examples(X_test.astype(np.float32), y_test.astype(np.int64))
    
    
@dataclass(kw_only=True)
class Circles(DatasetBuilder):
    """Circles toy dataset."""
    
    # number of training examples
    n: int = 200
    
    # number of test examples
    n_test: int = 100
    
    # amount of noise
    noise: float = 0.2
    
    # scale factor for inner and outer circle (see sklearn docs)
    factor: float = 0.8
    
    # random seed
    seed: int = 10
    
    def get_output_dim(self) -> int:
        return 1

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        return mse_binary_loss, binary_accuracy

    def make(self, raw_data=None):
        from sklearn.datasets import make_circles
        X_train, y_train = make_circles(n_samples=self.n, noise=self.noise, random_state=self.seed, factor=self.factor)
        X_test, y_test = make_circles(n_samples=self.n_test, noise=self.noise, random_state=self.seed, factor=self.factor)
        return Examples(X_train.astype(np.float32), y_train.astype(np.int64)), Examples(X_test.astype(np.float32), y_test.astype(np.int64))


@dataclass(kw_only=True)
class Sorting(DatasetBuilder):
    """Toy sorting task.
    
    This code is adapted from Karpathy's: https://github.com/karpathy/minGPT/blob/master/demo.ipynb.
    
    """
    
    # the loss criterion
    criterion: Literal["ce", "mse"]
    
    # the number of training examples
    n: int = 1000
    
    # the number of test examples
    n_test: int = 1000
    
    # the size of the vocabulary
    vocab_size: int = 4
    
    # the number of tokens to sort
    length: int = 8
    
    def get_output_dim(self) -> int:
        return self.vocab_size

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        if self.criterion == "ce":
            criterion_fn = ce_categorical_loss
        elif self.criterion == "mse":
            criterion_fn = mse_categorical_loss
        accuracy_fn = categorical_accuracy
        return criterion_fn, accuracy_fn
    
    def make(self, raw_data=None):
        # generate both train and test sequences
        rng = np.random.default_rng(0)
        # Each row is the (unsorted) input to a sorting problem
        inp = rng.integers(0, self.vocab_size, size=(self.n + self.n_test, self.length))
        # Each row is the sorted input
        sol = np.sort(inp, axis=1)
        # Each row is the concatenation of the unsorted and sorted sequences
        cat = np.concatenate((inp, sol), axis=1)
        # Inputs and labels for next-token prediction
        x, y = cat[:, :-1].copy(), cat[:, 1:].copy()

        # -1 means mask - these positions won't be used to compute the loss
        y[:, : self.length - 1] = -1

        # separate train and test
        train_x, test_x = x[:self.n], x[self.n :]
        train_y, test_y = y[:self.n], y[self.n :]

        train, test = Examples(train_x, train_y), Examples(test_x, test_y)
        return train, test
        
        
@dataclass(kw_only=True)
class Copying(DatasetBuilder):
    """Toy copying task."""
    
    # the loss criterion
    criterion: Literal["ce", "mse"]
    
    # the number of training examples
    n: int = 1000
    
    # the number of test examples
    n_test: int = 1000

    # the size of the vocabulary
    vocab_size: int = 4
    
    # the number of tokens to copy
    length: int = 8
    
    def get_output_dim(self) -> int:
        return self.vocab_size

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        if self.criterion == "ce":
            criterion_fn = ce_categorical_loss
        elif self.criterion == "mse":
            criterion_fn = mse_categorical_loss
        accuracy_fn = categorical_accuracy
        return criterion_fn, accuracy_fn

    def make(self, raw_data=None):
        rng = np.random.default_rng(0)
        
        # generate both train and test sequences
        inp = rng.integers(0, self.vocab_size, size=(self.n + self.n_test, self.length))
        cat = np.concatenate((inp, inp), axis=1)
        # Inputs and labels for next-token prediction
        x, y = cat[:, :-1].copy(), cat[:, 1:].copy()
        
        # -1 means mask - these positions won't be used to compute the loss
        y[:, : self.length - 1] = -1
        
        # separate train from test
        train_x, test_x = x[: self.n], x[self.n :]
        train_y, test_y = y[: self.n], y[self.n :]
        
        return Examples(train_x, train_y), Examples(test_x, test_y)


@dataclass(kw_only=True)
class Regression(DatasetBuilder):
    """Basic linear regression task."""

    # the number of training examples
    n: int = 10000

    # the number of test examples
    n_test: int = 1000

    # size of each example
    n_features: int = 500

    # whether to unevenly scale features (makes the problem harder)
    uneven_scale: bool = True

    def get_output_dim(self) -> int:
        return 1

    def get_loss_and_acc(self) -> Tuple[Callable, Callable]:
        return mse_loss, None

    def make(self, raw_data=None):
        print(self.n)
        rng = np.random.default_rng(seed=42)
        X = torch.from_numpy(
            rng.standard_normal((self.n + self.n_test, self.n_features)).astype(
                np.float32
            )
        )

        if self.uneven_scale:
            scales = torch.logspace(0, 3, self.n_features)
            X = X * scales

        w = torch.from_numpy(
            rng.standard_normal((self.n_features, 1)).astype(np.float32)
        )

        y = torch.mm(X, w) + 0.1 * torch.from_numpy(
            rng.standard_normal((self.n + self.n_test, 1)).astype(np.float32)
        )
        X = X.numpy()
        y = y.numpy()

        # separate train from test
        train_x, test_x = X[: self.n], X[self.n :]
        train_y, test_y = y[: self.n], y[self.n :]

        return Examples(train_x, train_y), Examples(test_x, test_y)
