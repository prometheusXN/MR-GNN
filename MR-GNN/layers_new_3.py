"""Custom Keras Layers.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import regularizers
from deepchem.nn import model_ops
from deepchem.nn.copy import Layer
from deepchem.nn.copy import Input
from deepchem.nn.copy import Dense
from deepchem.nn.copy import Dropout

def affine(x, W, b):
  '''
  print(x_shape)
  mini_block = int(4)
  num = 128

  i=0
  x1 = x[:,i*mini_block:(i+1)*mini_block]
  w1 = W[i*mini_block:(i+1)*mini_block,i*mini_block:(i+1)*mini_block]
  y = tf.matmul(x1,w1)
  for i in range(1,int(num / mini_block)):
    x1 = x[:,i*mini_block:(i+1)*mini_block]
    w1 = W[i*mini_block:(i+1)*mini_block,i*mini_block:(i+1)*mini_block]
    y1 = tf.matmul(x1,w1)
    y = tf.concat([y,y1],1)
  y = y + b
  return y
  '''

  return tf.matmul(x, W) + b


def sum_neigh(atoms, deg_adj_lists, max_deg):
  """Store the summed atoms by degree"""
  deg_summed = max_deg * [None]

  # Tensorflow correctly processes empty lists when using concat
  for deg in range(1, max_deg + 1):
    gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
    # Sum along neighbors as well as self, and store
    summed_atoms = tf.reduce_sum(gathered_atoms, 1)
    deg_summed[deg - 1] = summed_atoms

  return deg_summed

def graph_gather(atoms, membership_placeholder, batch_size):
  """
  Parameters
  ----------
  atoms: tf.Tensor
    Of shape (n_atoms, n_feat)
  membership_placeholder: tf.Placeholder
    Of shape (n_atoms,). Molecule each atom belongs to.
  batch_size: int
    Batch size for deep model.

  Returns
  -------
  tf.Tensor
    Of shape (batch_size, n_feat)
  """

  # WARNING: Does not work for Batch Size 1! If batch_size = 1, then use reduce_sum!
  assert batch_size > 1, "graph_gather requires batches larger than 1"

  # Obtain the partitions for each of the molecules
  activated_par = tf.dynamic_partition(atoms, membership_placeholder,
                                       batch_size)

  # Sum over atoms for each molecule
  sparse_reps = [
      tf.reduce_sum(activated, 0, keep_dims=True) for activated in activated_par
  ]

  # Get the final sparse representations
  sparse_reps = tf.concat(axis=0, values=sparse_reps)

  return sparse_reps

def graph_conv(atoms, deg_adj_lists, deg_slice, max_deg, min_deg, W_list,
               b_list, membership, batch_size):
    """Core tensorflow function implementing graph convolution

    Parameters
    ----------
    atoms: tf.Tensor
    Should be of shape (n_atoms, n_feat)
    deg_adj_lists: list
    Of length (max_deg+1-min_deg). The deg-th element is a list of
    adjacency lists for atoms of degree deg.
    deg_slice: tf.Tensor
    Of shape (max_deg+1-min_deg,2). Explained in GraphTopology.
    max_deg: int
    Maximum degree of atoms in molecules.
    min_deg: int
    Minimum degree of atoms in molecules
    W_list: list
    List of learnable weights for convolution.
    b_list: list
    List of learnable biases for convolution.

    Returns
    -------
    tf.Tensor
    Of shape (n_atoms, n_feat)
    """
    W = iter(W_list)
    b = iter(b_list)

    #Sum all neighbors using adjacency matrix
    deg_summed = sum_neigh(atoms, deg_adj_lists, max_deg)

    # Get collection of modified atom features
    new_rel_atoms_collection = (max_deg + 1 - min_deg) * [None]

    new_gather_atoms_collection = (max_deg + 1 - min_deg) * [None]

    for deg in range(1, max_deg + 1):
        # Obtain relevant atoms for this degree
        rel_atoms = deg_summed[deg-1]

        # Get self atoms
        begin = tf.stack([deg_slice[deg - min_deg, 0], 0])
        size = tf.stack([deg_slice[deg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        # Apply hidden affine to relevant atoms and append
        rel_out = affine(rel_atoms, next(W), next(b))
        self_out = affine(self_atoms, next(W), next(b))
        # out_gather = affine(self_atoms, next(W), next(b))

        out = rel_out + self_out

        new_rel_atoms_collection[deg - min_deg] = out
        # new_gather_atoms_collection[deg - min_deg] = out_gather


  # Determine the min_deg=0 case
    if min_deg == 0:
        deg = 0

        begin = tf.stack([deg_slice[deg - min_deg, 0], 0])
        size = tf.stack([deg_slice[deg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        # Only use the self layer
        out = affine(self_atoms, next(W), next(b))
        # out_gather = affine(self_atoms, next(W), next(b))

        new_rel_atoms_collection[deg - min_deg] = out
        # new_gather_atoms_collection[deg - min_deg] = out_gather

    for dg in range(1, max_deg + 1):
        begin = tf.stack([deg_slice[dg - min_deg, 0], 0])
        size = tf.stack([deg_slice[dg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        out_gather = affine(self_atoms, next(W), next(b))
        new_gather_atoms_collection[dg - min_deg] = out_gather

    if min_deg == 0:
        dg = 0

        begin = tf.stack([deg_slice[dg - min_deg, 0], 0])
        size = tf.stack([deg_slice[dg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        # Only use the self layer
        out_gather = affine(self_atoms, next(W), next(b))

        new_gather_atoms_collection[dg - min_deg] = out_gather


    gather_atoms = tf.concat(axis=0, values=new_gather_atoms_collection)
    atom_gather = graph_gather(gather_atoms, membership, batch_size)
    gather = []
    activated_atoms = tf.concat(axis=0, values=new_rel_atoms_collection)
    # activated_par = tf.dynamic_partition(activated_atoms, membership, batch_size)
    # #tf.shape(gather_atoms)[0]
    # for i in range(64):
    #     j = tf.concat(axis=0, values=[[atom_gather[i]], activated_par[i]])
    #     gather.append(tf.reduce_max(j, 0))

    # gathered_atoms = tf.concat(axis=1, values=[atom_gather, ])
    # gather_atom = tf.reduce_mean(gather_atoms, 0)
  # Combine all atoms back into the list

    return activated_atoms, atom_gather

def gather_node(activated_atoms, membership, batch_size):
    gather = []
    activated_par = tf.dynamic_partition(activated_atoms, membership, batch_size)
    # tf.shape(gather_atoms)[0]
    for i in range(batch_size):
        j = tf.concat(axis=0, values=[activated_par[i]])
        gather.append(tf.reduce_max(j, 0))

    return gather


class Dense(Layer):
  """Just your regular densely-connected NN layer.
  TODO(rbharath): Make this functional in deepchem
  Example:
  >>> import deepchem as dc
  >>> # as first layer in a sequential model:
  >>> model = dc.models.Sequential()
  >>> model.add(dc.nn.Input(shape=16))
  >>> model.add(dc.nn.Dense(32))
  >>> # now the model will take as input arrays of shape (*, 16)
  >>> # and output arrays of shape (*, 32)
  >>> # this is equivalent to the above:
  >>> model = dc.models.Sequential()
  >>> model.add(dc.nn.Input(shape=16))
  >>> model.add(dc.nn.Dense(32))
  Parameters
  ----------
  output_dim: int > 0.
  init: name of initialization function for the weights of the layer
  activation: name of activation function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: a(x) = x).
  W_regularizer: (eg. L1 or L2 regularization), applied to the main weights matrix.
  b_regularizer: instance of regularize applied to the bias.
  activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
    applied to the network output.
  bias: whether to include a bias
    (i.e. make the layer affine rather than linear).
  input_dim: dimensionality of the input (integer). This argument
    (or alternatively, the keyword argument `input_shape`)
    is required when using this layer as the first layer in a model.
  # Input shape
    nD tensor with shape: (nb_samples, ..., input_dim).
    The most common situation would be
    a 2D input with shape (nb_samples, input_dim).
  # Output shape
    nD tensor with shape: (nb_samples, ..., output_dim).
    For instance, for a 2D input with shape `(nb_samples, input_dim)`,
    the output would have shape `(nb_samples, output_dim)`.
  """

  def __init__(self,
               output_dim,
               input_dim,
               init='glorot_uniform',
               activation="relu",
               bias=True,
               **kwargs):
    self.init = initializations.get(init)
    self.activation = activations.get(activation)
    self.output_dim = output_dim
    self.input_dim = input_dim

    self.bias = bias

    input_shape = (self.input_dim,)
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(Dense, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.W = self.add_weight(
        (self.input_dim, self.output_dim),
        initializer=self.init,
        name='{}_W'.format(self.name))
    self.b = self.add_weight(
        (self.output_dim,), initializer='zero', name='{}_b'.format(self.name))

  def __call__(self, x):
    output = model_ops.dot(x, self.W)
    if self.bias:
      output += self.b
    return output


def graph_conv1(atoms, deg_adj_lists, deg_slice, max_deg, min_deg, W_list,
               b_list, membership, batch_size):
    """Core tensorflow function implementing graph convolution

    Parameters
    ----------
    atoms: tf.Tensor
    Should be of shape (n_atoms, n_feat)
    deg_adj_lists: list
    Of length (max_deg+1-min_deg). The deg-th element is a list of
    adjacency lists for atoms of degree deg.
    deg_slice: tf.Tensor
    Of shape (max_deg+1-min_deg,2). Explained in GraphTopology.
    max_deg: int
    Maximum degree of atoms in molecules.
    min_deg: int
    Minimum degree of atoms in molecules
    W_list: list
    List of learnable weights for convolution.
    b_list: list
    List of learnable biases for convolution.

    Returns
    -------
    tf.Tensor
    Of shape (n_atoms, n_feat)
    """
    W = iter(W_list)
    b = iter(b_list)

    #Sum all neighbors using adjacency matrix
    deg_summed = sum_neigh(atoms, deg_adj_lists, max_deg)

    # Get collection of modified atom features
    new_rel_atoms_collection = (max_deg + 1 - min_deg) * [None]

    new_gather_atoms_collection = (max_deg + 1 - min_deg) * [None]

  # Determine the min_deg=0 case

    for dg in range(1, max_deg + 1):
        begin = tf.stack([deg_slice[dg - min_deg, 0], 0])
        size = tf.stack([deg_slice[dg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        out_gather = affine(self_atoms, next(W), next(b))
        new_gather_atoms_collection[dg - min_deg] = out_gather

    if min_deg == 0:
        dg = 0

        begin = tf.stack([deg_slice[dg - min_deg, 0], 0])
        size = tf.stack([deg_slice[dg - min_deg, 1], -1])
        self_atoms = tf.slice(atoms, begin, size)

        # Only use the self layer
        out_gather = affine(self_atoms, next(W), next(b))

        new_gather_atoms_collection[dg - min_deg] = out_gather


    gather_atoms = tf.concat(axis=0, values=new_gather_atoms_collection)
    atom_gather = graph_gather(gather_atoms, membership, batch_size)
    # gather_atom = tf.reduce_mean(gather_atoms, 0)
  # Combine all atoms back into the list
  #   activated_atoms = tf.concat(axis=0, values=new_rel_atoms_collection)

    return atom_gather


class GraphConv_and_gather(Layer):
    """"Performs a graph convolution.

    Note this layer expects the presence of placeholders defined by GraphTopology
    and expects that they follow the ordering provided by
    GraphTopology.get_input_placeholders().
    """

    def __init__(self,
               nb_filter,
               n_atom_features,
               batch_size,
               init='glorot_uniform',
               activation='linear',
               dropout=None,
               max_deg=10,
               min_deg=0,
               **kwargs):
        """
        Parameters
        ----------
        nb_filter: int
          Number of convolutional filters.
        n_atom_features: int
          Number of features listed per atom.
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied after convolution.
        dropout: float, optional
          Dropout probability.
        max_deg: int, optional
          Maximum degree of atoms in molecules.
        min_deg: int, optional
          Minimum degree of atoms in molecules.
        """
        warnings.warn("The dc.nn.GraphConv is "
                      "deprecated. Will be removed in DeepChem 1.4. "
                      "Will be replaced by dc.models.tensorgraph.layers.GraphConv",
                      DeprecationWarning)
        super(GraphConv_and_gather, self).__init__(**kwargs)

        self.init = initializations.get(init)  # Set weight initialization
        self.activation = activations.get(activation)  # Get activations
        self.nb_filter = nb_filter  # Save number of filters
        self.dropout = dropout  # Save dropout params
        self.max_deg = max_deg
        self.min_deg = min_deg
        self.batch_size = batch_size
        # Is there a solid explanation here?
        self.nb_affine = 3 * max_deg + (2 - min_deg)
        self.n_atom_features = n_atom_features
        n_atom_features = self.n_atom_features

        self.beta_init = initializations.get('zero')
        self.gamma_init = initializations.get('one')
        self.epsilon = 1e-5
        self.momentum = 0.99
        self.gamma_regularizer = regularizers.get(None)
        self.beta_regularizer = regularizers.get(None)


        # Generate the nb_affine weights and biases
        self.W_list = [
            self.init([n_atom_features, self.nb_filter])
            for k in range(self.nb_affine)
            ]
        self.b_list = [
            model_ops.zeros(shape=[
                self.nb_filter,
            ]) for k in range(self.nb_affine)
            ]

        self.trainable_weights = self.W_list + self.b_list

    def build_bn(self, input_shape):
        shape = (input_shape[-1],)

        self.gamma = self.add_weight(
            shape,
            initializer=self.gamma_init,
            regularizer=self.gamma_regularizer,
            name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(
            shape,
            initializer=self.beta_init,
            regularizer=self.beta_regularizer,
            name='{}_beta'.format(self.name))
        # Not Trainable
        self.running_mean = self.add_weight(
            shape, initializer='zero', name='{}_running_mean'.format(self.name))
        # Not Trainable
        self.running_std = self.add_weight(
            shape, initializer='one', name='{}_running_std'.format(self.name))

    def get_output_shape_for(self, input_shape):
        """Output tensor shape produced by this layer."""
        atom_features_shape = input_shape[0]
        assert len(atom_features_shape) == 2, \
                "MolConv only takes 2 dimensional tensors for x"
        n_atoms = atom_features_shape[0]
        return (n_atoms, self.nb_filter)

    def call(self, x, mask=None):
        """Execute this layer on input tensors.

        This layer is meant to be executed on a Graph. So x is expected to
        be a list of placeholders, with the first placeholder the list of
        atom_features (learned or input) at this level, the second the deg_slice,
        the third the membership, and the remaining the deg_adj_lists.

        Visually

        x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

        Parameters
        ----------
        x: list
          list of Tensors of form described above.
        mask: bool, optional
          Ignored. Present only to shadow superclass call() method.

        Returns
        -------
        atom_features: tf.Tensor
          Of shape (n_atoms, nb_filter)
        """
        # Add trainable weights
        # self.build()

        # Extract atom_features
        atom_features_ori = x[0]

        # Extract graph topology
        deg_slice, membership, deg_adj_lists = x[1], x[2], x[3:]
        training = x[-2]

        # Perform the mol conv
        atom_features, gather_feature = graph_conv(atom_features_ori, deg_adj_lists, deg_slice,
                                   self.max_deg, self.min_deg, self.W_list,
                                   self.b_list, membership, self.batch_size)

        atom_features = self.activation(atom_features)
        gather_feature = self.activation(gather_feature)

        xx = atom_features
        yy = gather_feature
        if not isinstance(xx, list):
            input_shape = model_ops.int_shape(xx)
        else:
            xx = xx[0]
            input_shape = model_ops.int_shape(xx)
        self.build_bn(input_shape)

        m = model_ops.mean(xx, axis=-1, keepdims=True)
        std = model_ops.sqrt(
            model_ops.var(xx, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (xx - m) / (std + self.epsilon)
        x_normed = self.gamma * x_normed + self.beta
        m_1 = model_ops.mean(yy, axis=-1, keepdims=True)
        std_1 = model_ops.sqrt(
            model_ops.var(yy, axis=-1, keepdims=True) + self.epsilon)
        y_normed = (yy - m_1) / (std_1 + self.epsilon)
        y_normed = self.gamma * y_normed + self.beta

        atom_features = x_normed
        gather_norm = gather_node(x_normed, membership, self.batch_size)
        gather = tf.convert_to_tensor(gather_norm, dtype=tf.float32)

        if self.dropout is not None:
            atom_features = training * tf.nn.dropout(atom_features, 1-self.dropout) + (1 -training) * atom_features
            gather = training * tf.nn.dropout(gather_feature, 1-self.dropout) + (1 -training) * gather_feature
        return atom_features, y_normed, gather

class Gather1(Layer):
    """"Performs a graph convolution.

    Note this layer expects the presence of placeholders defined by GraphTopology
    and expects that they follow the ordering provided by
    GraphTopology.get_input_placeholders().
    """

    def __init__(self,
               nb_filter,
               n_atom_features,
               batch_size,
               init='glorot_uniform',
               activation='linear',
               dropout=None,
               max_deg=10,
               min_deg=0,
               **kwargs):
        """
        Parameters
        ----------
        nb_filter: int
          Number of convolutional filters.
        n_atom_features: int
          Number of features listed per atom.
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied after convolution.
        dropout: float, optional
          Dropout probability.
        max_deg: int, optional
          Maximum degree of atoms in molecules.
        min_deg: int, optional
          Minimum degree of atoms in molecules.
        """
        warnings.warn("The dc.nn.GraphConv is "
                      "deprecated. Will be removed in DeepChem 1.4. "
                      "Will be replaced by dc.models.tensorgraph.layers.GraphConv",
                      DeprecationWarning)
        super(Gather1, self).__init__(**kwargs)

        self.init = initializations.get(init)  # Set weight initialization
        self.activation = activations.get(activation)  # Get activations
        self.nb_filter = nb_filter  # Save number of filters
        self.dropout = dropout  # Save dropout params
        self.max_deg = max_deg
        self.min_deg = min_deg
        self.batch_size = batch_size
        # Is there a solid explanation here?
        self.nb_affine = max_deg + (1 - min_deg)
        self.n_atom_features = n_atom_features

    def build(self):
        """"Construct internal trainable weights.

        n_atom_features should provide the number of features per atom.

        Parameters
        ----------
        n_atom_features: int
          Number of features provied per atom.
        """
        n_atom_features = self.n_atom_features

        # Generate the nb_affine weights and biases
        self.W_list = [
            self.init([n_atom_features, self.nb_filter])
            for k in range(self.nb_affine)
        ]
        self.b_list = [
            model_ops.zeros(shape=[
                self.nb_filter,
            ]) for k in range(self.nb_affine)
        ]

        self.trainable_weights = self.W_list + self.b_list

    def get_output_shape_for(self, input_shape):
        """Output tensor shape produced by this layer."""
        atom_features_shape = input_shape[0]
        assert len(atom_features_shape) == 2, \
                "MolConv only takes 2 dimensional tensors for x"
        n_atoms = atom_features_shape[0]
        return (n_atoms, self.nb_filter)

    def call(self, x, mask=None):
        """Execute this layer on input tensors.

        This layer is meant to be executed on a Graph. So x is expected to
        be a list of placeholders, with the first placeholder the list of
        atom_features (learned or input) at this level, the second the deg_slice,
        the third the membership, and the remaining the deg_adj_lists.

        Visually

        x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

        Parameters
        ----------
        x: list
          list of Tensors of form described above.
        mask: bool, optional
          Ignored. Present only to shadow superclass call() method.

        Returns
        -------
        atom_features: tf.Tensor
          Of shape (n_atoms, nb_filter)
        """
        # Add trainable weights
        self.build()

        # Extract atom_features
        atom_features = x[0]

        # Extract graph topology
        deg_slice, membership, deg_adj_lists = x[1], x[2], x[3:]
        training = x[-2]

        # Perform the mol conv
        gather_feature = graph_conv1(atom_features, deg_adj_lists, deg_slice,
                                   self.max_deg, self.min_deg, self.W_list,
                                   self.b_list, membership, self.batch_size)

        # atom_features = self.activation(atom_features)
        gather_feature = self.activation(gather_feature)

        if self.dropout is not None:
            # atom_features = training * tf.nn.dropout(atom_features, 1-self.dropout) + (1 -training) * atom_features
            gather_feature = training * tf.nn.dropout(gather_feature, 1-self.dropout) + (1 -training) * gather_feature
        return gather_feature