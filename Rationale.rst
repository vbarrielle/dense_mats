Rationale behind dense_mats traits
==================================


This file explores the intended usage and benefits of the traits provided
by this crate.


Abstracting over matrices
-------------------------

The current rust ecosystem already contains some dense matrices implementations.
However, currently interoperation between those implementations is cumbersome.
It is unreasonable for each crate to add dependencies to every other crate that
also implements dense matrices for the sake of interoperability.

My take on that is that a single crate needs to provide the required
abstractions for all dense matrices implementations to interoperate. This crate
is an attempt at solving this problem.


Taking a foreign matrix as input
--------------------------------

Efficient C or Fortran matrices are implemented as an indexing abstraction over
a contiguous piece of memory. This indexing abstraction relies on knowing the
shape of the matrix, and knowing the strides of the matrix. The strides inform
an indexing algorithm of the distance in the data array between two consecutive
elements along a matrix' dimension. For instance, a row stride of 1 means
accessing all elements in a row can be done by taking a slice of the data array.

Matrices with a row (resp. col) stride of 1 are said to have a row (resp. col)
contiguous storage.

Therefore, a trait representing a view on a matrix can be:

.. code-block:: rust

 pub trait DenseMatView<N> {

     fn rows(&self) -> usize;
 
     fn cols(&self) -> usize;
 
     fn strides(&self) -> [usize; 2];
 
     fn data(&self) -> &[N];
 }

 pub trait DenseMatViewMut<N> : DenseMatView<N> {
     fn data_mut(&self) -> &mut [N];
 }

To my knowledge, these traits represent the bare minimum for representing a
view of a dense matrix. A library author implementing its own matrix type could
abstract over any matrix implementing `DenseMatView<N>` by having his own
concrete implementation of a matrix view. This can be done as:


.. code-block:: rust

  pub struct DMat<N, Storage> where Storage: Deref<Target=[N]> {
      data: Storage,
      rows: usize,
      cols: usize,
      strides: [usize; 2],
  }

This dense matrix structure can represent either an owned matrix, or a matrix
view. Such a matrix can be created from a `DenseMatView<N>` implementor:


.. code-block:: rust

  impl<'a, N: 'a> DMat<N, &'a [N]> {
  
      pub fn wrap_view<M: 'a + DenseMatView<N>>(m: &'a M) -> DMatView<'a, N> {
          DMat {
              data: m.data(),
              rows: m.rows(),
              cols: m.cols(),
              strides: m.strides(),
          }
      }
  }


Converting from a foreign matrix
--------------------------------

What happens if we use another library for some linear algebra operation, but
want to get a matrix of our own data type as a result? We need a way to convert
from a foreign matrix while taking ownership of its data array. This is provided
by the DenseMatOwned trait:

.. code-block:: rust

  pub trait DenseMatOwned<N> : DenseMatView<N> {
  
      fn into_data(self) -> Vec<N>;
  }

Using this trait, we can convert an owned result into our `DMat<N>` type:

.. code-block:: rust

  impl<N> DMat<N, Vec<N>> {
    pub fn from_owned<Mat: DenseMatOwned<N>>(m: Mat) -> DMatOwned<N> {
        let rows = m.rows();
        let cols = m.cols();
        let strides = m.strides();
        DMat {
            data: m.into_data(),
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }
  }


Conclusion
----------

I hope providing these traits is enough for dense matrix interoperability.
While some useful methods (indexing, iteration) could be provided automatically
by the trait, I feel this would not be appropriate, as matrix library authors
are probably in a better situation to implement these coherently with the rest
of their API.
