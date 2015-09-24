/*!

This crate contains strided dense tensor implementations.

Tensors are defined as a contiguous piece of memory, the data array,
which is interpreted using its stride information.
*/

extern crate num;

pub mod tensor;
pub mod errors;
pub mod array_like;

pub use tensor::{Tensor, MatView, MatViewMut, MatOwned};
pub use tensor::{VecView, VecViewMut, VecOwned};

/// Describe the storage order of a matrix.
#[derive(PartialEq, Debug)]
pub enum StorageOrder {
    /// C storage order, ie column major storage for matrices
    /// The dimensions are sorted in decreasing order of variation
    F,
    /// C storage order, ie row major storage for matrices
    /// The dimensions are sorted in increasing order of variation
    C,
    /// Nothing special can be assumed about the order
    Unordered
}
