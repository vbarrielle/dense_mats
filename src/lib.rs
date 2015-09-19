/*!

This crate contains strided dense matrices implementations.

Matrices are defined as a contiguous piece of memory, the data array,
which is interpreted as a matrix stored in either column major order or row
major order.

*/

extern crate num;

mod mat;
pub mod errors;
pub mod array_like;

pub use mat::{Tensor, MatView, MatViewMut, MatOwned};
pub use mat::{VecView, VecViewMut, VecOwned};

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
