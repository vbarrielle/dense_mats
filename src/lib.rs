/*!

This crate contains strided dense matrices implementations.

Matrices are defined as a contiguous piece of memory, the data array,
which is interpreted as a matrix stored in either column major order or row
major order.

*/

extern crate num;
extern crate array_like;

mod mat;
pub mod errors;

pub use mat::{Tensor, MatView, MatViewMut, MatOwned};
pub use mat::{VecView, VecViewMut, VecOwned};

/// Describe the storage order of a matrix.
pub enum StorageOrder {
    /// Column major storage
    ColMaj,
    /// Row major storage
    RowMaj
}
