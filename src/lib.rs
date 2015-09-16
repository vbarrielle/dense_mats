/*!

This crate contains strided dense matrices implementations.

The goal is to enable interoperability between various linear algebra
libraries, by providing a common definition.

Matrices are thus defined as a contiguous piece of memory, the data array,
which is interpreted as a matrix stored in either column major order or row
major order.

*/

extern crate num;

mod mat;
pub mod errors;

pub use mat::{StridedMat, StridedMatView, StridedMatViewMut, StridedMatOwned};
pub use mat::{StridedVec, StridedVecView, StridedVecViewMut, StridedVecOwned};

/// Describe the storage order of a matrix.
pub enum StorageOrder {
    /// Column major storage 
    ColMaj,
    /// Row major storage 
    RowMaj
}
