/*!

This crate contains traits to abstract over dense matrices.

The goal is to enable interoperability between various linear algebra
libraries, by providing a common definition.

Matrices are thus defined as a contiguous piece of memory, the data array,
which is interpreted as a matrix stored in either column major order or row
major order.

*/

/// Describe the storage order of a matrix.
pub enum StorageOrder {
    /// Column major storage 
    ColMaj,
    /// Row major storage 
    RowMaj
}

/// Represents a view into a dense matrix
pub trait DenseMatView<N> {
    
    /// The number of rows of the matrix
    fn rows(&self) -> usize;

    /// The number of cols of the matrix
    fn cols(&self) -> usize;

    /// The number of rows and cols of the matrix
    fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }

    /// The strides of the matrix.
    /// 
    /// self.strides()[0] gives the number of elements that must be skipped
    /// into self.data() to get to the element of the next row with the same
    /// column.
    /// self.strides()[1] gives the number of elements that must be skipped
    /// into self.data() to get to the element of the next column with the same
    /// row.
    /// 
    /// For a row major matrix of shape (3, 4) with contiguous storage,
    /// the strides would be [4, 1].
    /// 
    /// For alignement reasons, it is possible to have strides that don't match
    /// the shape of the matrix (meaning that some elements of the data array
    /// are unused).
    fn strides(&self) -> [usize; 2];

    /// Storage order. Specifies which dimension is stored the most contiguously
    /// in memory
    fn ordering(&self) -> StorageOrder {
        if self.strides()[0] > self.strides()[1] {
            return StorageOrder::RowMaj;
        }
        else {
            return StorageOrder::ColMaj;
        }
    }

    /// Access to the matrix's data
    /// 
    /// Getting access to the element located at row i and column j
    /// can be done by indexing self.data() at the location
    /// computed by i * strides[0] + j * strides[1]
    fn data(&self) -> &[N];

    /// Give the index into self.data() for accessing the element
    /// at row i and column j
    fn data_index(&self, i: usize, j: usize) -> usize {
        i * self.strides()[0] + j * self.strides()[1]
    }
}

/// A mutable view into a dense matrix
pub trait DenseMatViewMut<N> : DenseMatView<N> {

    /// Mutable access to the matrix's data
    /// 
    /// Getting access to the element located at row i and column j
    /// can be done by indexing self.data() at the location
    /// computed by i * strides[0] + j * strides[1]
    fn data_mut(&mut self) -> &mut [N];
}

/// Represents a dense matrix we own
/// 
/// This can be useful if another matrix library returned its own matrix
/// type, but we want to transform to our own type.
pub trait DenseMatOwned<N> : DenseMatView<N> {

    /// Get the underlying data array as a vector
    fn into_data(self) -> Vec<N>;
}
