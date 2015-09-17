///! A strided matrix implementation

use std::ops::{Deref, DerefMut, Range};
use std::iter::Map;
use std::slice::{Chunks, ChunksMut};
use num::traits::Num;

use errors::DMatError;
use StorageOrder;

/// A simple dense matrix
#[derive(PartialEq, Debug)]
pub struct StridedMat<N, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    rows: usize,
    cols: usize,
    strides: [usize; 2],
}

pub type MatView<'a, N> = StridedMat<N, &'a [N]>;
pub type MatViewMut<'a, N> = StridedMat<N, &'a mut [N]>;
pub type MatOwned<N> = StridedMat<N, Vec<N>>;

impl<N> StridedMat<N, Vec<N>> {
    /// Create a dense matrix from owned data
    pub fn new_owned(data: Vec<N>, rows: usize,
                     cols: usize, strides: [usize;2]) -> MatOwned<N> {
        StridedMat {
            data: data,
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Create an all-zero dense matrix
    pub fn zeros(rows: usize, cols: usize,
                 order: StorageOrder) -> MatOwned<N>
    where N: Num + Copy {
        let strides = match order {
            StorageOrder::RowMaj => [cols, 1],
            StorageOrder::ColMaj => [1, rows],
        };
        StridedMat {
            data: vec![N::zero(); rows*cols],
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Return the identity matrix of dimension `dim`
    ///
    /// The storage will be row major, however one can transpose if needed
    pub fn eye(dim: usize) -> MatOwned<N>
    where N: Num + Copy {
        let data = (0..dim*dim).map(|x| {
            if x % dim == x / dim { N::one() } else { N::zero() }
        }).collect();
        StridedMat {
            data: data,
            rows: dim,
            cols: dim,
            strides: [dim, 1],
        }
    }


    /// Get the underlying data array as a vector
    pub fn into_data(self) -> Vec<N> {
        self.data
    }
}

impl<'a, N: 'a> StridedMat<N, &'a [N]> {

    /// Create a view of a matrix implementing DenseMatView
    pub fn new_borrowed(data: &'a [N], rows: usize, cols: usize,
                        strides: [usize; 2]) -> MatView<'a, N> {
        StridedMat {
            data: data,
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Slice along the least varying dimension of the matrix, from
    /// index `start` and taking `count` vectors.
    ///
    /// e.g. for a row major matrix, get a view of `count` rows starting
    /// from `start`.
    pub fn middle_outer_views(&self,
                              start: usize,
                              count: usize
                             ) -> Result<MatView<'a, N>, DMatError> {
        let end = start + count;
        if count == 0 {
            return Err(DMatError::EmptyView);
        }
        if start >= self.outer_dims() || end > self.outer_dims() {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let (rows, cols) = match self.ordering() {
            StorageOrder::RowMaj => (count, self.cols()),
            StorageOrder::ColMaj => (self.rows(), count),
        };

        let s = self.outer_stride();
        let sliced_data = &self.data[start * s .. end * s];
        Ok(MatView {
            data: sliced_data,
            rows: rows,
            cols: cols,
            strides: self.strides,
        })
    }
}

impl<N, Storage> StridedMat<N, Storage>
where Storage: Deref<Target=[N]> {

    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The number of least varying dimensions
    pub fn outer_dims(&self) -> usize {
        match self.ordering() {
            StorageOrder::RowMaj => self.rows(),
            StorageOrder::ColMaj => self.cols(),
        }
    }

    /// The number of most varying dimensions
    pub fn inner_dims(&self) -> usize {
        match self.ordering() {
            StorageOrder::RowMaj => self.cols(),
            StorageOrder::ColMaj => self.rows(),
        }
    }

    /// The stride for the outer dimension
    pub fn outer_stride(&self) -> usize {
        match self.ordering() {
            StorageOrder::RowMaj => self.strides[0],
            StorageOrder::ColMaj => self.strides[1],
        }
    }

    /// The stride for the inner dimension
    pub fn inner_stride(&self) -> usize {
        match self.ordering() {
            StorageOrder::RowMaj => self.strides[1],
            StorageOrder::ColMaj => self.strides[0],
        }
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
    pub fn strides(&self) -> [usize; 2] {
        self.strides
    }

    /// Access to the matrix's data
    ///
    /// Getting access to the element located at row i and column j
    /// can be done by indexing self.data() at the location
    /// computed by i * strides[0] + j * strides[1]
    pub fn data(&self) -> &[N] {
        &self.data[..]
    }

    /// The number of rows and cols of the matrix
    pub fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }

    /// Storage order. Specifies which dimension is stored the most contiguously
    /// in memory
    pub fn ordering(&self) -> StorageOrder {
        if self.strides()[0] > self.strides()[1] {
            return StorageOrder::RowMaj;
        }
        else {
            return StorageOrder::ColMaj;
        }
    }

    /// Give the index into self.data() for accessing the element
    /// at row i and column j
    pub fn data_index(&self, i: usize, j: usize) -> usize {
        i * self.strides()[0] + j * self.strides()[1]
    }

    fn row_range_rowmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index(i, 0);
        let stop = self.data_index(i + 1, 0);
        start..stop
    }

    fn row_range_colmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index(i, 0);
        let stop = self.data_index(i + 1, self.cols() - 1);
        start..stop
    }

    fn col_range_rowmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index(0, j);
        let stop = self.data_index(self.rows() - 1, j + 1);
        start..stop
    }

    fn col_range_colmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index(0, j);
        let stop = self.data_index(0, j + 1);
        start..stop
    }



    /// Get a view into the specified row
    pub fn row(&self, i: usize) -> Result<VecView<N>, DMatError> {
        if i >= self.rows {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.row_range_rowmaj(i),
            StorageOrder::ColMaj => self.row_range_colmaj(i),
        };
        Ok(StridedVec {
            data: &self.data[range],
            dim: self.cols,
            stride: self.strides[1],
        })
    }

    /// Get a view into the specified column
    pub fn col(&self, j: usize) -> Result<VecView<N>, DMatError> {
        if j >= self.cols {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.col_range_rowmaj(j),
            StorageOrder::ColMaj => self.col_range_colmaj(j),
        };
        Ok(StridedVec {
            data: &self.data[range],
            dim: self.cols,
            stride: self.strides[0],
        })
    }

    pub fn outer_block_iter(&self, block_size: usize) -> ChunkOuterBlocks<N> {
        let mat = MatView {
            data: &self.data[..],
            rows: self.rows,
            cols: self.cols,
            strides: self.strides,
        };
        ChunkOuterBlocks {
            mat: mat,
            dims_in_bloc: block_size,
            bloc_count: 0,
        }
    }
}

impl<N, Storage> StridedMat<N, Storage>
where Storage: DerefMut<Target=[N]> {
    /// Mutable access to the matrix's data
    ///
    /// Getting access to the element located at row i and column j
    /// can be done by indexing self.data() at the location
    /// computed by i * strides[0] + j * strides[1]
    pub fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }

    /// Get a mutable view into the specified row
    pub fn row_mut(&mut self, i: usize) -> Result<VecViewMut<N>, DMatError> {
        if i >= self.rows {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.row_range_rowmaj(i),
            StorageOrder::ColMaj => self.row_range_colmaj(i),
        };
        Ok(StridedVec {
            data: &mut self.data[range],
            dim: self.cols,
            stride: self.strides[1],
        })
    }

    /// Get a mutable view into the specified column
    pub fn col_mut(&mut self,
                   j: usize) -> Result<VecViewMut<N>, DMatError> {
        if j >= self.cols {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.col_range_rowmaj(j),
            StorageOrder::ColMaj => self.col_range_colmaj(j),
        };
        Ok(StridedVec {
            data: &mut self.data[range],
            dim: self.cols,
            stride: self.strides[0],
        })
    }
}


/// A simple dense vector
#[derive(PartialEq, Debug)]
pub struct StridedVec<N, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    dim: usize,
    stride: usize,
}

pub type VecView<'a, N> = StridedVec<N, &'a [N]>;
pub type VecViewMut<'a, N> = StridedVec<N, &'a mut [N]>;
pub type VecOwned<N> = StridedVec<N, Vec<N>>;

fn take_first<N>(chunk: &[N]) -> &N {
    &chunk[0]
}

fn take_first_mut<N>(chunk: &mut [N]) -> &mut N {
    &mut chunk[0]
}



impl<N, Storage> StridedVec<N, Storage>
where Storage: Deref<Target=[N]> {

    /// Iterate over a dense vector's values by reference
    pub fn iter(&self) -> Map<Chunks<N>, fn(&[N]) -> &N> {
        self.data.chunks(self.stride).map(take_first)
    }

    /// The underlying data
    pub fn data(&self) -> &[N] {
        &self.data[..]
    }

    /// The number of dimensions
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The stride of this vector
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl<N, Storage> StridedVec<N, Storage>
where Storage: DerefMut<Target=[N]> {

    /// Iterate over a dense vector's values by mutable reference
    pub fn iter_mut(&mut self) -> Map<ChunksMut<N>, fn(&mut [N]) -> &mut N> {
        self.data.chunks_mut(self.stride).map(take_first_mut)
    }

    /// The underlying data as a mutable slice
    pub fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

/// An iterator over non-overlapping blocks of a matrix,
/// along the least-varying dimension
pub struct ChunkOuterBlocks<'a, N: 'a> {
    mat: MatView<'a, N>,
    dims_in_bloc: usize,
    bloc_count: usize
}

impl<'a, N: 'a> Iterator for ChunkOuterBlocks<'a, N> {
    type Item = MatView<'a, N>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let cur_dim = self.dims_in_bloc * self.bloc_count;
        let end_dim = self.dims_in_bloc + cur_dim;
        let count = if self.dims_in_bloc == 0 {
            return None;
        }
        else if end_dim > self.mat.outer_dims() {
            let count = self.mat.outer_dims() - cur_dim;
            self.dims_in_bloc = 0;
            count
        }
        else {
            self.dims_in_bloc
        };
        let view = self.mat.middle_outer_views(cur_dim,
                                               count).unwrap();
        self.bloc_count += 1;
        Some(view)
    }
}


#[cfg(test)]
mod tests {

    use super::{StridedMat, MatOwned};
    use errors::DMatError;

    #[test]
    fn row_view() {

        let mat = StridedMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [3, 1]);
        let view = mat.row(0).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[1., 1., 0.]);
        let view = mat.row(1).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[0., 1., 0.]);
        let view = mat.row(2).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[0., 0., 1.]);
        let res = mat.row(3);
        assert_eq!(res, Err(DMatError::OutOfBoundsIndex));
    }

    #[test]
    fn col_view() {

        let mat = StridedMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [3, 1]);
        let view = mat.col(0).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[1., 1., 0., 0., 1., 0., 0.]);
        let view = mat.col(1).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[1., 0., 0., 1., 0., 0., 0.]);
        let view = mat.col(2).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[0., 0., 1., 0., 0., 0., 1.]);
        let res = mat.col(3);
        assert_eq!(res, Err(DMatError::OutOfBoundsIndex));
    }

    #[test]
    fn row_iter() {
        let mat = StridedMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [1, 3]);

        {
            let row = mat.row(0).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let row = mat.row(1).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let row = mat.row(2).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), None);
        }

        let mut mat = mat;
        let mut row = mat.row_mut(0).unwrap();
        {
            let mut iter = row.iter_mut();
            *iter.next().unwrap() = 2.;
        }
        let mut iter = row.iter();
        assert_eq!(iter.next(), Some(&2.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn col_iter() {
        let mat = StridedMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [1, 3]);

        {
            let col = mat.col(0).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let col = mat.col(1).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let col = mat.col(2).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), None);
        }

        let mut mat = mat;
        let mut col = mat.col_mut(0).unwrap();
        {
            let mut iter = col.iter_mut();
            *iter.next().unwrap() = 2.;
        }
        let mut iter = col.iter();
        assert_eq!(iter.next(), Some(&2.));
        assert_eq!(iter.next(), Some(&1.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn eye() {
        let mat: MatOwned<f64> = StridedMat::eye(3);
        assert_eq!(mat.data(), &[1., 0., 0.,
                                 0., 1., 0.,
                                 0., 0., 1.]);
    }

    #[test]
    fn outer_block_iter() {
        let mat: MatOwned<f64> = StridedMat::eye(11);
        let mut block_iter = mat.outer_block_iter(3);
        assert_eq!(block_iter.next().unwrap().rows(), 3);
        assert_eq!(block_iter.next().unwrap().rows(), 3);
        assert_eq!(block_iter.next().unwrap().rows(), 3);
        assert_eq!(block_iter.next().unwrap().rows(), 2);
        assert_eq!(block_iter.next(), None);

        let mut block_iter = mat.outer_block_iter(4);
        assert_eq!(block_iter.next().unwrap().cols(), 11);
        assert_eq!(block_iter.next().unwrap().strides()[0], 11);
        assert_eq!(block_iter.next().unwrap().strides()[1], 1);
        assert_eq!(block_iter.next(), None);

        let mat: MatOwned<f64> = StridedMat::eye(3);
        let mut block_iter = mat.outer_block_iter(2);
        assert_eq!(block_iter.next().unwrap().data(), &[1., 0., 0.,
                                                        0., 1., 0.]);
        assert_eq!(block_iter.next().unwrap().data(), &[0., 0., 1.]);
        assert_eq!(block_iter.next(), None);
    }
}
