///! A strided matrix implementation

use std::ops::{Deref, DerefMut, Range};
use std::iter::Map;
use std::slice::{Chunks, ChunksMut};
use num::traits::Num;

use array_like::{ArrayLike, ArrayLikeMut};

use errors::DMatError;
use StorageOrder;

/// A simple dense matrix
#[derive(PartialEq, Debug)]
pub struct Tensor<N, DimArray, Storage>
where Storage: Deref<Target=[N]>,
      DimArray: ArrayLike<usize> {
    data: Storage,
    shape: DimArray,
    strides: DimArray,
}

pub type TensorView<'a, N, DimArray> = Tensor<N, DimArray, &'a [N]>;
pub type TensorViewMut<'a, N, DimArray> = Tensor<N, DimArray, &'a mut [N]>;
pub type TensorOwned<N, DimArray> = Tensor<N, DimArray, Vec<N>>;

/// Methods available for all tensors regardless of their dimension count
impl<'a, N: 'a, DimArray> Tensor<N, DimArray, &'a[N]>
where DimArray: ArrayLikeMut<usize> {

    /// Slice along the least varying dimension of the matrix, from
    /// index `start` and taking `count` vectors.
    ///
    /// e.g. for a row major matrix, get a view of `count` rows starting
    /// from `start`.
    pub fn middle_outer_views(&self,
                              start: usize,
                              count: usize
                             ) -> Result<TensorView<'a, N, DimArray>,
                                         DMatError> {
        let end = start + count;
        if count == 0 {
            return Err(DMatError::EmptyView);
        }
        let outer_shape = try!(self.outer_shape()
                                   .ok_or(DMatError::ZeroDimTensor));
        if start >= outer_shape || end > outer_shape {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let dim_index = try!(self.outer_dim().ok_or(DMatError::ZeroDimTensor));
        let mut shape = self.shape.clone();
        shape.as_mut()[dim_index] = count;

        let s = try!(self.outer_stride().ok_or(DMatError::ZeroDimTensor));
        let sliced_data = &self.data[start * s .. end * s];
        Ok(TensorView {
            data: sliced_data,
            shape: shape,
            strides: self.strides(),
        })
    }
}

/// Methods available for all tensors regardless of their dimension count
impl<N, DimArray, Storage> Tensor<N, DimArray, Storage>
where DimArray: ArrayLike<usize>,
      Storage: Deref<Target=[N]> {

    /// The strides of the tensor.
    ///
    /// # Explanations on a matrix.
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
    pub fn strides(&self) -> DimArray {
        self.strides.clone()
    }

    /// Get the strides by reference
    pub fn strides_ref(&self) -> &[usize] {
        self.strides.as_ref()
    }

    /// Access to the tensors's data
    ///
    /// # Explanations on a matrix.
    ///
    /// Getting access to the element located at row i and column j
    /// can be done by indexing self.data() at the location
    /// computed by i * strides[0] + j * strides[1]
    pub fn data(&self) -> &[N] {
        &self.data[..]
    }

    /// The shape of the tensor
    pub fn shape(&self) -> DimArray {
        self.shape.clone()
    }

    /// Get the shape by reference
    pub fn shape_ref(&self) -> &[usize] {
        self.shape.as_ref()
    }

    /// Get the storage order of this tensor
    pub fn ordering(&self) -> StorageOrder {
        let ascending = self.strides_ref().windows(2).all(|w| w[0] < w[1]);
        let descending = self.strides_ref().windows(2).all(|w| w[0] > w[1]);
        match (ascending, descending) {
            (true, false) => StorageOrder::F,
            (false, true) => StorageOrder::C,
            _ => StorageOrder::Unordered,
        }
    }

    /// Get the slowest varying dimension index
    pub fn outer_dim(&self) -> Option<usize> {
        self.strides_ref().iter().enumerate()
                           .fold(None, |max, (i, &x)| {
                               max.map_or(Some((i, x)), |(i0, x0)| {
                                   if x > x0 {
                                       Some((i, x))
                                   }
                                   else {
                                       Some((i0, x0))
                                   }
                               })
                           }).map(|(i, _)| i)
    }

    /// Get the fastest varying dimension index
    pub fn inner_dim(&self) -> Option<usize> {
        self.strides_ref().iter().enumerate()
                           .fold(None, |min, (i, &x)| {
                               min.map_or(Some((i, x)), |(i0, x0)| {
                                   if x < x0 {
                                       Some((i, x))
                                   }
                                   else {
                                       Some((i0, x0))
                                   }
                               })
                           }).map(|(i, _)| i)
    }

    /// The stride for the outer dimension
    pub fn outer_stride(&self) -> Option<usize> {
        self.outer_dim().map(|i| self.strides_ref()[i])
    }

    /// The stride for the inner dimension
    pub fn inner_stride(&self) -> Option<usize> {
        self.inner_dim().map(|i| self.strides_ref()[i])
    }

    /// The shape of the outer dimension
    pub fn outer_shape(&self) -> Option<usize> {
        self.outer_dim().map(|i| self.shape_ref()[i])
    }

    /// The stride for the inner dimension
    pub fn inner_shape(&self) -> Option<usize> {
        self.inner_dim().map(|i| self.shape_ref()[i])
    }

    /// Get a view into this tensor
    pub fn borrowed(&self) -> TensorView<N, DimArray> {
        TensorView {
            data: &self.data[..],
            shape: self.shape(),
            strides: self.strides(),
        }
    }

}

/// Methods available for all tensors regardless of their dimension count
impl<'a, N: 'a, DimArray> Tensor<N, DimArray, &'a mut [N]>
where DimArray: ArrayLike<usize> {

    /// Get a mutable view into this tensor
    pub fn borrowed_mut(&mut self) -> TensorViewMut<N, DimArray> {
        let shape = self.shape();
        let stride = self.strides();
        TensorViewMut {
            data: &mut self.data[..],
            shape: shape,
            strides: stride,
        }
    }
}

pub type MatView<'a, N> = Tensor<N, [usize; 2], &'a [N]>;
pub type MatViewMut<'a, N> = Tensor<N, [usize; 2], &'a mut [N]>;
pub type MatOwned<N> = Tensor<N, [usize; 2], Vec<N>>;

impl<N> Tensor<N, [usize; 2], Vec<N>> {
    /// Create a dense matrix from owned data
    pub fn new_owned(data: Vec<N>, rows: usize,
                     cols: usize, strides: [usize;2]) -> MatOwned<N> {
        Tensor {
            data: data,
            shape: [rows, cols],
            strides: strides,
        }
    }

    /// Create an all-zero dense matrix
    ///
    /// Defaults to C order if order equals Unordered
    pub fn zeros(rows: usize, cols: usize,
                 order: StorageOrder) -> MatOwned<N>
    where N: Num + Copy {
        let strides = match order {
            StorageOrder::C => [cols, 1],
            StorageOrder::F => [1, rows],
            StorageOrder::Unordered => [cols, 1]
        };
        Tensor {
            data: vec![N::zero(); rows*cols],
            shape: [rows, cols],
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
        Tensor {
            data: data,
            shape: [dim, dim],
            strides: [dim, 1],
        }
    }


    /// Get the underlying data array as a vector
    pub fn into_data(self) -> Vec<N> {
        self.data
    }
}

impl<'a, N: 'a> Tensor<N, [usize; 2], &'a [N]> {

    /// Create a view of a matrix implementing DenseMatView
    pub fn new_mat_view(data: &'a [N], rows: usize, cols: usize,
                        strides: [usize; 2]) -> MatView<'a, N> {
        Tensor {
            data: data,
            shape: [rows, cols],
            strides: strides,
        }
    }

}

impl<N, Storage> Tensor<N, [usize; 2], Storage>
where Storage: Deref<Target=[N]> {

    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.shape[1]
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
        if i >= self.rows() {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::C => self.row_range_rowmaj(i),
            StorageOrder::F => self.row_range_colmaj(i),
            StorageOrder::Unordered => unreachable!(),
        };
        Ok(Tensor {
            data: &self.data[range],
            shape: [self.cols()],
            strides: [self.strides[1]],
        })
    }

    /// Get a view into the specified column
    pub fn col(&self, j: usize) -> Result<VecView<N>, DMatError> {
        if j >= self.cols() {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::C => self.col_range_rowmaj(j),
            StorageOrder::F => self.col_range_colmaj(j),
            StorageOrder::Unordered => unreachable!(),
        };
        Ok(Tensor {
            data: &self.data[range],
            shape: [self.rows()],
            strides: [self.strides[0]],
        })
    }

    pub fn outer_block_iter(&self, block_size: usize) -> ChunkOuterBlocks<N> {
        let mat = MatView {
            data: &self.data[..],
            shape: [self.rows(), self.cols()],
            strides: self.strides,
        };
        ChunkOuterBlocks {
            mat: mat,
            dims_in_bloc: block_size,
            bloc_count: 0,
        }
    }
}

impl<N, Storage> Tensor<N, [usize; 2], Storage>
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
        if i >= self.rows() {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::C => self.row_range_rowmaj(i),
            StorageOrder::F => self.row_range_colmaj(i),
            StorageOrder::Unordered => unreachable!(),
        };
        let dim = self.cols();
        Ok(Tensor {
            data: &mut self.data[range],
            shape: [dim],
            strides: [self.strides[1]],
        })
    }

    /// Get a mutable view into the specified column
    pub fn col_mut(&mut self,
                   j: usize) -> Result<VecViewMut<N>, DMatError> {
        if j >= self.cols() {
            return Err(DMatError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::C => self.col_range_rowmaj(j),
            StorageOrder::F => self.col_range_colmaj(j),
            StorageOrder::Unordered => unreachable!(),
        };
        let dim = self.cols();
        Ok(Tensor {
            data: &mut self.data[range],
            shape: [dim],
            strides: [self.strides[0]],
        })
    }
}


pub type VecView<'a, N> = Tensor<N, [usize; 1], &'a [N]>;
pub type VecViewMut<'a, N> = Tensor<N, [usize; 1], &'a mut [N]>;
pub type VecOwned<N> = Tensor<N, [usize; 1], Vec<N>>;

fn take_first<N>(chunk: &[N]) -> &N {
    &chunk[0]
}

fn take_first_mut<N>(chunk: &mut [N]) -> &mut N {
    &mut chunk[0]
}



impl<N, Storage> Tensor<N, [usize;1], Storage>
where Storage: Deref<Target=[N]> {

    /// Iterate over a dense vector's values by reference
    pub fn iter(&self) -> Map<Chunks<N>, fn(&[N]) -> &N> {
        self.data.chunks(self.stride()).map(take_first)
    }

    /// The number of dimensions
    pub fn dim(&self) -> usize {
        self.shape[0]
    }

    /// The stride of this vector
    pub fn stride(&self) -> usize {
        self.strides[0]
    }
}

impl<N, Storage> Tensor<N, [usize;1], Storage>
where Storage: DerefMut<Target=[N]> {

    /// Iterate over a dense vector's values by mutable reference
    pub fn iter_mut(&mut self) -> Map<ChunksMut<N>, fn(&mut [N]) -> &mut N> {
        self.data.chunks_mut(self.strides[0]).map(take_first_mut)
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
        else if end_dim > self.mat.outer_shape().unwrap() {
            let count = self.mat.outer_shape().unwrap() - cur_dim;
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

    use super::{Tensor, MatOwned};
    use errors::DMatError;

    #[test]
    fn row_view() {

        let mat = Tensor::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
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

        let mat = Tensor::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
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
        let mat = Tensor::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
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
        let mat = MatOwned::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
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
        let mat: MatOwned<f64> = Tensor::eye(3);
        assert_eq!(mat.data(), &[1., 0., 0.,
                                 0., 1., 0.,
                                 0., 0., 1.]);
    }

    #[test]
    fn outer_block_iter() {
        let mat: MatOwned<f64> = Tensor::eye(11);
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

        let mat: MatOwned<f64> = Tensor::eye(3);
        let mut block_iter = mat.outer_block_iter(2);
        assert_eq!(block_iter.next().unwrap().data(), &[1., 0., 0.,
                                                        0., 1., 0.]);
        assert_eq!(block_iter.next().unwrap().data(), &[0., 0., 1.]);
        assert_eq!(block_iter.next(), None);
    }
}
