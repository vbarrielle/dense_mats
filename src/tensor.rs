///! A strided tensor implementation

use std::ops::{Deref, DerefMut, Range, Index, IndexMut};
use std::iter::{Map, Take};
use std::slice::{self, Chunks, ChunksMut};
use num::traits::Num;

use array_like::{ArrayLike, ArrayLikeMut};

use errors::DMatError;
use StorageOrder;

/// A type for indexing an axis of a tensor
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Axis(pub usize);

/// A simple dense matrix
#[derive(PartialEq, Debug)]
pub struct Tensor<N, DimArray, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    shape: DimArray,
    strides: DimArray,
}

fn strides_from_shape_c_order<DimArray>(shape: &DimArray) -> DimArray
where DimArray: ArrayLikeMut<usize> {
    let mut strides = shape.clone();
    let mut prev = 1;
    for (stride, &dim) in strides.as_mut().iter_mut().rev()
                                 .zip(shape.as_ref().iter().rev()) {
        *stride = prev;
        prev *= dim;
    }
    strides
}

fn strides_from_shape_f_order<DimArray>(shape: &DimArray) -> DimArray
where DimArray: ArrayLikeMut<usize> {
    let mut strides = shape.clone();
    let mut prev = 1;
    for (stride, &dim) in strides.as_mut().iter_mut().zip(shape.as_ref()) {
        *stride = prev;
        prev *= dim;
    }
    strides
}

pub type TensorView<'a, N, DimArray> = Tensor<N, DimArray, &'a [N]>;
pub type TensorViewMut<'a, N, DimArray> = Tensor<N, DimArray, &'a mut [N]>;
pub type TensorOwned<N, DimArray> = Tensor<N, DimArray, Vec<N>>;

/// Methods available for all tensors regardless of their dimension count
impl<'a, N: 'a, DimArray> Tensor<N, DimArray, &'a[N]>
where DimArray: ArrayLikeMut<usize> {

}

impl<'a, N: 'a, DimArray> Tensor<N, DimArray, &'a mut [N]>
where DimArray: ArrayLikeMut<usize> {

}

impl<N, DimArray> Tensor<N, DimArray, Vec<N>>
where DimArray: ArrayLikeMut<usize> {

    /// Create an all-zero tensor in C order
    pub fn zeros(shape: DimArray) -> TensorOwned<N, DimArray>
    where N: Num + Copy {
        let strides = strides_from_shape_c_order(&shape);
        let size = shape.as_ref().iter().fold(1, |prod, x| prod * x);
        Tensor {
            data: vec![N::zero(); size],
            shape: shape,
            strides: strides,
        }
    }

    /// Create an all-zero tensor in F order
    pub fn zeros_f(shape: DimArray) -> TensorOwned<N, DimArray>
    where N: Num + Copy {
        let strides = strides_from_shape_f_order(&shape);
        let size = shape.as_ref().iter().fold(1, |prod, x| prod * x);
        Tensor {
            data: vec![N::zero(); size],
            shape: shape,
            strides: strides,
        }
    }

    /// Create an all-one tensor in C order
    pub fn ones(shape: DimArray) -> TensorOwned<N, DimArray>
    where N: Num + Copy {
        let strides = strides_from_shape_c_order(&shape);
        let size = shape.as_ref().iter().fold(1, |prod, x| prod * x);
        Tensor {
            data: vec![N::one(); size],
            shape: shape,
            strides: strides,
        }
    }

    /// Create an all-one tensor in F order
    pub fn ones_c(shape: DimArray) -> TensorOwned<N, DimArray>
    where N: Num + Copy {
        let strides = strides_from_shape_f_order(&shape);
        let size = shape.as_ref().iter().fold(1, |prod, x| prod * x);
        Tensor {
            data: vec![N::one(); size],
            shape: shape,
            strides: strides,
        }
    }
}

/// Methods available for all tensors regardless of their dimension count
impl<N, DimArray, Storage> Tensor<N, DimArray, Storage>
where DimArray: ArrayLike<usize>,
      Storage: Deref<Target=[N]> {

    /// The number of dimensions of this tensor
    pub fn ndims(&self) -> usize {
        self.strides.as_ref().len()
    }

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

    /// Returns true if the array is contiguous, ie the fastest varying
    /// axis has stride 1, and no unused data is present in the array
    pub fn is_contiguous(&self) -> bool {
        let stride = self.inner_stride();
        stride.is_none() || (stride.unwrap() == 1
                             && self.is_nearly_contiguous())
    }

    /// Checks whether all dimensions except the fastest varying one
    /// are contiguous. Having that property verified allows flattened
    /// views of the tensor using `ravel()`. Otherwise copies have to be done.
    pub fn is_nearly_contiguous(&self) -> bool {
        if self.ndims() == 0 {
            return true;
        }
        let outer_dim_index = self.outer_dim().unwrap();
        let outer_stride = self.outer_stride().unwrap();
        let inner_stride = self.inner_stride().unwrap();
        let dim_prod = self.shape_ref().iter().enumerate()
                                              .fold(1, |p, (i, x)| {
            if i == outer_dim_index {
                p
            }
            else {
                x * p
            }
        });
        (outer_stride / inner_stride) == dim_prod
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

    /// Iteration on outer blocks views of size block_size
    pub fn outer_block_iter<'a>(&'a self, block_size: usize
                               ) -> ChunkOuterBlocks<'a, N, DimArray, Storage> {
        ChunkOuterBlocks {
            tensor: &self,
            dims_in_bloc: block_size,
            bloc_count: 0,
        }
    }

    /// Index into the data array for the given N-dimensional index
    pub fn data_index(&self, index: DimArray) -> usize {
        index.as_ref().iter().zip(self.strides_ref())
                      .map(|(x, y)| x * y)
                      .fold(0, |sum, y| sum + y)
    }

    pub fn to_owned(&self) -> TensorOwned<N, DimArray>
    where N: Copy {
        TensorOwned {
            data: self.data.to_vec(),
            shape: self.shape(),
            strides: self.strides(),
        }
    }

    /// Iteration on the given axis
    pub fn iter_axis<'a>(&'a self, axis: Axis
                        ) -> Slices<'a, N, DimArray, Storage> {
        Slices {
            tensor: self,
            axis: axis,
            index: 0,
        }
    }

    /// Get a view as a tensor of lower dimension count, by
    /// slicing into the given axis at the given index
    pub fn slice_dim<'a>(&'a self, Axis(dim): Axis, index: usize
                        ) -> TensorView<'a, N, DimArray::Pred>
    where DimArray: ArrayLikeMut<usize> {
        let shape = self.shape.remove_val(dim);
        let strides = self.strides.remove_val(dim);
        let mut indexing = self.shape.clone();
        for val in indexing.as_mut().iter_mut() {
            *val = 0
        }
        indexing.as_mut()[dim] = index;
        let data_index = self.data_index(indexing);
        let data = &self.data[data_index..];
        TensorView {
            data: data,
            shape: shape,
            strides: strides
        }
    }

    /// Get a view over the tensor's diagonal
    ///
    /// The diagonal vector is as long as the smallest dimension.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is zero-dim.
    pub fn diag_view(&self) -> TensorView<N, [usize; 1]> {
        let strides = [self.strides_ref().iter().fold(0, |sum, x| sum + x)];
        let shape = [*self.shape_ref().iter().min().unwrap()];
        TensorView {
            data: &self.data[..],
            shape: shape,
            strides: strides,
        }
    }
}

impl<N, DimArray, Storage> Tensor<N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: Deref<Target=[N]> {
    /// Slice along the least varying dimension of the matrix, from
    /// index `start` and taking `count` vectors.
    ///
    /// e.g. for a row major matrix, get a view of `count` rows starting
    /// from `start`.
    pub fn middle_outer_views<'a>(&'a self,
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


impl<N, DimArray, Storage> Tensor<N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: DerefMut<Target=[N]> {

    /// Iteration on mutable outer blocks views of size block_size
    pub fn outer_block_iter_mut<'a>(&'a mut self, block_size: usize
                                   ) -> ChunkOuterBlocksMut<'a, N,
                                                            DimArray, Storage> {
        ChunkOuterBlocksMut {
            tensor: self,
            dims_in_bloc: block_size,
            bloc_count: 0,
        }
    }

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

    /// Get a mutable view as a tensor of lower dimension count, by
    /// slicing into the given axis at the given index
    pub fn slice_dim_mut<'a>(&'a mut self, Axis(dim): Axis, index: usize
                            ) -> TensorViewMut<'a, N, DimArray::Pred>
    where DimArray: ArrayLikeMut<usize> {
        let shape = self.shape.remove_val(dim);
        let strides = self.strides.remove_val(dim);
        let mut indexing = self.shape.clone();
        for val in indexing.as_mut().iter_mut() {
            *val = 0
        }
        indexing.as_mut()[dim] = index;
        let data_index = self.data_index(indexing);
        let data = &mut self.data[data_index..];
        TensorViewMut {
            data: data,
            shape: shape,
            strides: strides
        }
    }

    /// Unsafe version of slice_dim_mut for mutable iteration
    unsafe fn slice_dim_mut_raw<'a>(&mut self,
                                    Axis(dim): Axis, index: usize
                                   ) -> TensorViewMut<'a, N, DimArray::Pred>
    where N: 'a {
        let shape = self.shape.remove_val(dim);
        let strides = self.strides.remove_val(dim);
        let mut indexing = self.shape.clone();
        for val in indexing.as_mut().iter_mut() {
            *val = 0
        }
        indexing.as_mut()[dim] = index;
        let data_index = self.data_index(indexing);
        let nb_elems = self.data.len() - data_index;
        let data = {
            let ptr = self.data.as_mut_ptr();
            slice::from_raw_parts_mut(ptr.offset(data_index as isize),
                                      nb_elems)
        };
        TensorViewMut {
            data: data,
            shape: shape,
            strides: strides
        }
    }

    pub fn iter_axis_mut<'a>(&'a mut self, axis: Axis
                            ) -> SlicesMut<'a, N, DimArray, Storage> {
        SlicesMut {
            tensor: self,
            axis: axis,
            index: 0,
        }
    }

    /// Slice mutably along the least varying dimension of the matrix, from
    /// index `start` and taking `count` vectors.
    ///
    /// e.g. for a row major matrix, get a view of `count` rows starting
    /// from `start`.
    pub fn middle_outer_views_mut<'a>(&'a mut self,
                                      start: usize,
                                      count: usize
                                     ) -> Result<TensorViewMut<'a, N, DimArray>,
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
        unsafe {
            self.mid_outer_views_mut(start, count)
        }
    }

    /// Core unsafe implementation of middle_outer_views_mut()
    /// Bounds checking should have been performed before
    unsafe fn mid_outer_views_mut<'a>(&mut self,
                                      start: usize,
                                      count: usize
                                     ) -> Result<TensorViewMut<'a, N, DimArray>,
                                                 DMatError>
    where N: 'a {
        let dim_index = try!(self.outer_dim().ok_or(DMatError::ZeroDimTensor));
        let mut shape = self.shape.clone();
        shape.as_mut()[dim_index] = count;

        let s = try!(self.outer_stride().ok_or(DMatError::ZeroDimTensor));
        let strides = self.strides();

        // safe because we already checked for out of bounds
        let sliced_data = {
            let ptr = self.data.as_mut_ptr();
            slice::from_raw_parts_mut(ptr.offset((start * s) as isize),
                                      count * s)
        };
        Ok(TensorViewMut {
            data: sliced_data,
            shape: shape,
            strides: strides,
        })
    }

    /// Get a mutable view over the tensor's diagonal
    ///
    /// The diagonal vector is as long as the smallest dimension.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is zero-dim.
    pub fn diag_view_mut(&mut self) -> TensorViewMut<N, [usize; 1]> {
        let strides = [self.strides_ref().iter().fold(0, |sum, x| sum + x)];
        let shape = [*self.shape_ref().iter().min().unwrap()];
        TensorViewMut {
            data: &mut self.data[..],
            shape: shape,
            strides: strides,
        }
    }
}

impl<N, DimArray, Storage> Index<DimArray> for Tensor<N, DimArray, Storage>
where DimArray: ArrayLike<usize>,
      Storage: Deref<Target=[N]> {

    type Output = N;

    fn index<'a>(&'a self, index: DimArray) -> &'a N {
        let data_index = self.data_index(index);
        &self.data[data_index]
    }
}

impl<N, DimArray, Storage> IndexMut<DimArray> for Tensor<N, DimArray, Storage>
where DimArray: ArrayLike<usize>,
      Storage: DerefMut<Target=[N]> {

    fn index_mut<'a>(&'a mut self, index: DimArray) -> &'a mut N {
        let data_index = self.data_index(index);
        &mut self.data[data_index]
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

    fn row_range_rowmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index([i, 0]);
        let stop = self.data_index([i + 1, 0]);
        start..stop
    }

    fn row_range_colmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index([i, 0]);
        let stop = self.data_index([i + 1, self.cols() - 1]);
        start..stop
    }

    fn col_range_rowmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index([0, j]);
        let stop = self.data_index([self.rows() - 1, j + 1]);
        start..stop
    }

    fn col_range_colmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index([0, j]);
        let stop = self.data_index([0, j + 1]);
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
    pub fn iter(&self) -> Map<Take<Chunks<N>>, fn(&[N]) -> &N> {
        self.data.chunks(self.stride()).take(self.dim()).map(take_first)
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
    pub fn iter_mut(&mut self
                   ) -> Map<Take<ChunksMut<N>>, fn(&mut [N]) -> &mut N> {
        let count = self.dim();
        self.data.chunks_mut(self.strides[0]).take(count)
                                             .map(take_first_mut)
                                             
    }

    /// The underlying data as a mutable slice
    pub fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

/// An iterator over non-overlapping blocks of a tensor,
/// along the least-varying dimension
pub struct ChunkOuterBlocks<'a, N: 'a, DimArray: 'a, Storage: 'a>
where DimArray: ArrayLike<usize>,
      Storage: Deref<Target=[N]> {
    tensor: &'a Tensor<N, DimArray, Storage>,
    dims_in_bloc: usize,
    bloc_count: usize
}

impl<'a, N: 'a, DimArray, Storage>
Iterator for ChunkOuterBlocks<'a, N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: Deref<Target=[N]> {
    type Item = TensorView<'a, N, DimArray>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let cur_dim = self.dims_in_bloc * self.bloc_count;
        let end_dim = self.dims_in_bloc + cur_dim;
        let count = if self.dims_in_bloc == 0 {
            return None;
        }
        else if end_dim > self.tensor.outer_shape().unwrap() {
            let count = self.tensor.outer_shape().unwrap() - cur_dim;
            self.dims_in_bloc = 0;
            count
        }
        else {
            self.dims_in_bloc
        };
        let view = self.tensor.middle_outer_views(cur_dim,
                                                  count).unwrap();
        self.bloc_count += 1;
        Some(view)
    }
}

/// An iterator over non-overlapping mutable blocks of a matrix,
/// along the least-varying dimension
pub struct ChunkOuterBlocksMut<'a, N: 'a, DimArray: 'a, Storage: 'a>
where DimArray: ArrayLike<usize>,
      Storage: DerefMut<Target=[N]> {
    tensor: &'a mut Tensor<N, DimArray, Storage>,
    dims_in_bloc: usize,
    bloc_count: usize
}

impl<'a, N: 'a, DimArray, Storage>
Iterator for ChunkOuterBlocksMut<'a, N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: DerefMut<Target=[N]> {
    type Item = TensorViewMut<'a, N, DimArray>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let cur_dim = self.dims_in_bloc * self.bloc_count;
        let end_dim = self.dims_in_bloc + cur_dim;
        let count = if self.dims_in_bloc == 0 {
            return None;
        }
        else if end_dim > self.tensor.outer_shape().unwrap() {
            let count = self.tensor.outer_shape().unwrap() - cur_dim;
            self.dims_in_bloc = 0;
            count
        }
        else {
            self.dims_in_bloc
        };
        let view = unsafe {
            self.tensor.mid_outer_views_mut(cur_dim, count).unwrap()
        };
        self.bloc_count += 1;
        Some(view)
    }
}

pub struct Slices<'a, N: 'a, DimArray: 'a, Storage: 'a>
where Storage: Deref<Target=[N]> {
    tensor: &'a Tensor<N, DimArray, Storage>,
    axis: Axis,
    index: usize,
}

impl<'a, N: 'a, DimArray, Storage>
Iterator for Slices<'a, N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: Deref<Target=[N]> {
    type Item = TensorView<'a, N, <DimArray as ArrayLike<usize>>::Pred>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let Axis(axis) = self.axis;
        if self.index >= self.tensor.shape_ref()[axis] {
            return None;
        }
        let view = self.tensor.slice_dim(self.axis, self.index);
        self.index += 1;
        Some(view)
    }
}

pub struct SlicesMut<'a, N: 'a, DimArray: 'a, Storage: 'a>
where Storage: DerefMut<Target=[N]> {
    tensor: &'a mut Tensor<N, DimArray, Storage>,
    axis: Axis,
    index: usize,
}

impl<'a, N: 'a, DimArray, Storage>
Iterator for SlicesMut<'a, N, DimArray, Storage>
where DimArray: ArrayLikeMut<usize>,
      Storage: DerefMut<Target=[N]> {
    type Item = TensorViewMut<'a, N, <DimArray as ArrayLike<usize>>::Pred>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let Axis(axis) = self.axis;
        if self.index >= self.tensor.shape_ref()[axis] {
            return None;
        }
        let view = unsafe {
            self.tensor.slice_dim_mut_raw(self.axis, self.index)
        };
        self.index += 1;
        Some(view)
    }
}

#[cfg(test)]
mod tests {

    use super::{Tensor, MatOwned, TensorOwned, Axis};
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

    #[test]
    fn outer_block_iter_mut() {
        let mut mat: MatOwned<f64> = Tensor::eye(11);
        let block2_ref = {
            let mat_view = mat.borrowed();
            mat_view.middle_outer_views(3, 3).unwrap().to_owned()
        };
        let mut block_iter = mat.outer_block_iter_mut(3);
        assert_eq!(block_iter.next().unwrap().rows(), 3);
        let mut block2 = block_iter.next().unwrap();
        block2.row_mut(0).unwrap().data_mut()[0] = 1.;
        assert_eq!(block2.data()[0], 1.);
        assert_eq!(&block2.data()[1..], &block2_ref.data()[1..]);
    }

    #[test]
    fn indexing() {
        let mut mat: MatOwned<f64> = Tensor::eye(11);
        assert_eq!(mat[[0,0]], 1.);
        assert_eq!(mat[[1,1]], 1.);
        assert_eq!(mat[[2,2]], 1.);
        assert_eq!(mat[[3,2]], 0.);
        assert_eq!(mat[[2,3]], 0.);

        mat[[0,0]] = 2.;
        assert_eq!(mat[[0,0]], 2.);
    }

    #[test]
    fn slice_dim() {
        let mut tensor: TensorOwned<f64,_> = Tensor::zeros([5, 4, 3]);
        tensor[[0,0,0]] = 2.;
        {
            let mat43_0 = tensor.slice_dim(Axis(0), 0);
            assert_eq!(mat43_0[[0,0]], 2.);
            let mat43_1 = tensor.slice_dim(Axis(0), 1);
            assert_eq!(mat43_1[[0,0]], 0.);
        }

        {
            let mut mat53_0 = tensor.slice_dim_mut(Axis(1), 0);
            mat53_0[[4,1]] = 3.;
        }
        assert_eq!(tensor[[4, 0, 1]], 3.);
        {
            let mut mat54_1 = tensor.slice_dim_mut(Axis(2), 1);
            mat54_1[[4,3]] = 4.;
        }
        assert_eq!(tensor[[4, 3, 1]], 4.);
    }

    #[test]
    fn iter_axis() {
        let mut tensor: TensorOwned<f64,_> = Tensor::zeros([5, 4, 3]);
        tensor[[0,0,0]] = 2.;
        tensor[[1,1,2]] = 4.;
        tensor[[3,2,0]] = 3.;

        let mut iter0 = tensor.iter_axis(Axis(0));
        let mat43_0 = iter0.next().unwrap();
        assert_eq!(mat43_0[[0,0]], 2.);
        assert_eq!(mat43_0[[1,1]], 0.);
        let mat43_1 = iter0.next().unwrap();
        assert_eq!(mat43_1[[0,0]], 0.);
        assert_eq!(mat43_1[[1,1]], 0.);
        let mat43_2 = iter0.next().unwrap();
        assert_eq!(mat43_2[[0,0]], 0.);
        assert_eq!(mat43_2[[1,1]], 0.);
        let mat43_3 = iter0.next().unwrap();
        assert_eq!(mat43_3[[0,0]], 0.);
        assert_eq!(mat43_3[[1,1]], 0.);
        let mat43_4 = iter0.next().unwrap();
        assert_eq!(mat43_4[[0,0]], 0.);
        assert_eq!(mat43_4[[1,1]], 0.);
        assert_eq!(iter0.next(), None);

        let mut iter2 = tensor.iter_axis(Axis(2));
        let mat53_0 = iter2.next().unwrap();
        assert_eq!(mat53_0[[3,2]], 3.);
        assert_eq!(mat53_0[[0,0]], 2.);
        let mat53_1 = iter2.next().unwrap();
        assert_eq!(mat53_1[[3,2]], 0.);
        assert_eq!(mat53_1[[0,0]], 0.);
        let mat53_2 = iter2.next().unwrap();
        assert_eq!(mat53_2[[3,2]], 0.);
        assert_eq!(mat53_2[[0,0]], 0.);
        assert_eq!(iter2.next(), None);
    }

    #[test]
    fn iter_axis_mut() {
        let mut tensor: TensorOwned<f64,_> = Tensor::zeros([5, 4, 3]);
        {
            let mut iter0 = tensor.iter_axis_mut(Axis(0));
            let mut mat43_0 = iter0.next().unwrap();
            mat43_0[[0,0]] = 2.;
        }
        {
            let mut iter1 = tensor.iter_axis_mut(Axis(1));
            let mut mat53_0 = iter1.next().unwrap();
            mat53_0[[1,0]] = 3.;
        }
        assert_eq!(tensor[[0,0,0]], 2.);
        assert_eq!(tensor[[1,0,0]], 3.);
    }

    #[test]
    fn contiguity() {
        let tensor: TensorOwned<f64,_> = Tensor::zeros([5, 4, 3]);
        assert!(tensor.is_nearly_contiguous());
        assert!(tensor.is_contiguous());
    }
}
