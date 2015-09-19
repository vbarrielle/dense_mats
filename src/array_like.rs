/*!
This module is a simple workaround around the fact that arrays don't
implement Copy or Deref (even though in practice they are).
This enables asking for arrays as a generic bound
*/

pub trait ArrayLike<T> : Eq + Clone + AsRef<[T]> {
    fn ndims(&self) -> usize {
        self.as_ref().len()
    }
}

pub trait ArrayLikeMut<T> : ArrayLike<T> + AsMut<[T]> {}

macro_rules! array_impl {
    ($len:expr) => (
        impl<T: Copy + Eq> ArrayLike<T> for [T; $len] {}
        impl<T: Copy + Eq> ArrayLikeMut<T> for [T; $len] {}
    )
}

array_impl!(0);
array_impl!(1);
array_impl!(2);
array_impl!(3);
array_impl!(4);
array_impl!(5);
array_impl!(6);
array_impl!(7);
array_impl!(8);
array_impl!(9);
array_impl!(10);
array_impl!(11);
array_impl!(12);
array_impl!(13);
array_impl!(14);
array_impl!(15);
array_impl!(16);
array_impl!(17);
array_impl!(18);
array_impl!(19);
array_impl!(20);
array_impl!(21);
array_impl!(22);
array_impl!(23);
array_impl!(24);
array_impl!(25);
array_impl!(26);
array_impl!(27);
array_impl!(28);
array_impl!(29);
array_impl!(30);
array_impl!(31);
array_impl!(32);

