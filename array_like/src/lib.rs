/*!
This crate is a simple workaround around the fact that arrays don't
implement Copy or Deref (even though in practice they are).
This enables asking for arrays as a generic bound
*/

use std::ops::{Deref, DerefMut};

#[derive(Debug, PartialEq)]
pub struct ArrayLike<A> {
    array: A,
}

impl<A> ArrayLike<A> {
    pub fn new(a: A) -> Self {
        ArrayLike {
            array: a,
        }
    }

    pub fn inner(&self) -> A where A: Copy {
        self.array
    }
}

macro_rules! array_impl {
    ($len:expr) => (
        impl<T> Deref for ArrayLike<[T; $len]> {
            type Target = [T];
            fn deref<'a>(&'a self) -> &'a [T] {
                &self.array[..]
            }
        }

        impl<T> DerefMut for ArrayLike<[T; $len]> {
            fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
                &mut self.array[..]
            }
        }

        impl<T: Copy> Copy for ArrayLike<[T; $len]> { }

        impl<T: Copy> Clone for ArrayLike<[T; $len]> {
            fn clone(&self) -> Self {
                *self
            }
        }
    )
}


macro_rules! array_impl_recursive {
    ($len:expr, $($more:expr,)*) => (
        array_impl!($len);
        array_impl_recursive!($($more,)*);
    );
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

