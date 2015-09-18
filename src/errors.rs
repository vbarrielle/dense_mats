//! Error type for sprs

use std::error::Error;
use std::fmt;

#[derive(PartialEq, Debug)]
pub enum DMatError {
    IncompatibleDimensions,
    BadWorkspaceDimensions,
    IncompatibleStorages,
    BadStorageType,
    NotImplemented,
    OutOfBoundsIndex,
    EmptyView,
    UnorderedStorage,
    ZeroDimTensor,
}

use self::DMatError::*;

impl DMatError {
    fn descr(&self) -> &str {
        match *self {
            IncompatibleDimensions => "matrices dimensions do not agree",
            BadWorkspaceDimensions =>
                "workspace dimension does not match requirements",
            IncompatibleStorages => "incompatible storages",
            BadStorageType => "wrong storage type",
            NotImplemented => "this method is not yet implemented",
            OutOfBoundsIndex => "an element in indices is out of bounds",
            EmptyView => "trying to create a view without elements",
            UnorderedStorage => "this algorithm expects C or F storage order",
            ZeroDimTensor => "this tensor has no dimensions",
        }
    }
}

impl Error for DMatError {
    fn description(&self) -> &str {
        self.descr()
    }
}

impl fmt::Display for DMatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.descr().fmt(f)
    }
}

