use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// One or more of the supplied parameters is invalid
    InvalidParameterError,
    /// The requested quantity is not mathematically defined
    UndefinedQuantityError,
    /// An algorithm has reached the maximum number of iterations allowed
    MaxIterationsError,
    /// Recieved an empty container, but requires a non-empty container
    EmptyContainerError,
}

impl ErrorKind {
    pub fn as_str(&self) -> &str {
        match self {
            ErrorKind::InvalidParameterError => "invalid parameter error",
            ErrorKind::UndefinedQuantityError => "undefined quantity error",
            ErrorKind::MaxIterationsError => "max iterations error",
            ErrorKind::EmptyContainerError => "empty container error",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error {
    msg: String,
    kind: ErrorKind,
}

impl Error {
    pub fn new(kind: ErrorKind, msg: &str) -> Self {
        Error {
            msg: String::from(msg),
            kind,
        }
    }

    pub fn description(&self) -> &str {
        self.msg.as_str()
    }
}
