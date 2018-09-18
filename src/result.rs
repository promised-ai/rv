use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// One or more of the supplied parameters is invalid
    InvalidParameter,
    /// The requested quantity is not mathematically defined
    UndefinedQuantity,
    /// An algorithm has reached the maximum number of iterations allowed
    MaxIterationsExceeded,
}

impl ErrorKind {
    pub fn as_str(&self) -> &str {
        match self {
            ErrorKind::InvalidParameter => "invalid parameter",
            ErrorKind::UndefinedQuantity => "undefined quantity",
            ErrorKind::MaxIterationsExceeded => "max iterations exceeded",
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
