# Version Information

The XAD version can be queried through a set of preprocessor macros,
described in the following.

| Macro | Description |
|-------|-------------|
| `XAD_VERSION_MAJOR` | Numeric value of the XAD major version number, i.e. `1` for `1.x.y`. |
| `XAD_VERSION_MINOR` | Numeric value of the XAD minor version number, i.e. `4` for `x.4.y`. |
| `XAD_VERSION_PATCH` |  Numeric value of the XAD path version number, up to the first non-numeric character. Thus, for `1.0.0rc1`, the value of this macro is `0`. |
| `XAD_VERSION_PATCHSTRING` | String representation of the XAD patch version number, including non-numeric suffixes. |
| `XAD_VERSION` | Numeric representation of the full XAD Version number (without non-numeric suffixes e.g. for release candiates and betas). It is computed as:     `XAD_VERSION = XAD_MAJOR * 10000 + XAD_MINOR * 100 + XAD_PATCH` |
| `XAD_VERSION_STRING` | String representation of the full XAD version, including non-numeric suffixes, e.g. `1.1.0rc1` |
