# Version Information

The XAD version can be queried through a set of preprocessor macros,
defined in the header `XAD/Version.hpp` (included by `XAD/XAD.hpp`)
and described in the following.

| Macro | Description |
|-------|-------------|
| `XAD_VERSION_MAJOR` | Numeric value of the XAD major version number, i.e. `1` for `1.x.y`. |
| `XAD_VERSION_MINOR` | Numeric value of the XAD minor version number, i.e. `4` for `x.4.y`. |
| `XAD_VERSION_PATCH` | Numeric value of the XAD patch version number, without a pre-release suffix. Thus, for `1.0.0-rc1`, the value of this macro is `0`. |
| `XAD_VERSION_PATCHSTRING` | String representation of the XAD patch version number, including a pre-release suffix, e.g. `"0-rc1"`. |
| `XAD_VERSION_SUFFIX` | String with the pre-release suffix, e.g. `"rc1"` for `1.0.0-rc1`; empty (`""`) for a release. |
| `XAD_IS_PRERELEASE` | Defined (without a value) if the version has a pre-release suffix, undefined for a release. |
| `XAD_VERSION` | Numeric representation of the full XAD version number (without pre-release suffixes, e.g. for release candidates and betas). It is computed as:     `XAD_VERSION = XAD_VERSION_MAJOR * 10000 + XAD_VERSION_MINOR * 100 + XAD_VERSION_PATCH` |
| `XAD_VERSION_STRING` | String representation of the full XAD version, including a pre-release suffix, e.g. `"1.1.0-rc1"`. |
