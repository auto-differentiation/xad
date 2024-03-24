##############################################################################
#
#  Build file for extension module - using pre-built binary with pybind.
#
#  This was inspired by:
#  https://github.com/tim-mitchell/prebuilt_binaries/tree/main
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

from distutils.file_util import copy_file
import os
from pathlib import Path

try:
    from setuptools import Extension as _Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.command.build_ext import (  # type: ignore[assignment]
        build_ext as _build_ext,
    )
    from distutils.extension import Extension as _Extension  # type: ignore[assignment]


class XadExtension(_Extension):
    """Extension module for XAD, using the pre-built file from CMAKE instead of
    actually building it."""

    def __init__(self, name: str, input_file: str):
        filepath = Path(input_file)
        if not filepath.exists():
            raise ValueError(f"extension file {input_file} does not exist")
        self.input_file = input_file

        super().__init__(f"xad_autodiff.{name}", ["dont-need-this-source-file.c"])


class build_ext(_build_ext):
    """Overrides build_ext to simply copy the file built with CMake into the
    right location, rather than actually building it"""

    def run(self):
        for ext in self.extensions:
            if not isinstance(ext, XadExtension):
                raise ValueError("Only pre-built extensions supported")

            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            dest_path = Path(self.build_lib) / "xad_autodiff"
            dest_path.mkdir(parents=True, exist_ok=True)
            dest_filename = dest_path / os.path.basename(filename)

            copy_file(
                ext.input_file,
                dest_filename,
                verbose=self.verbose,
                dry_run=self.dry_run,
            )

        if self.inplace:
            self.copy_extensions_to_source()


def build(setup_kwargs: dict):
    """Main extension build command. Needs to have the file `prebuilt_file.txt`
    in the same directory, holding the file name of the dynamic library that has 
    been prebuilt, so that it can copy it to the right location."""

    prebuilt_file = Path(__file__).parent / "prebuilt_file.txt"
    with open(prebuilt_file, "rt") as f:
        filename = f.read().strip()
    ext_modules = [XadExtension("_xad_autodiff", filename)]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
            "zip_safe": False,
        }
    )
