%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab version of the smax function
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2022 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = smax(a, b)
y = 0.5*(a+b+sabs(a-b));
