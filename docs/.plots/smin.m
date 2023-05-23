%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab smin function - for generating the figure for the docs
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2023 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = smin(a, b)
y = 0.5*(a+b-sabs(a-b));
