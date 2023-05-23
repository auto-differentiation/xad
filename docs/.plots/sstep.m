%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab sstep function based on smin and smax
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2023 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = sstep(x)
if (x < 0)
    y = 500*smax(x+0.001, 0);
else
    y = 500 *smin(x-0.001, 0) + 1;
end
