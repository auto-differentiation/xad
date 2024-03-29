%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab Plot of the sabs function - for generating the figure for the docs
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function y = sabs(x)
 
    c = 0.001;
    if (x > c || x < -c)
        y = abs(x);
    elseif (x >= 0)
        
        y = x^2*(2/c - 1/c^2 * x);
    else
        y = x^2*(2/c + 1/c^2 * x);
    end

end
