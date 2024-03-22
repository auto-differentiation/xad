%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab Plot of the sstep function - for generating the figure for the docs
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(-0.003, 0.003, 4000);
y = zeros(size(x));
for xi = 1:length(x)
    y(xi) = sstep(x(xi));
end
plot(x,y,'-b', 'LineWidth', 2);
%axis equal
set(gca, 'FontSize', 14.0, 'XLim', [-0.003, 0.003], 'YLim', [-0.1,1.1]);
grid on
set(gcf, 'Color', [1 1 1])
