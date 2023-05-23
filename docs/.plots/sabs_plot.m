%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab Plot of the sabs function - for generating the figure for the docs
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2023 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(-0.002, 0.002, 2000);
y = zeros(size(x));
for xi = 1:length(x)
    y(xi) = sabs(x(xi));
end
plot(x,y,'-b', 'LineWidth', 2);
%axis equal
%set(gca, 'FontSize', 14.0, 'XLim', [-0.002, 0.002], 'YLim', [0,0.002]);
grid on
set(gcf, 'Color', [1 1 1])

%dx=5e-7;
%dy = zeros(size(x));
%d2y = zeros(size(x));
%for xi = 1:length(x)
%    yh = sabs(x(xi)+dx);
%    y2h = sabs(x(xi)+2*dx);
%    dy(xi) = (yh-y(xi))/dx;
%    d2y(xi) = (y2h - 2*yh + y(xi))/dx/dx;
%end

%hold on
%plot(x, dy, '-r')
%plot(x, d2y, '-g')


