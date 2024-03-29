%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Matlab Plot of the smax function - for generating the figure for the docs
%
%  This file is part of the XAD user manual.
%  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
%  See the file index.rst for copying condition. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = linspace(-0.002, 0.002, 200);
x2 = linspace(-0.002, 0.002, 200);
y = zeros(length(x1), length(x2));
for x1i = 1:length(x1)
    for x2i = 1:length(x2)
        y(x1i,x2i) = smax(x1(x1i),x2(x2i));
    end
end
%plot(x,y,'-b', 'LineWidth', 2);
%axis equal
%set(gca, 'FontSize', 14.0, 'XLim', [-0.002, 0.002], 'YLim', [0,0.002]);
%grid on
%set(gcf, 'Color', [1 1 1])

dx=5e-7;
dx1x1 = zeros(size(y));
dx1dx2 = zeros(size(y));
dx2x2 = zeros(size(y));

for x1i = 1:length(x1)
    for x2i = 1:length(x2)
        yx1ph = smax(x1(x1i)+dx, x2(x2i));
        yx2ph = smax(x1(x1i), x2(x2i)+dx);
        yx1mh = smax(x1(x1i)-dx, x2(x2i));
        yx2mh = smax(x1(x1i), x2(x2i)-dx);
        yx1phx2ph = smax(x1(x1i)+dx, x2(x2i)+dx);
        yx1mhx2mh = smax(x1(x1i)-dx, x2(x2i)-dx);

        dx1x1(x1i,x2i) = (yx1ph - 2*y(x1i,x2i) + yx1mh) / dx / dx;
        dx2x2(x1i,x2i) = (yx2ph - 2*y(x1i,x2i) + yx2mh) / dx / dx; 
        dx1dx2(x1i,x2i) = (yx1phx2ph - yx1ph -yx2ph + 2*y(x1i,x2i) - yx1mh - yx2mh + yx1mhx2mh) / (2*dx*dx);
    end
end

[X1,X2] = meshgrid(x1, x2);
surf(X1, X2, dx1dx2)

