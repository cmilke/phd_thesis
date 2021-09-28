set parametric
set view equal xyz
set lmargin 0
set zeroaxis
set view ,,1.5
set xyplane at 0
#set grid ztics
set xlabel "z"
set ylabel "x"
set zlabel "y"
#set xrange [-3:3]

s = 0 #pi/2

d0 = .5
z0= 0.3
phi = 2*pi * ( 20 * 1. / 360 )
theta = .8* pi/2
r = 1.

x0 = d0*cos(phi+s)
y0 = d0*sin(phi+s)


xl = -r*cos(phi+s) + x0
yl = -r*sin(phi+s) + y0
x(t) = r * cos(t+s) + xl
y(t) = r * sin(t+s) + yl
z(t) = z0 - (t-phi) * r / tan(theta)
arcx(t) = ( 0 < t && t < phi ) ? (r/2.)*cos(t+s)-(r/2.)*cos(phi+s)+x0: 1/0
arcy(t) = ( 0 < t && t < phi ) ? (r/2.)*sin(t+s)-(r/2.)*sin(phi+s)+y0 : 1/0

set arrow 1 from 0,0,0 to 0,x0,y0
set arrow 2 from z0,0,0 to z0,xl,yl
set arrow 3 from 0,x0,y0 to z0,x0,y0
splot [t=-pi:pi] z(t), x(t), y(t) title 'Track Helix', \
     z(t), xl, yl title 'Helix Axis', \
     0, arcx(t), arcy(t) title 'Phi', \
    '-' using 1:2:3 with points pt 5 title 'PV'
    0 0 0
    e
pause -1
