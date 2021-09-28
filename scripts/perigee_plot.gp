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
phi = 2*pi * ( 60 * 1. / 360 )
theta = 0.7* pi/2
r = 1.

x0 = d0*cos(phi+s)
y0 = d0*sin(phi+s)


xl = -r*cos(phi+s) + x0
yl = -r*sin(phi+s) + y0
xm(r,t) = r*cos(t)
ym(r,t)= r*sin(t)
xm2(r,t)=r*sin(t)+xl 
zm(r,t)=r*cos(t)+z(pi/2) 
x(t) = r * cos(t+s) + xl
y(t) = r * sin(t+s) + yl
z(t) = z0 - (t-phi) * r / tan(theta)
arcx(t) = ( 0 <= t && t <= phi ) ? xm(r/4,t) : 1/0
arcy(t) = ( 0 <= t && t <= phi ) ? ym(r/4,t) : 1/0
arcx2(t) = ( theta <= t && t <= pi/2 ) ? xm2(r/4,t) : 1/0
arcz(t) = ( theta <= t && t <= pi/2 ) ? zm(r/4,t) : 1/0

set label 1 '   PV' at 0,0,0 point pointtype 5 lc rgb 'black' textcolor rgb 'black'
set label 2 '   PoCA' at z0,x0,y0 point pointtype 5 lc rgb 'dark-red' textcolor rgb 'dark-red'
set label 3 'd0   ' right at 0,3*x0/4,3*y0/4 textcolor rgb 'blue'
set label 4 'z0' at z0/2,1.1*x0,1.1*y0 textcolor rgb 'orange'
set label 5 at 0,xm(r/3,phi/2),ym(r/3,phi/2) '{/Symbol j}' font ',10' textcolor  rgb 'orchid'
set label 6 at zm(r/3,((pi/2)+theta)/2),xm2(r/3,((pi/2)+theta)/2),y(0) '{/Symbol q}' font ',10' textcolor rgb 'royalblue'
set label 7 at z(-pi/4),x(-pi/4),y(-pi/4) '   r=qB/p' font ',10' textcolor rgb 'red'

set arrow 1 from 0,0,0 to 0,x0,y0 lc rgb 'blue'
set arrow 2 from 0,x(phi),y(phi) to z0,x(phi),y(phi) lc rgb 'orange'
set arrow 3 from z(pi/2),xl,yl to z(pi/2),x(0),y(0)
set arrow 4 from z(pi/2),xl,yl to (x(0)-x(pi/2))/tan(theta)+z(pi/2),x(0),y(0)
set arrow 5 from z(-pi/4),xl,yl to z(-pi/4),x(-pi/4),y(-pi/4) lc rgb 'red'

splot [t=-pi:pi] z(t), x(t), y(t) title 'Track Helix' lc rgb 'dark-violet', \
     z(t), xl, yl title 'Helix Axis' lc rgb 'dark-green', \
     0, arcx(t), arcy(t) notitle lc rgb 'orchid', \
     arcz(t), arcx2(t), y(0) notitle lc rgb 'royalblue'
pause -1
