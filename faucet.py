import numpy as np 
from matplotlib import pyplot as plt
#import matplotlib as mpl

np.random.seed(419)

Y_MIN, Y_MAX = 0, 1300
X_MIN, X_MAX = 0, 2100
tort = 0.0035
to_geo = lambda r,c: (lambda y,x: (y*(Y_MAX-Y_MIN)+Y_MIN, 
                                   x*(X_MAX-X_MIN)+X_MIN))(
                      r + tort*np.cos( 8*(r*r+c*c)), c + tort*np.sin(10*(r*r+c*c))
                     ) 

NB_PIPES            = 4000
NB_FAUCETS_PER_PIPE =   50
PIPE_LENGTH         = 0.15
PIPE_WIDTH          = 0.0004  
INF_RATE            = 0.1
SAMP_RATE           = 0.02 
LONG_DIAG_PROB      = 0.1
FAUCET_NOISE        = 0.1

district_centers = [(400,400),(300,1100),(800,400),( 700, 700)]
which_district = lambda y,x: (sorted([((y-yy)**2+(x-xx)**2, i) for i,(yy,xx) in enumerate(district_centers)])[-1][1]) 

                            
min_distance = lambda y,x,ddy,ddx : max(0,min(
       ((y+0.0*ddy-1.0)**2+(x+0.0*ddx-1.0)**2 + 0.9*(y+0.0*ddy-1.0)**3 - 0.1*(x+0.0*ddx-0.0)**4,
        (y+0.2*ddy-1.0)**2+(x+0.2*ddx-1.0)**2 + 0.9*(y+0.2*ddy-1.0)**3 - 0.1*(x+0.2*ddx-0.0)**4,
        (y+0.5*ddy-1.0)**2+(x+0.5*ddx-1.0)**2 + 0.9*(y+0.5*ddy-1.0)**3 - 0.1*(x+0.5*ddx-0.0)**4,
        (y+0.8*ddy-1.0)**2+(x+0.8*ddx-1.0)**2 + 0.9*(y+0.8*ddy-1.0)**3 - 0.1*(x+0.8*ddx-0.0)**4,
        (y+1.0*ddy-1.0)**2+(x+1.0*ddx-1.0)**2 + 0.9*(y+1.0*ddy-1.0)**3 - 0.1*(x+1.0*ddx-0.0)**4,)))

snap = lambda x,N: x if np.random.random()<0.5 else round(N*x)/N 

inf_factor = lambda y,x: (1.0 + np.cos(10*(y+x))) #- 0.4*np.cos(13*(y**2+x**2)))#**2
snap_by_loc = lambda y,x:(
    (lambda dd: (100,200) if dd<0.5 else (50,100) if dd<1.0 else (25, 50))((y-0.7)**2+(x-0.7)**2)) 
#angle_by_loc = lambda y,x: which_district()

pipes   = [
    (y,x,y+ddy,x+ddx, 1 if np.random.random()<INF_RATE*inf_factor(y+0.05*ddy,x+0.05*ddx) else 0)
    for _ in range(NB_PIPES)
    for yy in [np.random.laplace()*0.2+0.7] ###
    for xx in [np.random.laplace()*0.2+0.7] ###
    for y  in [snap(yy,snap_by_loc(yy,xx)[0])] ###
    for x  in [snap(xx,snap_by_loc(yy,xx)[1])] ###
    for is_long in [np.random.binomial(1, LONG_DIAG_PROB)]
    for dy in [snap(np.random.laplace()*[0.5,2.0][is_long]*PIPE_LENGTH,snap_by_loc(yy,xx)[0])]
    for dx in [snap(np.random.laplace()*[0.5,2.0][is_long]*PIPE_LENGTH,snap_by_loc(yy,xx)[1])]
    for dddy in [(dy if is_long or abs(dy)>abs(dx) else 0 )]
    for dddx in [(dx if is_long or abs(dx)>abs(dy) else 0 )]
    for angle in [[0.00,0.01,0.06,0.02][which_district(*to_geo(y,x))]]
    for ddy in [ np.cos(angle)*dddy - np.sin(angle)*dddx]
    for ddx in [ np.sin(angle)*dddy + np.cos(angle)*dddx]
    if np.random.random() < (min_distance(y,x,ddy,ddx)-0.15)/0.1
]

trunc = lambda W: (lambda x: (x+(1 if x>0 else -1)))(np.random.laplace()*W)

faucets = [
  #[
    (a*(y1-y0)+y0 + PIPE_WIDTH*trunc(0.3),
     a*(x1-x0)+x0 + PIPE_WIDTH*trunc(0.3), inf+(1-2*inf)*(np.random.binomial(1,FAUCET_NOISE)))
  #]
  for (y0,x0,y1,x1, inf) in pipes
    for a in np.random.random(int(np.random.random()*NB_FAUCETS_PER_PIPE))
]

img = np.ones([int(Y_MAX),int(X_MAX),3], dtype=np.float)
img[:,:] = [1.0,1.0,0.9]

for y in range(0+6,Y_MAX-6,12):
    for x in range(0+6,X_MAX-6,12):
        distances = np.array([np.sqrt((y-yc)**2+(x-xc)**2) for yc,xc in district_centers])
        colors = np.array([(0.98,0.98,0.87),(1.00,0.98,0.87),(0.98,1.00,0.87),(1.00,1.00,0.87)]) 
        d = np.exp(-40*distances/1000.0)
        s = np.sum(d)
        cc = np.array([0.0, 0.0, 0.0])
        for c,dd in zip(colors, d):
            cc += c*dd/s
        img[y-6:y+6, x-6:x+6] = cc   
 
sigm = lambda x : 1.0/(1.0 + np.exp(-x))
for y in range(0+6,int(Y_MAX-3),3):
    for x in range(0+6,int(X_MAX-3),3):
        distance = min_distance(y/Y_MAX,x/X_MAX,0,0)
        img[y-3:y+0, x-3:x+0] += sigm(320*(0.145-distance)) * (np.array((0.78,0.95,1.00)) - img[y-3:y+0, x-3:x+0])

for (y_,x_,inf) in faucets:
    y,x = to_geo(y_,x_)
    y,x = int(y), int(x)
    if not (Y_MIN+1<=y<Y_MAX-1) or not (X_MIN+1<=x<X_MAX-1): continue
    img[y, x] = [0.3,0.3,0.4]#[0.0,0.0,0.1]#

    if not ((925<y<1075) and (926<x<1076)): continue
    y=int(3.5*(y-925)+750)
    x=int(3.5*(x-926)+1550)
    img[y-1:y+2, x-1:x+2] = [0.3,0.3,0.4]#[0.0,0.0,0.1]


for (y_,x_,inf) in faucets:
    y,x = to_geo(y_,x_)
    y,x = int(y), int(x)
    if not (Y_MIN+1<=y<Y_MAX-1) or not (X_MIN+1<=x<X_MAX-1): continue
    if ((( 925<y<1075) and (926<x<1076) and np.random.random()< 0.5        ) or
        (                                 np.random.random()<   SAMP_RATE)   ):
        if inf:
            img[y        , x        ] += 0.8*([0.8,0.0,0.5]-img[y        , x        ])
            img[y-1:y+2:2, x-1:x+2:2] += 0.8*([0.8,0.0,0.5]-img[y-1:y+2:2, x-1:x+2:2])
            img[y-2:y+3:4, x-2:x+3:4] += 0.8*([0.8,0.0,0.5]-img[y-2:y+3:4, x-2:x+3:4])
        else:
            img[y-2:y+3  , x        ] += 0.8*([0.0,0.6,0.9]-img[y-2:y+3  , x        ])
            img[y        , x-2:x+3  ] += 0.8*([0.0,0.6,0.9]-img[y        , x-2:x+3  ])

        if not ((925<y<1075) and (926<x<1076)): continue
        y=int(3.5*(y-925)+750)
        x=int(3.5*(x-926)+1550)
        if inf:
            img[y        , x        ] += 0.8*([0.8,0.0,0.5]-img[y        , x        ])
            img[y-1:y+2:2, x-1:x+2:2] += 0.8*([0.8,0.0,0.5]-img[y-1:y+2:2, x-1:x+2:2])
            img[y-2:y+3:4, x-2:x+3:4] += 0.8*([0.8,0.0,0.5]-img[y-2:y+3:4, x-2:x+3:4])
            img[y-3:y+4:6, x-3:x+4:6] += 0.8*([0.8,0.0,0.5]-img[y-3:y+4:6, x-3:x+4:6])
        else:
            img[y-3:y+4  , x        ] += 0.8*([0.0,0.6,0.9]-img[y-3:y+4  , x        ])
            img[y        , x-3:x+4  ] += 0.8*([0.0,0.6,0.9]-img[y        , x-3:x+4  ])

img[925    , 926:1076] = [0.8,0.5,0.1]
img[    1075, 926:1076] = [0.8,0.5,0.1]
img[925:1075, 926    ] = [0.8,0.5,0.1]
img[925:1075,     1076] = [0.8,0.5,0.1]

img[int(3.5*  0+ 750)                  , int(3.5*  0+1550):int(3.5*150+1550)] = [0.8,0.5,0.1]
img[                  int(3.5*150+ 750), int(3.5*  0+1550):int(3.5*150+1550)] = [0.8,0.5,0.1]
img[int(3.5*  0+ 750):int(3.5*150+ 750), int(3.5*  0+1550)                  ] = [0.8,0.5,0.1]
img[int(3.5*  0+ 750):int(3.5*150+ 750),                   int(3.5*150+1550)] = [0.8,0.5,0.1]

plt.imsave('hei.png', img)

# TODO: infection model 
#       low frequency spatial variation in  
#       sparse infection measurement noise on top of everything else
#       pipe-pipe interactions (some intersection geometry, kd trees useful)
#
#
# TODO: input-data distribution:
#       low frequency spatial variation in grid parameters such as avg road length, squareness and T-flushness of pipe spacing 
#       near beach, roads align with beach; large central pipe cutting diagonally to beach 
#       add dense altitude information 
#       add sparse localized tags for building age,  
#

# + 0.01*np.cos(18*(x+0.0*ddx+(y+0.0*ddy))) + 0.005*np.sin(30*(x+0.0*ddx-(y+0.0*ddy)))
# + 0.01*np.cos(18*(x+0.2*ddx+(y+0.2*ddy))) + 0.005*np.sin(30*(x+0.2*ddx-(y+0.2*ddy)))
# + 0.01*np.cos(18*(x+0.5*ddx+(y+0.5*ddy))) + 0.005*np.sin(30*(x+0.5*ddx-(y+0.5*ddy)))
# + 0.01*np.cos(18*(x+0.8*ddx+(y+0.8*ddy))) + 0.005*np.sin(30*(x+0.8*ddx-(y+0.8*ddy)))
# + 0.01*np.cos(18*(x+1.0*ddx+(y+1.0*ddy))) + 0.005*np.sin(30*(x+1.0*ddx-(y+1.0*ddy)))

#-2.5/np.sqrt(1+960*((y+0.0*ddy)-(x+0.0*ddx)**2-0.5)**2) + 
#-2.5/np.sqrt(1+960*((y+0.2*ddy)-(x+0.2*ddx)**2-0.5)**2) + 
#-2.5/np.sqrt(1+960*((y+0.5*ddy)-(x+0.5*ddx)**2-0.5)**2) + 
#-2.5/np.sqrt(1+960*((y+0.8*ddy)-(x+0.8*ddx)**2-0.5)**2) + 
#-2.5/np.sqrt(1+960*((y+1.0*ddy)-(x+1.0*ddx)**2-0.5)**2) + 
