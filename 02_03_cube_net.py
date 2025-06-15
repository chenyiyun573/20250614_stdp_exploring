#!/usr/bin/env python3
"""
48³ isotropic cube, input‑focused
---------------------------------
• input plane injects ‘input_gain’ current (default 3.0)
• weaker recurrent weights (w0≈0.6) and faster leak (rho=0.78)
• longer refractory & stronger adaptation → inner waves damp quickly
• no global quota; pure local refractory + adaptation
• network updated <ratio> times per single dot movement
"""

import argparse, random, numpy as np
import matplotlib.pyplot as plt, matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from collections import deque

# -------------------- dot input --------------------------------------------
class DotEnv:
    def __init__(self,n,speed=1): self.n,self.s=n,speed; self.reset()
    def reset(self):
        edge=random.choice('tblr')
        if edge=='t': self.x,self.y,self.vx,self.vy=random.randrange(self.n),0,random.choice([-1,1]), self.s
        elif edge=='b':self.x,self.y,self.vx,self.vy=random.randrange(self.n),self.n-1,random.choice([-1,1]),-self.s
        elif edge=='l':self.x,self.y,self.vx,self.vy=0,random.randrange(self.n), self.s,random.choice([-1,1])
        else: self.x,self.y,self.vx,self.vy=self.n-1,random.randrange(self.n),-self.s,random.choice([-1,1])
    def frame(self):
        img=np.zeros((self.n,self.n),np.int8); img[self.y,self.x]=1; return img
    def step(self):
        self.x+=self.vx; self.y+=self.vy
        if self.x<0:self.x,self.vx=-self.x, self.s
        if self.x>=self.n:self.x,self.vx=2*self.n-2-self.x,-self.s
        if self.y<0:self.y,self.vy=-self.y, self.s
        if self.y>=self.n:self.y,self.vy=2*self.n-2-self.y,-self.s
        return self.frame()

# -------------------- isotropic cube ----------------------------------------
class CubeNet:
    def __init__(self,n=48,L=48,seed=0,
                 w0=0.6,input_gain=3.0,
                 rho=0.78,bias=0.015,noise=0.0,
                 eta=0.01,decay=0.001,
                 tau_ref=3,tau_adapt=40,d_theta=0.35,theta0=1.0):
        random.seed(seed); np.random.seed(seed)
        self.n,self.L,self.N=n,L,n*n*L; self.idx=lambda z,y,x:z*n*n+y*n+x
        # neighbours
        self.neigh=[[] for _ in range(self.N)]
        for z in range(L):
            for y in range(n):
                for x in range(n):
                    i=self.idx(z,y,x)
                    for dz,dy,dx in ((0,1,0),(0,-1,0),(0,0,1),(0,0,-1),(1,0,0),(-1,0,0)):
                        zz,yy,xx=z+dz,y+dy,x+dx
                        if 0<=zz<L and 0<=yy<n and 0<=xx<n:
                            self.neigh[i].append(self.idx(zz,yy,xx))
        rng=np.random.default_rng(seed)
        self.W={(i,j):rng.uniform(0.8*w0,1.2*w0) for i in range(self.N) for j in self.neigh[i]}
        # state
        self.v=np.zeros(self.N); self.theta0=theta0
        self.adapt=np.zeros(self.N); self.timer=np.zeros(self.N,int); self.last=np.zeros(self.N)
        self.input_gain=input_gain
        self.rho=rho; self.bias=bias; self.noise=noise
        self.eta=eta; self.decay=decay
        self.tau_ref=tau_ref; self.tau_adapt=tau_adapt; self.d_theta=d_theta
        self.fire_hist=deque(maxlen=200)

    def forward(self,inp):
        I=np.zeros(self.N)
        for y,x in zip(*np.where(inp==1)):
            I[self.idx(0,y,x)] += self.input_gain          # stronger external drive
        for pre in np.where(self.last==1)[0]:
            for q in self.neigh[pre]: I[q]+=self.W[(pre,q)]
        self.v=self.rho*self.v+I+self.bias+np.random.normal(0,self.noise,self.N)
        ref=self.timer>0; self.v[ref]=-1.0; self.timer[ref]-=1
        s=(self.v>=self.theta0+self.adapt).astype(float)
        fired=np.where(s==1)[0]
        self.v[fired]=0.0; self.timer[fired]=self.tau_ref
        self.adapt*=(1-1/self.tau_adapt); self.adapt[fired]+=self.d_theta
        return s

    def hebb(self,s):
        pre=np.where(self.last==1)[0]; post=set(np.where(s==1)[0])
        for p in pre:
            for q in self.neigh[p]:
                if q in post: self.W[(p,q)]=min(1.2,self.W[(p,q)]+self.eta)
                else:         self.W[(p,q)]=max(0.0,self.W[(p,q)]-self.decay)
        self.last=s

    def step(self,inp):
        s=self.forward(inp); self.hebb(s); self.fire_hist.append(s.mean()); return s
    def fire_avg(self): return np.mean(self.fire_hist) if self.fire_hist else 0.

# -------------------- aggregation -------------------------------------------
def aggregate(s,n,L,d):
    nz=np.where(s==1)[0]; step=d
    if not nz.size: return np.empty((0,3)),np.empty(0)
    zs,ys,xs = nz//(n*n), (nz//n)%n, nz%n
    key=(zs//step)*(n//step)*(n//step)+(ys//step)*(n//step)+(xs//step)
    uniq,cnt=np.unique(key,return_counts=True)
    cx=(uniq%(n//step))*step+step/2
    cy=((uniq//(n//step))%(n//step))*step+step/2
    cz=(uniq//((n//step)*(n//step)))*step+step/2
    return np.vstack([cx,cy,cz]).T,cnt

# -------------------- main ---------------------------------------------------
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--ratio',type=int,default=8)
    pa.add_argument('--down', type=int,default=4)
    pa.add_argument('--bias', type=float,default=0.015)
    pa.add_argument('--theta0',type=float,default=1.0)
    pa.add_argument('--d_theta',type=float,default=0.35)
    pa.add_argument('--rho',   type=float,default=0.78)
    pa.add_argument('--tau_ref',type=int,default=3)
    pa.add_argument('--input_gain',type=float,default=3.0)
    pa.add_argument('--w0',type=float,default=0.6)
    args=pa.parse_args()

    n=L=48
    env=DotEnv(n)
    net=CubeNet(n,L,bias=args.bias,theta0=args.theta0,d_theta=args.d_theta,
                rho=args.rho,tau_ref=args.tau_ref,input_gain=args.input_gain,
                w0=args.w0)

    plt.style.use('ggplot')
    fig=plt.figure(figsize=(10,6)); gs=fig.add_gridspec(1,2,width_ratios=[1,2])
    ax_in=fig.add_subplot(gs[0]); ax_3d=fig.add_subplot(gs[1],projection='3d')
    im_in=ax_in.imshow(env.frame(),cmap='Greys',vmin=0,vmax=1)
    ax_in.set_title('Real input'); ax_in.set_xticks([]); ax_in.set_yticks([])
    scat_in=ax_3d.scatter([],[],[],s=80,c='r',marker='s')
    scat_h =ax_3d.scatter([],[],[],s=[],c='b',alpha=0.6)
    ax_3d.set(xlim=(0,n),ylim=(0,n),zlim=(0,L))
    ax_3d.text(n+2,n/2,0,'input z=0',color='r')
    txt=fig.text(0.02,0.95,'')

    cur_in=env.frame(); sub=0
    def upd(t):
        nonlocal cur_in,sub
        if sub==0:
            cur_in=env.step()
            yx=np.where(cur_in==1)
            scat_in._offsets3d=(yx[1],yx[0],np.zeros_like(yx[0]))
        sub=(sub+1)%args.ratio
        s=net.step(cur_in)
        pts,cnt=aggregate(s,n,L,args.down)
        scat_h._offsets3d=(pts[:,0],pts[:,1],pts[:,2]) if cnt.size else ([],[],[])
        scat_h.set_sizes(cnt*4 if cnt.size else [])
        im_in.set_data(cur_in)
        txt.set_text(f'step {t} | fire {net.fire_avg():.3f}')
        return im_in,scat_in,scat_h,txt

    ani=anim.FuncAnimation(fig,upd,interval=60,blit=False); globals()['_keep']=ani
    plt.tight_layout(); plt.show()

if __name__=='__main__': main()
