#!/usr/bin/env python3
"""
Directional 48³ cube
--------------------
• forward (+z) + lateral links; no global back links
• one residual layer z_res gets extra recurrent 6‑neighbour links
• depth‑decaying bias, local refractory+adaptation
"""

import argparse, random, numpy as np
import matplotlib.pyplot as plt, matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from collections import deque

# ---------------- stimulus ---------------------------------------------------
class DotEnv:
    def __init__(self,n,speed=1): self.n,self.s=n,speed; self.reset()
    def reset(self):
        edge=random.choice('tblr')
        if edge=='t': self.x,self.y,self.vx,self.vy=random.randrange(self.n),0, random.choice([-1,1]), self.s
        elif edge=='b':self.x,self.y,self.vx,self.vy=random.randrange(self.n),self.n-1, random.choice([-1,1]),-self.s
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

# ---------------- cube -------------------------------------------------------
class CubeNet:
    def __init__(self, *, n=48, L=48, seed=0,
                 w_fwd=1.0, w_lat=0.35, w_res=0.6, z_res=16,
                 input_gain=4.5,
                 bias0=0.032, lamb=14,
                 rho=0.82, noise=0.0,
                 eta=0.01, decay=0.001,
                 tau_ref=3, tau_adapt=40, d_theta=0.23, theta0=1.0):
        random.seed(seed); np.random.seed(seed)
        self.n,self.L=n,L; self.N=n*n*L
        self.idx=lambda z,y,x:z*n*n+y*n+x

        self.neigh_fwd=[[] for _ in range(self.N)]
        self.neigh_lat=[[] for _ in range(self.N)]
        self.neigh_res=[[] for _ in range(self.N)]

        for z in range(L):
            for y in range(n):
                for x in range(n):
                    i=self.idx(z,y,x)
                    # lateral 4-neighbour
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy,xx=y+dy,x+dx
                        if 0<=yy<n and 0<=xx<n:
                            self.neigh_lat[i].append(self.idx(z,yy,xx))
                    # forward (+z) only
                    if z+1<L: self.neigh_fwd[i].append(self.idx(z+1,y,x))
                    # residual recurrent (same z)
                    if z==z_res:
                        for dz,dy,dx in ((0,1,0),(0,-1,0),(0,0,1),(0,0,-1),(1,0,0),(-1,0,0)):
                            zz,yy,xx=z+dz,y+dy,x+dx
                            if 0<=zz<L and 0<=yy<n and 0<=xx<n:
                                self.neigh_res[i].append(self.idx(zz,yy,xx))

        rng=np.random.default_rng(seed)
        self.W={}
        for i in range(self.N):
            for j in self.neigh_fwd[i]: self.W[(i,j)]=rng.uniform(0.8*w_fwd,1.2*w_fwd)
            for j in self.neigh_lat[i]: self.W[(i,j)]=rng.uniform(0.8*w_lat,1.2*w_lat)
            for j in self.neigh_res[i]: self.W[(i,j)]=rng.uniform(0.8*w_res,1.2*w_res)

        z_vec=np.repeat(np.arange(L), n*n)
        self.bias=bias0*np.exp(-z_vec/lamb)
        self.input_gain=input_gain; self.rho=rho; self.noise=noise
        self.v=np.zeros(self.N); self.adapt=np.zeros(self.N); self.timer=np.zeros(self.N,int)
        self.theta0=theta0; self.tau_ref=tau_ref; self.tau_adapt=tau_adapt; self.d_theta=d_theta
        self.last=np.zeros(self.N); self.eta=eta; self.decay=decay; self.fire_hist=deque(maxlen=200)

    def forward(self,inp):
        I=np.zeros(self.N)
        for y,x in zip(*np.where(inp==1)): I[self.idx(0,y,x)]+=self.input_gain
        for pre in np.where(self.last==1)[0]:
            for q in (*self.neigh_fwd[pre],*self.neigh_lat[pre],*self.neigh_res[pre]):
                I[q]+=self.W[(pre,q)]
        self.v=self.rho*self.v+I+self.bias+np.random.normal(0,self.noise,self.N)
        ref=self.timer>0; self.v[ref]=-1.0; self.timer[ref]-=1
        s=(self.v>=self.theta0+self.adapt).astype(float)
        fired=np.where(s==1)[0]; self.v[fired]=0.0; self.timer[fired]=self.tau_ref
        self.adapt*=(1-1/self.tau_adapt); self.adapt[fired]+=self.d_theta
        return s

    def hebb(self,s):
        pre=np.where(self.last==1)[0]; post=set(np.where(s==1)[0])
        for p in pre:
            for q in (*self.neigh_fwd[p],*self.neigh_lat[p],*self.neigh_res[p]):
                if q in post: self.W[(p,q)]=min(1.2,self.W[(p,q)]+self.eta)
                else:         self.W[(p,q)]=max(0.0,self.W[(p,q)]-self.decay)
        self.last=s

    def step(self,inp): s=self.forward(inp); self.hebb(s); self.fire_hist.append(s.mean()); return s
    def fire_avg(self): return np.mean(self.fire_hist) if self.fire_hist else 0.

# --------------- visual helper ---------------------------------------------
def aggregate(spike_vec,n,L,d):
    nz=np.where(spike_vec==1)[0]
    if nz.size==0: return np.empty((0,3)),np.empty(0)
    step=d; zs,ys,xs=nz//(n*n),(nz//n)%n,nz%n
    key=(zs//step)*(n//step)*(n//step)+(ys//step)*(n//step)+(xs//step)
    uniq,cnt=np.unique(key,return_counts=True)
    cx=(uniq%(n//step))*step+step/2; cy=((uniq//(n//step))%(n//step))*step+step/2
    cz=(uniq//((n//step)*(n//step)))*step+step/2
    return np.vstack([cx,cy,cz]).T,cnt

# ---------------- main ------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ratio',type=int,default=8)
    ap.add_argument('--down', type=int,default=4)
    ap.add_argument('--z_res',type=int,default=16)
    ap.add_argument('--w_res',type=float,default=0.6)
    args=ap.parse_args()

    n=L=48
    env=DotEnv(n)
    net=CubeNet(n=n,L=L,z_res=args.z_res,w_res=args.w_res)

    plt.style.use('ggplot')
    fig=plt.figure(figsize=(10,6)); gs=fig.add_gridspec(1,2,width_ratios=[1,2])
    ax_in=fig.add_subplot(gs[0]); ax_3d=fig.add_subplot(gs[1],projection='3d')
    im_in=ax_in.imshow(env.frame(),cmap='Greys',vmin=0,vmax=1)
    ax_in.set_title('Real input'); ax_in.set_xticks([]); ax_in.set_yticks([])
    scat_in=ax_3d.scatter([],[],[],s=80,c='r',marker='s')
    scat_h =ax_3d.scatter([],[],[],s=[],c='b',alpha=0.65)
    ax_3d.set(xlim=(0,n),ylim=(0,n),zlim=(0,L))
    ax_3d.text(n+2,n/2,0,'input z=0',color='r')
    txt=fig.text(0.02,0.95,'')

    cur=env.frame(); sub=0
    def upd(t):
        nonlocal cur,sub
        if sub==0:
            cur=env.step()
            yx=np.where(cur==1); scat_in._offsets3d=(yx[1],yx[0],np.zeros_like(yx[0]))
        sub=(sub+1)%args.ratio
        s=net.step(cur)
        pts,cnt=aggregate(s,n,L,args.down)
        scat_h._offsets3d=(pts[:,0],pts[:,1],pts[:,2]) if cnt.size else ([],[],[])
        scat_h.set_sizes(cnt*4 if cnt.size else [])
        im_in.set_data(cur)
        txt.set_text(f'step {t} | fire {net.fire_avg():.3f}')
        return im_in,scat_in,scat_h,txt

    ani=anim.FuncAnimation(fig,upd,interval=60,blit=False); globals()['_keep']=ani
    plt.tight_layout(); plt.show()

if __name__=='__main__': main()
