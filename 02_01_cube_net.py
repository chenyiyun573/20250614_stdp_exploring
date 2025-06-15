#!/usr/bin/env python3
"""
48³ cubic spiking network, direction‑selective weights
------------------------------------------------------
• stronger forward (+z) synapses, weak lateral, very weak backward
• very low bias/noise -> spikes mainly input‑driven
• reward also punishes global over‑activity
"""

import argparse, random
import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import deque

# -------- Dot environment ----------------------------------------------------
class DotEnv:
    def __init__(self, n:int, speed:int=1):
        self.n, self.speed = n, speed; self.reset()
    def reset(self):
        edge=random.choice(['top','bottom','left','right'])
        if edge in ('top','bottom'):
            self.x,self.y=random.randrange(self.n),0 if edge=='top' else self.n-1
            self.vx,self.vy=random.choice([-1,1]), self.speed if edge=='top' else -self.speed
        else:
            self.y,self.x=random.randrange(self.n),0 if edge=='left' else self.n-1
            self.vy,self.vx=random.choice([-1,1]), self.speed if edge=='left' else -self.speed
    def frame(self):
        img=np.zeros((self.n,self.n),dtype=np.int8); img[self.y,self.x]=1; return img
    def step(self):
        self.x+=self.vx; self.y+=self.vy
        if self.x<0:self.x,self.vx=-self.x,+self.speed
        if self.x>=self.n:self.x,self.vx=2*self.n-2-self.x,-self.speed
        if self.y<0:self.y,self.vy=-self.y,+self.speed
        if self.y>=self.n:self.y,self.vy=2*self.n-2-self.y,-self.speed
        return self.frame()

# -------- 48³ cube network ---------------------------------------------------
class CubeNet:
    def __init__(self, n=48, L=48, seed=0,
                 w_fwd=1.0, w_lat=0.3, w_back=0.05,
                 bias=0.01, noise=0.0,
                 rho=0.8, eta=0.01, eta_theta=0.01,
                 target=0.05, tau_out=1):
        random.seed(seed); np.random.seed(seed)
        self.n,self.L=n,L; self.N=n*n*L
        self.idx=lambda z,y,x: z*n*n+y*n+x
        # neighbour list + typed direction
        self.neigh_fwd=[[] for _ in range(self.N)]
        self.neigh_lat=[[] for _ in range(self.N)]
        self.neigh_back=[[] for _ in range(self.N)]
        for z in range(L):
            for y in range(n):
                for x in range(n):
                    i=self.idx(z,y,x)
                    for dz,dy,dx in ((0,1,0),(0,-1,0),(0,0,1),(0,0,-1),(1,0,0),(-1,0,0)):
                        zz,yy,xx=z+dz,y+dy,x+dx
                        if 0<=zz<L and 0<=yy<n and 0<=xx<n:
                            j=self.idx(zz,yy,xx)
                            if dz==1: self.neigh_fwd[i].append(j)
                            elif dz==-1: self.neigh_back[i].append(j)
                            else: self.neigh_lat[i].append(j)
        # synaptic weights dictionary
        rng=np.random.default_rng(seed)
        self.W={}
        for i in range(self.N):
            for j in self.neigh_fwd[i]:  self.W[(i,j)]=rng.uniform(w_fwd*0.8,w_fwd*1.2)
            for j in self.neigh_lat[i]:  self.W[(i,j)]=rng.uniform(w_lat*0.8,w_lat*1.2)
            for j in self.neigh_back[i]: self.W[(i,j)]=rng.uniform(w_back*0.8,w_back*1.2)
        self.v=np.zeros(self.N); self.theta=np.full(self.N,1.0)
        self.last=np.zeros(self.N)
        self.bias=bias; self.noise=noise
        self.rho=rho; self.eta=eta; self.eta_theta=eta_theta
        self.target=target; self.tau_out=tau_out
        self.fire_hist,self.iou_hist=deque(maxlen=100),deque(maxlen=100)

    def forward(self, inp):
        I=np.zeros(self.N)
        for y,x in zip(*np.where(inp==1)): I[self.idx(0,y,x)]+=1.0
        pre=np.where(self.last==1)[0]
        for p in pre:
            for q in (*self.neigh_fwd[p],*self.neigh_lat[p],*self.neigh_back[p]):
                I[q]+=self.W[(p,q)]
        I+=self.bias+np.random.normal(0,self.noise,self.N)
        self.v=self.rho*self.v+I
        s=(self.v>=self.theta).astype(float); self.v[s==1]=0
        return s

    def learn(self,s,target_img,reward_global):
        pre=np.where(self.last==1)[0]; post=set(np.where(s==1)[0])
        for p in pre:
            for q in self.neigh_fwd[p]:
                if q in post: self.W[(p,q)]=min(1.2,self.W[(p,q)]+self.eta*reward_global)
            for q in self.neigh_lat[p]:
                if q in post: self.W[(p,q)]=min(1.2,self.W[(p,q)]+self.eta*0.3*reward_global)
            for q in self.neigh_back[p]:
                if q in post: self.W[(p,q)]=min(1.2,self.W[(p,q)]+self.eta*0.1*reward_global)
        self.theta+=self.eta_theta*(s-self.target)
        np.clip(self.theta,0.2,3.0,out=self.theta)
        self.last=s

    def step(self, inp, target):
        s=self.forward(inp)
        pred=np.zeros_like(inp)
        z_back=self.L-1; start=z_back*self.n*self.n
        idx=np.where(s[start:start+self.n*self.n]==1)[0]
        for k in idx: y,x=divmod(k,self.n); pred[y,x]=1
        # reward
        fire=s.mean()
        inter=np.logical_and(pred,target).sum(); union=np.logical_or(pred,target).sum()
        iou=inter/union if union else 1.
        reward=+1 if (iou>=0.8 and fire<2*self.target) else -1
        self.learn(s,target,reward)
        self.fire_hist.append(fire); self.iou_hist.append(iou)
        return pred,s

    def diag(self): return np.mean(self.fire_hist), np.mean(self.iou_hist)

# ---------- helper -----------------------------------------------------------
def agg_spikes(s, n, L, down):
    nz=np.where(s==1)[0]; step=down
    if not nz.size: return np.empty((0,3)), np.empty(0)
    zs,ys,xs = nz//(n*n), (nz//n)%n, nz%n
    key=(zs//step)*(n//step)*(n//step)+(ys//step)*(n//step)+(xs//step)
    uniq,cnt=np.unique(key,return_counts=True)
    cx=(uniq%(n//step))*step+step/2
    cy=((uniq//(n//step))%(n//step))*step+step/2
    cz=(uniq//((n//step)*(n//step)))*step+step/2
    return np.vstack([cx,cy,cz]).T, cnt

def arr(x): return np.asarray(x,dtype=float).reshape(-1)

# ---------- main -------------------------------------------------------------
def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--down',type=int,default=4)
    pa.add_argument('--seed',type=int,default=0); args=pa.parse_args()
    n=48; L=48
    env=DotEnv(n); net=CubeNet(n,L,seed=args.seed)
    plt.style.use('ggplot')
    fig=plt.figure(figsize=(12,6)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,2])
    ax_in=fig.add_subplot(gs[0]); ax_pr=fig.add_subplot(gs[1])
    ax_3d=fig.add_subplot(gs[2],projection='3d')
    im_in=ax_in.imshow(env.frame(),cmap='Greys',vmin=0,vmax=1)
    im_pr=ax_pr.imshow(np.zeros((n,n)),cmap='Greys',vmin=0,vmax=1)
    for a,t in zip((ax_in,ax_pr),('Real input','Prediction (z=47)')):
        a.set_title(t); a.set_xticks([]); a.set_yticks([])
    scat_in=ax_3d.scatter([],[],[],s=60,c='r',marker='s')
    scat_out=ax_3d.scatter([],[],[],s=60,c='g',marker='s')
    scat_h=ax_3d.scatter([],[],[],s=[],c='b',alpha=0.6)
    ax_3d.set(xlim=(0,n),ylim=(0,n),zlim=(0,L))
    ax_3d.text(n+2,n/2,0,'input z=0',color='r')
    ax_3d.text(n+2,n/2,L,'output z=47',color='g')
    txt=fig.text(0.02,0.95,'')
    X_prev=env.frame()
    def update(t):
        nonlocal X_prev
        X_now=env.step(); pred,s=net.step(X_prev,X_now)
        im_in.set_data(X_now); im_pr.set_data(pred)
        iny,inx=np.where(X_now==1); scat_in._offsets3d=(inx,iny,np.zeros_like(inx))
        oy,ox=np.where(pred==1); scat_out._offsets3d=(ox,oy,np.full_like(ox,L-1))
        pts,cnt=agg_spikes(s,n,L,args.down)
        if cnt.size:
            scat_h._offsets3d=(pts[:,0],pts[:,1],pts[:,2]); scat_h.set_sizes(cnt*4)
        else:
            scat_h._offsets3d=([],[],[]); scat_h.set_sizes([])
        fire,iou=net.diag(); txt.set_text(f'step {t} | fire {fire:.3f} | IoU₁₀₀ {iou:.2f}')
        X_prev=X_now
        return im_in,im_pr,scat_in,scat_out,scat_h,txt
    #anim.FuncAnimation(fig,update,interval=70,blit=False)
    #plt.tight_layout(); plt.show()

        # --- animation ----------------------------------------------------------
    ani = anim.FuncAnimation(
        fig, update, interval=70, blit=False, cache_frame_data=False
    )

    # keep a reference (important!)
    globals()['_cube_anim'] = ani

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    main()
