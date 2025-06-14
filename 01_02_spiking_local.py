#!/usr/bin/env python3
"""
Spiking demo - local receptive field, minimal noise
---------------------------------------------------
* Hidden = input shape (n×n), each unit connects to r=1 neighborhood (9 pixels)
* Almost no noise, output threshold low, so input dominates
* Input is cancelled between step 3001 and 4000, and resumes after 4000
"""

import argparse, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from collections import deque

# ---------- ENV --------------------------------------------------
class DotEnv:
    def __init__(self, n:int, speed:int=1):
        self.n, self.speed = n, speed; self.reset()
    def reset(self):
        edge = random.choice(['top','bottom','left','right'])
        if edge in ('top','bottom'):
            self.x = random.randrange(self.n); self.y = 0 if edge=='top' else self.n-1
            self.vx = random.choice([-1,1]);    self.vy =  self.speed if edge=='top' else -self.speed
        else:
            self.y = random.randrange(self.n); self.x = 0 if edge=='left' else self.n-1
            self.vy = random.choice([-1,1]);    self.vx =  self.speed if edge=='left' else -self.speed
    def frame(self):
        img = np.zeros((self.n,self.n), dtype=np.int8)
        img[self.y,self.x] = 1; return img
    def step(self):                    # move dot, return new frame
        self.x += self.vx; self.y += self.vy
        if self.x < 0: self.x, self.vx = -self.x,  +self.speed
        if self.x >= self.n: self.x, self.vx = 2*self.n-2-self.x, -self.speed
        if self.y < 0: self.y, self.vy = -self.y,  +self.speed
        if self.y >= self.n: self.y, self.vy = 2*self.n-2-self.y, -self.speed
        return self.frame()

# ---------- NETWORK ---------------------------------------------
class LocalSpikingNet:
    def __init__(self, n:int=48, r:int=1, rho=0.9,
                 eta=0.01, eta_out=0.02,
                 target_rate=0.1, tau_out=1,
                 bias=0.05, sigma_noise=0.002, seed=0):
        self.n, self.N = n, n*n
        self.r = r
        self.rho = rho
        self.eta, self.eta_out = eta, eta_out
        self.tau_out = tau_out
        self.target = target_rate
        self.bias = bias
        self.sigma = sigma_noise
        self.M = self.N                    # hidden = input size

        rng = np.random.default_rng(seed)
        self.W_in = np.zeros((self.M, self.N))
        idx = lambda y,x: y*self.n + x
        for y in range(n):
            for x in range(n):
                h = idx(y,x)
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < n and 0 <= xx < n:
                            self.W_in[h, idx(yy,xx)] = rng.uniform(0.5, 1.0)  # strong positive
        self.W_out = rng.normal(0, 0.1, size=(self.N, self.M))

        self.v = np.zeros(self.M)
        self.theta = np.full(self.M, 1.0)
        self.last = np.zeros(self.M)

        self.fire_hist, self.iou_hist = deque(maxlen=100), deque(maxlen=100)

    def forward(self, X):
        I = self.W_in @ X + self.bias + np.random.normal(0, self.sigma, self.M)
        self.v = self.rho*self.v + I
        s = (self.v >= self.theta).astype(float)
        self.v[s==1] = 0
        out = (self.W_out @ s > self.tau_out).astype(np.int8)
        return out, s

    def update(self, X, s, out, target):
        err = target - out
        self.W_in += self.eta * np.outer(s, X)
        np.clip(self.W_in, 0, 1.5, out=self.W_in)
        self.W_out += self.eta_out * err[:,None] * s[None,:]
        self.theta += 0.01 * (s - self.target)
        np.clip(self.theta, 0.3, 3.0, out=self.theta)

    def step(self, X, target):
        out, s = self.forward(X)
        inter = np.logical_and(out, target).sum()
        union = np.logical_or(out, target).sum()
        self.iou_hist.append(inter/union if union else 1.)
        self.fire_hist.append(s.mean())
        self.update(X, s, out, target)
        self.last = s
        return out.reshape(self.n,self.n), s

    def diag(self):
        return np.mean(self.fire_hist), np.mean(self.iou_hist)

# ---------- VIS --------------------------------------------------
def safe(x):
    return np.asarray(x, dtype=float).reshape(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    env = DotEnv(args.n)
    net = LocalSpikingNet(n=args.n, seed=args.seed)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9,6)); gs = fig.add_gridspec(2,2)
    ax_real = fig.add_subplot(gs[0,0]); ax_pred = fig.add_subplot(gs[0,1])
    ax_err  = fig.add_subplot(gs[1,0]); ax_3d = fig.add_subplot(gs[1,1], projection='3d')

    im_real = ax_real.imshow(env.frame(), cmap='Greys', vmin=0, vmax=1)
    im_pred = ax_pred.imshow(np.zeros((args.n,args.n)), cmap='Greys', vmin=0, vmax=1)
    im_err  = ax_err.imshow(np.zeros((args.n,args.n)), cmap='bwr', vmin=-1, vmax=1)
    for a,t in zip((ax_real,ax_pred,ax_err,ax_3d),
                   ('Real X_t','Prediction Ŷ_t','Error','3‑layer spikes')):
        a.set_title(t); a.set_xticks([]); a.set_yticks([])

    scat_in = ax_3d.scatter([],[],[],s=20,c='r',marker='s')
    scat_h  = ax_3d.scatter([],[],[],s=10,c='b',marker='o')
    scat_o  = ax_3d.scatter([],[],[],s=20,c='g',marker='s')
    ax_3d.set(xlim=(-0.5,args.n-0.5),ylim=(-0.5,args.n-0.5),zlim=(-0.5,2.5))

    txt = fig.text(0.02,0.95,'')

    X_prev = env.frame()
    step_counter = [0]  # Use mutable container for closure

    def update(t):
        nonlocal X_prev
        step_counter[0] += 1

        # Cancel input from step 3001 to 4000; normal otherwise
        if step_counter[0] <= 3000:
            X_now = env.step()
        elif step_counter[0] <= 4000:
            X_now = np.zeros_like(X_prev)
        else:
            X_now = env.step()

        out_img, s = net.step(X_prev.flatten(), X_now.flatten())

        im_real.set_data(X_now); im_pred.set_data(out_img)
        im_err.set_data(X_now - out_img)

        # scatter
        in_idx = np.argwhere(X_now==1)
        hid_idx = np.argwhere(s==1)
        out_idx = np.argwhere(out_img==1)
        scat_in._offsets3d = (safe(in_idx[:,1]), safe(in_idx[:,0]), safe(np.zeros(len(in_idx))))
        if hid_idx.size:
            xh,yh = hid_idx%args.n, hid_idx//args.n
            scat_h._offsets3d = (safe(xh), safe(yh), safe(np.ones(len(xh))))
        else:
            scat_h._offsets3d = (np.zeros(0),)*3
        scat_o._offsets3d = (safe(out_idx[:,1]), safe(out_idx[:,0]), safe(np.full(len(out_idx),2)))

        fire, iou = net.diag()
        txt.set_text(f'Step {step_counter[0]} | fire {fire:.3f} | IoU₁₀₀ {iou:.2f}')
        X_prev = X_now
        return im_real,im_pred,im_err,scat_in,scat_h,scat_o,txt

    ani = anim.FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
