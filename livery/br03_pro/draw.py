"""Движок отрисовки ливреи BR03 (прототип из этапа дизайна): холст по маске островов, линии переменной
ширины, штриховки, шашки, размещение логотипов, проверка зазоров до краёв деталей."""
import numpy as np, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
S=4096
FONT='/home/user/dramatron/livery/br03/brand/fonts/Exo2-Italic[wght].ttf'
BR='/home/user/dramatron/livery/br03/brand/'
def hexc(h): h=h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))
C=dict(base='#1B1C22',facet='#262830',deep='#0F1014',yellow='#F0D166',red='#E5342C',azure='#1694DC',white='#F4F5F7',grey='#9AA0AC',ink='#15161B',pattern='#2A2D36',cyan='#00B5EF',orange='#EE6A26')
C={k:hexc(v) for k,v in C.items()}
import os
HERE=os.path.dirname(os.path.abspath(__file__))
ZONES=os.path.join(HERE,'zones')
M=np.array(Image.open(os.path.join(ZONES,'_island_mask.png')).convert('L').resize((S,S),Image.NEAREST))>127
def zmask(name):
    return np.array(Image.open(os.path.join(ZONES,name+'.png')).convert('L').resize((S,S),Image.NEAREST))>127
class Canvas:
    def __init__(self):
        self.img=Image.new('RGB',(S,S),(0,0,0))
        arr=np.zeros((S,S,3),np.uint8); arr[M]=C['base']; self.img=Image.fromarray(arr)
        self.log=[]
    def paint_alpha(self,alpha,color,name=None,clip=None):
        # alpha: L image full size
        a=np.array(alpha).astype(np.float32)/255.
        if clip is not None: a=a*clip
        a=a*M
        base=np.array(self.img).astype(np.float32)
        col=np.array(color,np.float32)
        base=base*(1-a[...,None])+col*a[...,None]
        self.img=Image.fromarray(base.clip(0,255).astype(np.uint8))
        if name: self.log.append((name,alpha))
    def paint_rgba(self,rgba_full,name=None):
        a=np.array(rgba_full).astype(np.float32)
        al=a[...,3:4]/255.*M[...,None]
        base=np.array(self.img).astype(np.float32)
        base=base*(1-al)+a[...,:3]*al
        self.img=Image.fromarray(base.clip(0,255).astype(np.uint8))
        if name: self.log.append((name,rgba_full.split()[3]))
def blank(): return Image.new('L',(S,S),0)
def place(small, center, rot, mode='L'):
    """small: L or RGBA image already at final size (w along reading dir). rot: PIL angle."""
    r=small.rotate(rot,expand=True,resample=Image.BICUBIC)
    full=Image.new(mode,(S,S),0 if mode=='L' else (0,0,0,0))
    x=int(round(center[0]-r.width/2)); y=int(round(center[1]-r.height/2))
    full.paste(r,(x,y))
    return full
def text_mask(txt,weight,size,tracking=0.0):
    f=ImageFont.truetype(FONT,size); 
    try: f.set_variation_by_axes([weight])
    except Exception as e: print(e)
    # render char by char for tracking
    w=0; xs=[]
    for ch in txt:
        xs.append(w); w+=f.getlength(ch)+tracking*size
    w-=tracking*size
    im=Image.new('L',(int(w+size*1.2),int(size*1.6)),0); d=ImageDraw.Draw(im)
    for ch,x in zip(txt,xs): d.text((x+size*0.1,size*0.1),ch,font=f,fill=255)
    bb=im.getbbox(); return im.crop(bb)
def fit(mask,w=None,h=None):
    W,H=mask.size
    if w and h: return mask.resize((int(w),int(h)),Image.LANCZOS)
    if w: return mask.resize((int(w),int(round(H*w/W))),Image.LANCZOS)
    return mask.resize((int(round(W*h/H)),int(h)),Image.LANCZOS)
def brand(name): 
    im=Image.open(BR+name+'.png').convert('L'); bb=im.getbbox(); return im.crop(bb)
ASSETS={}
def asset(name):
    return ASSETS[name].copy()
def vline(pts, color_alpha_cb=None):
    """pts: list of (x,y,w). returns L mask full canvas of variable width polyline (polygon of offsets)."""
    P=np.array([(p[0],p[1]) for p in pts],float); W=np.array([p[2] for p in pts],float)
    n=len(P); left=[];right=[]
    for i in range(n):
        if i==0: d=P[1]-P[0]
        elif i==n-1: d=P[-1]-P[-2]
        else:
            d1=P[i]-P[i-1]; d2=P[i+1]-P[i]
            d=d1/np.linalg.norm(d1)+d2/np.linalg.norm(d2)
        d=d/np.linalg.norm(d); nrm=np.array([-d[1],d[0]])
        # miter scale
        sc=1.0
        if 0<i<n-1:
            d1=(P[i]-P[i-1]); d1/=np.linalg.norm(d1); n1=np.array([-d1[1],d1[0]])
            sc=1/max(0.35,abs(np.dot(nrm,n1)))
        left.append(P[i]+nrm*W[i]/2*sc); right.append(P[i]-nrm*W[i]/2*sc)
    poly=[tuple(p) for p in left]+[tuple(p) for p in right[::-1]]
    im=Image.new('L',(S*2,S*2),0); d=ImageDraw.Draw(im)
    d.polygon([(x*2,y*2) for x,y in poly],fill=255)
    return im.resize((S,S),Image.LANCZOS)
def offset_pts(pts,off):
    P=np.array([(p[0],p[1]) for p in pts],float); out=[]
    for i in range(len(P)):
        if i==0: d=P[1]-P[0]
        elif i==len(P)-1: d=P[-1]-P[-2]
        else:
            d1=P[i]-P[i-1]; d2=P[i+1]-P[i]; d=d1/np.linalg.norm(d1)+d2/np.linalg.norm(d2)
        d=d/np.linalg.norm(d); nrm=np.array([-d[1],d[0]])
        out.append((P[i][0]+nrm[0]*off,P[i][1]+nrm[1]*off,pts[i][2]))
    return out
def poly_mask(poly):
    im=Image.new('L',(S,S),0); ImageDraw.Draw(im).polygon(poly,fill=255); return im
def hatch(poly, period, bar, lean=(1,-1), fade=None):
    """bars along direction where (x*lean0 + y*lean1) const... stripes: s=(x*a+y*b)/sqrt2 mod period < barw.
    fade: (axis 'x'|'y', v0, v1, w0, w1) bar width linear in coord."""
    yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    a,b=lean; s=(xx*a+yy*b)/math.hypot(a,b)
    ph=np.mod(s,period)
    if fade:
        ax,v0,v1,w0,w1=fade; c=xx if ax=='x' else yy
        t=np.clip((c-v0)/(v1-v0),0,1); bw=w0+(w1-w0)*t
    else: bw=bar
    m=(ph<bw).astype(np.float32)
    pm=np.array(poly_mask(poly))/255.
    return Image.fromarray((m*pm*255).astype(np.uint8))
def checker_rect(x0,y0,x1,y1,sq,phase=0):
    yy,xx=np.mgrid[0:S,0:S]
    m=((((xx-x0)//sq)+((yy-y0)//sq)+phase)%2==0)&(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    return Image.fromarray((m*255).astype(np.uint8))
def checker_path(path, sq, ncols, phase=0):
    """checker tape following polyline path (centerline), ncols columns of sq."""
    out=np.zeros((S,S),np.uint8)
    yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    acc=0.0
    for (x0,y0),(x1,y1) in zip(path[:-1],path[1:]):
        L=math.hypot(x1-x0,y1-y0); d=np.array([(x1-x0)/L,(y1-y0)/L]); n=np.array([-d[1],d[0]])
        bx0=int(min(x0,x1)-sq*ncols); bx1=int(max(x0,x1)+sq*ncols); by0=int(min(y0,y1)-sq*ncols); by1=int(max(y0,y1)+sq*ncols)
        X=xx[by0:by1,bx0:bx1]-x0; Y=yy[by0:by1,bx0:bx1]-y0
        t=X*d[0]+Y*d[1]; s=X*n[0]+Y*n[1]+sq*ncols/2
        inside=(t>=0)&(t<L)&(s>=0)&(s<sq*ncols)
        cell=(np.floor((t+acc)/sq)+np.floor(s/sq)+phase)%2==0
        out[by0:by1,bx0:bx1][inside&cell]=255
        acc+=L
    return Image.fromarray(out)
def disc(cx,cy,r):
    im=Image.new('L',(S*2,S*2),0); ImageDraw.Draw(im).ellipse([(cx-r)*2,(cy-r)*2,(cx+r)*2,(cy+r)*2],fill=255)
    return im.resize((S,S),Image.LANCZOS)
def clearance(alpha):
    """min distance (px) from painted alpha>0.5 pixels to outside of island mask"""
    from scipy import ndimage
    global DT
    if 'DT' not in globals():
        DT=ndimage.distance_transform_edt(M)
    a=np.array(alpha)>127
    if not a.any(): return None,0
    return float(DT[a].min()), float((a&~M).sum()/a.sum())
