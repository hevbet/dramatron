"""kn5 parser that also keeps node transforms (world matrices)."""
import struct, numpy as np
import sys; sys.path.insert(0,'/home/user/dramatron/livery/tools')
from kn5 import R
def load(path):
    r=R(open(path,'rb').read()); assert r.raw(6)==b'sc6969'; ver=r.i()
    if ver>5: r.i()
    for _ in range(r.i()): r.i(); r.s(); n=r.i(); r.raw(n)
    mats=[]
    for _ in range(r.i()):
        name=r.s(); r.s(); r.u8(); r.u8()
        if ver>4: r.i()
        for _ in range(r.i()): r.s(); r.f(); r.raw(36)
        sm={}
        for _ in range(r.i()):
            sn=r.s(); r.i(); sm[sn]=r.s()
        mats.append(dict(name=name,samplers=sm))
    meshes=[]; nodes=[]
    def node(path_, M):
        cls=r.i(); name=r.s(); nch=r.i(); r.u8(); full=path_+'/'+name
        W=M
        if cls==1:
            L=np.frombuffer(r.raw(64),dtype='<f4').reshape(4,4).astype(float)
            W=L@M  # row-vector convention
            nodes.append((full,W))
        elif cls==2:
            r.u8();r.u8();r.u8(); nv=r.i(); v=np.frombuffer(r.raw(nv*44),dtype='<f4').reshape(nv,11)
            ni=r.i(); idx=np.frombuffer(r.raw(ni*2),dtype='<u2'); mid=r.i(); r.i(); r.f(2); r.f(3); r.f(); r.u8()
            meshes.append(dict(name=full,pos=v[:,0:3].astype(float),nrm=v[:,3:6].astype(float),uv=v[:,6:8].astype(float),idx=idx.reshape(-1,3).astype(int),mat=mats[mid]['name'],W=M))
        elif cls==3:
            r.u8();r.u8();r.u8()
            for _ in range(r.i()): r.s(); r.raw(64)
            nv=r.i(); v=np.frombuffer(r.raw(nv*76),dtype='<f4').reshape(nv,19)
            ni=r.i(); idx=np.frombuffer(r.raw(ni*2),dtype='<u2'); mid=r.i(); r.i(); r.raw(8)
            meshes.append(dict(name=full,pos=v[:,0:3].astype(float),nrm=v[:,3:6].astype(float),uv=v[:,6:8].astype(float),idx=idx.reshape(-1,3).astype(int),mat=mats[mid]['name'],W=M))
        for _ in range(nch): node(full,W)
    node('',np.eye(4))
    for m in meshes:
        p=np.c_[m['pos'],np.ones(len(m['pos']))]@m['W']; m['wpos']=p[:,:3]
        m['wnrm']=m['nrm']@m['W'][:3,:3]
    return mats,meshes,nodes
