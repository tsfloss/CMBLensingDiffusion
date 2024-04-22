import jax
import jax.numpy as jnp
import numpy as np

class PSFAST():
    def __init__(self,grid,BoxSize,fc,dk,Nbins):
        self.BoxSize = BoxSize
        self.grid = grid
        
        self.kF = 2*np.pi/BoxSize
        print('grid', grid)
        print('BoxSize',BoxSize)
        print('kF',self.kF)
        print('Nyquist',self.kF*grid/2)
        
        kx = jnp.fft.fftfreq(grid,1./grid)
        kz = jnp.fft.rfftfreq(grid,1./grid)
        kmesh = jnp.meshgrid(kx,kz,indexing='ij')
        self.kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2.)
        print('kmax in grid',self.kgrid.max() * self.kF)
        
        bin_center = jnp.array([fc+i*dk for i in range(Nbins)],dtype=jnp.float32)
        bin_lower = bin_center - dk/2
        bin_upper = bin_center + dk/2

        print('kmax',bin_upper[-1]*self.kF)

        self.bools = jnp.array([(self.kgrid >= bin_lower[i])*(self.kgrid < bin_upper[i]) for i in range(Nbins)],dtype=jnp.complex64)
        binned = jnp.fft.irfft2(self.bools)
        self.counts = jnp.sum(binned**2.,axis=[1,2]) * self.grid**2

        field_k = (self.kgrid).astype(jnp.complex64)
        field_k = jnp.sqrt(field_k)
        binned = jnp.einsum("jk,ljk->ljk",field_k,self.bools)
        binned = jnp.fft.irfft2(binned)
        self.k_means = np.asarray(self.kF * (jnp.sum(binned**2.,axis=[1,2]) * self.grid**2 / self.counts))
        self.k_centers = np.asarray(self.kF * bin_center)

    def lowpass(self,field,k_max_kF):
        field = jnp.array(field,dtype=jnp.float32)
        field = jnp.fft.rfft2(field)
        booly = jnp.array(self.kgrid <= k_max_kF,dtype=jnp.complex64)
        field = jnp.einsum("ijk,jk->ijk",field,booly)
        field = jnp.fft.irfft2(field)
        return np.asarray(field)
    
    def Pk(self,field):
        field = jnp.array(field,dtype=jnp.float32)
        field = jnp.fft.rfft2(field,axes=(-2,-1))
        field = jnp.einsum("ijk,ljk->iljk",field,self.bools)
        field = jnp.fft.irfft2(field,axes=(-2,-1))
        pp = jnp.sum(field**2.,axis=[-2,-1]) / self.counts * self.BoxSize**2 / self.grid**2

        return pp

    def Ck(self,field1,field2):
        field1 = jnp.array(field1,dtype=jnp.float32)
        field1 = jnp.fft.rfft2(field1)
        binned1 = jnp.einsum("ijk,ljk->iljk",field1,self.bools)
        binned1 = jnp.fft.irfft2(binned1)
        p1 = jnp.sum(binned1**2.,axis=[2,3]) / self.counts * self.BoxSize**2 / self.grid**2

        field2 = jnp.array(field2,dtype=jnp.float32)
        field2 = jnp.fft.rfft2(field2)
        binned2 = jnp.einsum("ijk,ljk->iljk",field2,self.bools)
        binned2 = jnp.fft.irfft2(binned2)
        p2 = jnp.sum(binned2**2.,axis=[2,3]) / self.counts * self.BoxSize**2 / self.grid**2

        pX = jnp.sum(binned1*binned2,axis=[2,3]) / self.counts * self.BoxSize**2 / self.grid**2
        return pX/jnp.sqrt(p1*p2)


    Pk = jax.jit(Pk,static_argnums=0)
    Ck = jax.jit(Ck,static_argnums=0)