import torch
import torch.nn.functional as F
import functorch
from functorch import jacrev
from functorch import make_functional, vmap, make_functional_with_buffers
import torch
import numpy as np

class Nodal_GradientModifier:
    def __init__(self, lambda1=1, lambda2=1, sample_times=100, do_unit_renormalize=False,L1_level=1,L2_level=1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sample_times = sample_times 
        self.cotangents_sum_along_x_dimension = None
        self.do_unit_renormalize = do_unit_renormalize
        self.L1_level = L1_level
        self.L2_level = L2_level
    def Normlization_Term_1(self, params, x, return_abs_value=False):
        if self.cotangents_sum_along_x_dimension is None or self.cotangents_sum_along_x_dimension.shape != x.shape:
            self.cotangents_sum_along_x_dimension = torch.ones_like(x)
        tvalues= functorch.jvp(lambda x: self.func_model(params, x),
                  (x,), (self.cotangents_sum_along_x_dimension,))[1]
        values = ((tvalues-1)**2).mean()
        #(B, Outputdim) -> (1,)
        N = np.prod(x.shape[1:])
        if self.do_unit_renormalize:
            # the varation of L1 for iid normal distribution is around 2n^2 + 4n
            coef   = 2*np.power(N, 2) + 4*N
            values = values/np.sqrt(coef)
        if return_abs_value:
            return values, tvalues.abs().mean()
        return values    
    def TrvJOJv_and_ETrAAT(self,params,x,cotangents_variable):
        # compute projection on vector `cotangent` only once.
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        dims = list(range(1,len(vJ.shape)))
        vJO  = vJ.sum(dims,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJO*vJ).sum(dim=dims)# <vJ|1-I|Jv> #should sum over all dimension except batch
        #vJOJv= vJOJv/np.sqrt(torch.prod(self.output_shape).item())
        ETrAAT = functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1] # (B,Ouputdim)
        # <vJ|1-I|J^T | J|1-I|J^Tv> = < k| J^T | J |k> where the  <k| = <vJ|
        # the `ETrAAT` now is just the < k| J^T |
        dims = list(range(1,len(ETrAAT.shape)))
        ETrAAT=torch.sum(ETrAAT**2,dim=dims)
        # the `ETrAAT` now is just the < k| J^T | J |k> 
        return vJOJv, ETrAAT# DO NOT average the batch_size also
    def get_TrvJOJv(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        dims = list(range(1,len(vJ.shape)))
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
        #vJOJv= vJOJv/np.sqrt(torch.prod(self.output_shape).item()) # notice we will do var later, so should get sqrt
        return vJOJv
    def get_ETrAAT(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        ETrAAT = functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1] # (B,Ouputdim)
        dims = list(range(1,len(ETrAAT.shape)))# (B,Ouputdim)
        ETrAAT=torch.sum(ETrAAT**2,dim=dims)# (B,Ouputdim) 
        return ETrAAT
    def get_ETrAAT_times(self,params,x,cotangents_variables):
        return vmap(self.get_ETrAAT, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def get_TrvJOJv_times(self,params,x,cotangents_variables):
        return vmap(self.get_TrvJOJv, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def Estimate_L2_once(self,params,x,cotangents1,cotangents2,cotangents3):
        """$L2 =\sum_\gamma [\sum_\alpha (J_\alpha^{\gamma})^2-1]^2=||L_{k}^{\gamma}||^2- 2 ||J||_2^2$
        Notice the ture L2 need a shift vaule equal to outshape
        ----------------------------------------
        import torch
        from mltool.visualization import *
        N = 10
        J = torch.randn(N,N)
        L = torch.einsum("ba,bc->bac",J,J).flatten(1,2)

        esitmates=[]
        for B in [1e2,1e3,1e4,1e5,1e6]:
            for _ in range(10):
                #a = torch.randint(0,2,size=(int(B),N))*2-1.0# <-- support [-1,1] or any random number 
                #b = torch.randint(0,2,size=(int(B),N))*2-1.0# <-- support [-1,1] or any random number 
                a = torch.randn(int(B),N)
                b = torch.randn(int(B),N)
                c = torch.randn(int(B),N)
                v1= torch.einsum('ij,bj->bi',J,a)
                v2= torch.einsum('ij,bj->bi',J,b)
                v = v1*v2
                v3= torch.einsum('ij,bi->bj',J,c)
                esitmates.append(torch.mean(v.norm(dim=1)**2) - 2*torch.mean(v3.norm(dim=1)**2))
        real_val = L.norm()**2- 2*J.norm()**2
        xrange = list(range(len(esitmates)))
        plt.plot(xrange, [real_val]*len(esitmates),'g')
        plt.plot(xrange, esitmates,'r')
        """
        # in order to avoid large value, we will divide len(output_shape)
        # this equal to make the offset value in L2 become 1
        vL1 = functorch.jvp(lambda x:self.func_model(params,x), (x,), (cotangents1,))[1] #(B, output_size)
        vL2 = functorch.jvp(lambda x:self.func_model(params,x), (x,), (cotangents2,))[1] #(B, output_size)
        dims = list(range(1,len(vL1.shape)))
        coef = np.sqrt(np.prod(vL1.shape[1:]))
        vL   = ((vL1/coef*vL2)**2).sum(dim=dims)
        vJ  = functorch.jvp(lambda x:self.func_model(params,x), (x,), (cotangents3,))[1] #(B, output_size)
        vJ   = ((vJ/coef)**2).sum(dim=dims)
        esitimate = vL - 2*vJ + 1 #(B, 1)
        return esitimate
    def inference(self,model,x,y, strict=True,return_abs_value=True):
        back_to_train_mode = model.training
        model.eval()
        buffers=[]
        if not strict:buffers = list(model.buffers())
        if len(buffers) > 0:
            self.func_model,params, buffer = make_functional_with_buffers(model, disable_autograd_tracking=True)
            self.func_model = lambda params, x: func_model(params, buffer, x)
        else:
            self.func_model, params = make_functional(model,disable_autograd_tracking=True)
        #self.func_model, params = make_functional(model,disable_autograd_tracking=True)
        self.output_shape = y.shape
        #with torch.no_grad():  # may occur unknow error when using make ---> RuntimeError: Mask should be Bool Scalar TypeFloat
        L11=L12=-1
        if self.lambda1 != 0:
            if return_abs_value:
                L11,L12=self.Normlization_Term_1(params, x,return_abs_value)
                L11=L11.item()
                L12=L12.item()
            else:
                L11=self.Normlization_Term_1(params, x,return_abs_value).item()
        
        L2=self.Normlization_Term_2(params, x).item() if self.lambda2 != 0 else -1
        if back_to_train_mode:model.train()
        return L11,L12,L2
    def backward(self,model, x, y, strict=True):
        
        model.eval()
        buffers=[]
        if not strict:buffers = list(model.buffers())
        if len(buffers) > 0:
            self.func_model,params, buffer = make_functional_with_buffers(model, disable_autograd_tracking=True)
            self.func_model = lambda params, x: func_model(params, buffer, x)
        else:
            self.func_model, params = make_functional(model,disable_autograd_tracking=True)

        self.output_shape       = y.shape[1:]
        with torch.no_grad():
            if self.lambda1 != 0:
                Derivation_Term_1 = jacrev(self.Normlization_Term_1, argnums=0)(params, x)
            if self.lambda2 != 0:
                Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params, x)
        model.train()
        for i, param in enumerate(model.parameters()):
            delta_p = 0
            if self.lambda1 != 0:delta_p += self.lambda1*Derivation_Term_1[i]
            if self.lambda2 != 0:delta_p += self.lambda2*Derivation_Term_2[i]
            if param.grad is not None:
                param.grad.detach().add_(delta_p.to(param.device)/2)# this coding style is for multiGPU usage
                #param.grad.data += delta_p
            else:
                param.grad = delta_p.to(param.device)
    def getL1loss(self,modelfun,x,coef=None):
        pos=time=None
        model = modelfun
        if isinstance(x,list):
            x, pos,time = x
            model = lambda x:modelfun(x,pos,time)
        if coef is not None:
            coef = coef.to(x.device)
            model = lambda x:modelfun(x*coef)
            x = x/(coef+1e-6) # this is to make share the delta_x can be normal distribution
        if self.cotangents_sum_along_x_dimension is None or self.cotangents_sum_along_x_dimension.shape != x.shape:
            self.cotangents_sum_along_x_dimension = torch.ones_like(x)
        tvalues= functorch.jvp(model,(x,), (self.cotangents_sum_along_x_dimension,))[1]
        values = ((tvalues-1)**2).mean()
        return values

    
    def getL2loss(self,model,x):
        raise NotImplementedError

    def normed(self,a):
        shape = a.shape
        a = a.reshape(a.size(0),-1)
        a = a/a.norm(dim=1,keepdim=True)
        a = a.reshape(shape)
        return a 

    def getRotationDeltaloss(self, modelfun, x, t , rotation_regular_mode = '0y0'):
        y, vjpfunc = functorch.vjp(modelfun, x) # notice this will calculate f(x) again, so can be reduced in real implement.
        
        if rotation_regular_mode =='0y0':
            delta = self.normed(y - x)
        elif rotation_regular_mode =='Yy0':
            delta = torch.cat([self.normed(t - x),self.normed(y - x)])
        elif rotation_regular_mode =='YyN':
            delta = torch.cat([self.normed(t - x),self.normed(y - x),self.normed(torch.rand_like(x))])
        else:
            raise NotImplementedError
        grad = vjpfunc(delta)[0]
        position_range = list(range(1,len(grad.shape))) # (B,P, W,H ) --> (1,2,3)
        penalty        = ((torch.sqrt(torch.sum(grad**2,dim=position_range))-1)**2).mean()
        return penalty
        


class NGmod_absolute(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x):
        dims   = [-t for t in range(1,len(x.shape))]
        dims.reverse()                         
        values = (((vmap(jacrev(self.func_model, argnums=1), (None, 0),randomness='same')(params, x)**2).sum(dim=dims)-1)**2).mean()
        N      = np.prod(x.shape[1:])
        if self.do_unit_renormalize: 
            coef = 8*np.power(N,3) + 24*np.power(N,2) + 24*N
            values = values/np.sqrt(coef)
            # the varation of L2 for iid normal distribution is around 8n^3 + 24n^2 + 24n
        return values
        
    def getL2loss(self,model,x):
        pos=time=None
        if isinstance(x,list):
            x, pos,time = x
        dims   = [-t for t in range(1,len(x.shape))]
        dims.reverse()     
        if pos is None:        
            values = (((vmap(jacrev(model), (0,),randomness='same')(x,)**2).sum(dim=dims)-1)**2).mean()
        else:
            values = (((vmap(jacrev(model), (0,0,0),randomness='same')(x,pos,time)**2).sum(dim=dims)-1)**2).mean()
        return values
class NGmod_absolute_set_level(NGmod_absolute):
    def new_Normlization_Term_1(self, params, x):
        return (self.Normlization_Term_1(params, x) - self.L1_level)**2
    def backward(self,model, x, y, strict=True):
        
        model.eval()
        buffers=[]
        if not strict:buffers = list(model.buffers())
        if len(buffers) > 0:
            func_model,params, buffer = make_functional_with_buffers(model, disable_autograd_tracking=True)
            self.func_model = lambda params, x: func_model(params, buffer, x)
        else:
            self.func_model, params = make_functional(model,disable_autograd_tracking=True)
        
        self.output_shape       = y.shape[1:]

        with torch.no_grad():
            if self.lambda1 != 0:
                Derivation_Term_1 = jacrev(self.new_Normlization_Term_1, argnums=0)(params, x)
            if self.lambda2 != 0:
                Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params, x)
        model.train()
        for i, param in enumerate(model.parameters()):
            delta_p = 0
            if self.lambda1 != 0:delta_p += self.lambda1*Derivation_Term_1[i]
            if self.lambda2 != 0:delta_p += self.lambda2*Derivation_Term_2[i]
            if param.grad is not None:
                param.grad.data += delta_p
            else:
                param.grad = delta_p

class NGmod_pathlength(Nodal_GradientModifier):
    def getPathLengthlossOld(self,modelfun,x,mean_path_length,path_length_mode='010',coef=None,decay=0.01):
        if path_length_mode == '221':
            inputs1 = torch.randn_like(x).repeat(2,1,1,1)
            inputs2 = torch.randn_like(x).repeat(2,1,1,1)*0.001 + x.repeat(2,1,1,1)
            inputs = torch.cat([inputs1,inputs2,x])
        elif path_length_mode == '020':
            inputs = torch.randn_like(x).repeat(2,1,1,1)*0.001 + x.repeat(2,1,1,1)
        elif path_length_mode == '010':
            inputs = torch.randn_like(x)*0.001 + x
        elif path_length_mode == '011':
            inputs2 = torch.randn_like(x)*0.001 + x
            inputs  = torch.cat([inputs2,x])
        elif path_length_mode == '001':
            inputs = x
        else:
            raise NotImplementedError
        x = inputs
        pos=time=None
        model = modelfun
        if isinstance(x,list):
            x, pos,time = x
            model = lambda x:modelfun(x,pos,time)
        if coef is not None:
            coef = coef.to(x.device)
            model = lambda x:modelfun(x*coef,pos,time)
            x = x/coef # this is to make share the delta_x can be normal distribution
        cotangents_sum_along_x_dimension = torch.randn_like(x) / np.sqrt(np.prod(x.shape[1:]))
        (_, vjpfunc) = functorch.vjp(model, x)
        grad = vjpfunc(cotangents_sum_along_x_dimension)[0]
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1)) #(B,H) 
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return path_penalty, path_mean.detach(), path_lengths
    
    def getPathLengthloss(self,modelfun,x,mean_path_length,path_length_mode='010',coef=None,decay=0.01):
        if path_length_mode == '221':
            inputs1 = torch.randn_like(x).repeat(2,1,1,1)
            inputs2 = torch.randn_like(x).repeat(2,1,1,1)*0.001 + x.repeat(2,1,1,1)
            inputs = torch.cat([inputs1,inputs2,x])
        elif path_length_mode == '020':
            inputs = torch.randn_like(x).repeat(2,1,1,1)*0.001 + x.repeat(2,1,1,1)
        elif path_length_mode == '010':
            inputs = torch.randn_like(x)*0.001 + x
        elif path_length_mode == '011':
            inputs2 = torch.randn_like(x)*0.001 + x
            inputs  = torch.cat([inputs2,x])
        elif path_length_mode == '001':
            inputs = x
        else:
            raise NotImplementedError
        x = inputs
        pos=time=None
        model = modelfun
        if isinstance(x,list):
            x, pos,time = x
            model = lambda x:modelfun(x,pos,time)
        if coef is not None:
            coef = coef.to(x.device)
            model = lambda x:modelfun(x*coef,pos,time)
            x = x/coef # this is to make share the delta_x can be normal distribution
        cotangents_sum_along_x_dimension = torch.randn_like(x) / np.sqrt(np.prod(x.shape[1:]))
        (_, vjpfunc) = functorch.vjp(model, x)
        grad = vjpfunc(cotangents_sum_along_x_dimension)[0]
        position_range = list(range(1,len(grad.shape))) # (B,P, W,H ) --> (1,2,3)
        path_lengths = torch.sqrt(torch.sum(grad**2,dim=position_range)) #(B,) # use mean to avoid explore
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = ((path_lengths - path_mean)**2).mean() # (B,) every Batch satisfy requirement
        return path_penalty, path_mean.detach(), path_lengths

    def getL2loss(self,modelfun,x,chunk=10,coef=None):
        raise
    def getL1loss(self,modelfun,x,chunk=10,coef=None):
        raise

    

class NGmod_RotationDeltaY(Nodal_GradientModifier):
    def normed(self,a):
        shape = a.shape
        a = a.reshape(a.size(0),-1)
        a = a/a.norm(dim=1,keepdim=True)
        a = a.reshape(shape)
        return a 

    def getRotationDeltaloss(self, modelfun, x,  t , rotation_regular_mode = '0y0'):
        y, vjpfunc = functorch.vjp(modelfun, x) # notice this will calculate f(x) again, so can be reduced in real implement.
        if rotation_regular_mode =='0y0':
            delta = self.normed(y - x)
        elif rotation_regular_mode =='0v0':
            delta = self.normed(y.detach() - x)
        elif rotation_regular_mode =='Yy0':
            delta = torch.cat([self.normed(t - x),self.normed(y - x)])
        elif rotation_regular_mode =='Yv0':
            delta = torch.cat([self.normed(t - x),self.normed(y.detach() - x)])
        elif rotation_regular_mode =='YyN':
            delta = torch.cat([self.normed(t - x),self.normed(y - x),self.normed(torch.rand_like(x))])
        else:
            raise NotImplementedError
        grad = vjpfunc(delta)[0] # another way is using functorch.jvp(model, (x,), (delta,))[1] # the result for two method is different
        position_range = list(range(1,len(grad.shape))) # (B,P, W,H ) --> (1,2,3)
        penalty        = ((torch.sum(grad**2,dim=position_range)-1)**2).mean()
        return penalty

    def getL2loss(self,modelfun,x,chunk=10,coef=None):
        raise
    def getL1loss(self,modelfun,x,chunk=10,coef=None):
        raise


class NGmod_RotationDeltaX(Nodal_GradientModifier):
    def normed(self,a):
        shape = a.shape
        a = a.reshape(a.size(0),-1)
        a = a/a.norm(dim=1,keepdim=True)
        a = a.reshape(shape)
        return a 

    def getRotationDeltaloss(self, modelfun, x, t , rotation_regular_mode = '0y0'):
        y = modelfun(x)
        if rotation_regular_mode =='0y0':
            delta = (self.normed(y - x),)
        elif rotation_regular_mode =='0v0':
            delta = (self.normed(y.detach() - x),)
        elif rotation_regular_mode =='Yy0':
            delta = (self.normed(t - x),self.normed(y - x))
        elif rotation_regular_mode =='Yv0':
            delta = (self.normed(t - x),self.normed(y.detach() - x))
        elif rotation_regular_mode =='YyN':
            delta = (self.normed(t - x),self.normed(y - x),self.normed(torch.rand_like(x)))
        else:
            raise NotImplementedError
        penalty= 0
        for delta_cons in delta:
            grad = functorch.jvp(modelfun, (x,), (delta_cons,))[1] 
            position_range = list(range(1,len(grad.shape))) # (B,P, W,H ) --> (1,2,3)
            penalty       += ((torch.sum(grad**2,dim=position_range)-1)**2).mean()
        return penalty

    def getL2loss(self,modelfun,x,chunk=10,coef=None):
        raise
    def getL1loss(self,modelfun,x,chunk=10,coef=None):
        raise



class NGmod_estimate_L2(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x):
        cotangents1s = torch.randint(0,2, (self.sample_times,*x.shape)).cuda()*2-1.0
        cotangents2s = torch.randint(0,2, (self.sample_times,*x.shape)).cuda()*2-1.0
        cotangents3s = torch.randint(0,2, (self.sample_times,*x.shape)).cuda()*2-1.0
        values = vmap(self.Estimate_L2_once, (None,None, 0,0,0))(params,x,cotangents1s,cotangents2s,cotangents3s).mean(0)
        values = values.mean()
        return values

    def Estimate_L2_once_model(self,model,x,cotangents1,cotangents2,cotangents3):
        # in order to avoid large value, we will divide len(output_shape)
        # this equal to make the offset value in L2 become 1
        
        vL1 = functorch.jvp(model, (x,), (cotangents1,))[1] #(B, output_size)
        vL2 = functorch.jvp(model, (x,), (cotangents2,))[1] #(B, output_size)
        dims = list(range(1,len(vL1.shape)))
        coef = np.sqrt(np.prod(vL1.shape[1:]))
        vL   = ((vL1/coef*vL2)**2).sum(dim=dims)
        vJ  = functorch.jvp(model, (x,), (cotangents3,))[1] #(B, output_size)
        vJ   = ((vJ/coef)**2).sum(dim=dims)**2
        esitimate = vL - 2*vJ + 1 #(B, 1)
        return esitimate
    def getL2loss(self,modelfun,x,chunk=10,coef=None):
        pos=time=None
        model= modelfun
        if isinstance(x,list):
            x, pos,time = x
            model = lambda x:modelfun(x,pos,time)
            # if isinstance(x,list):
            # x, pos,time = x
            # B= x.size(0)
            # xshape = x.shape[1:]
            # pshape = pos.shape[1:]
            # tshape = time.shape[1:]
            # a = np.prod(xshape)
            # b = np.prod(pshape)
            # c = np.prod(tshape)
            # x = torch.cat([x.flatten(1,-1),pos.flatten(1,-1),time.flatten(1,-1)],1)
            # model = lambda x:modelfun(x[:,0:a].reshape(-1,*xshape),
            #               x[:,a:a+b].reshape(-1,*pshape),
            #               x[:,a+b:a+b+c].reshape(-1,*tshape))
        if coef is not None:
            coef = coef.to(x.device)
            model = lambda x:modelfun(x*coef)
            x = x/(coef+1e-6)
        
        cotangents1s = torch.randint(0,2, (self.sample_times,*x.shape)).to(x.device)*2-1.0
        cotangents2s = torch.randint(0,2, (self.sample_times,*x.shape)).to(x.device)*2-1.0
        cotangents3s = torch.randint(0,2, (self.sample_times,*x.shape)).to(x.device)*2-1.0
        values = vmap(self.Estimate_L2_once_model, (None,None, 0,0,0),randomness='same')(model,x,cotangents1s,cotangents2s,cotangents3s).mean(0)
        values = values.mean()
        return values.abs()



class NGmod_absoluteNone(NGmod_absolute):
    def backward(self,model, x, y, strict=True):
        pass


class NGmod_delta_mean(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x):
        '''
        \sum_\beta\sum_{\alpha\neq\beta} J_\alpha^{\gamma}J_\beta^{\gamma}
        '''
        J = vmap(jacrev(self.func_model, argnums=1), (None, 0))(params, x) #(B, O, I)
        B, O, I =J.shape
        K = torch.ones(I,I) - torch.eye(I,I)
        K = K.to(J.device)
        # L^\gamma = \sum_\beta\sum_{\alpha\neq\beta} J_\alpha^{\gamma}J_\beta^{\gamma}
        C = torch.einsum('bij,jk,bik->bi',J,K,J) #(B,O)  L^\gamma
        C = (C**2).sum(-1) #(B) \sum_\gamma (L^\gamma)^2
        return C.mean()





class NGmode_estimate_delta(Nodal_GradientModifier):
    def Normlization_Term_2(self, params, x):
        Batch_size = x.size(0)
        # work_index    = list(range(Batch_size))
        # good_estimate = torch.zeros(Batch_size).to(x.device)
        # while len(work_index)>0:
        #     cotangents_variables = torch.randint(2,(self.sample_times,len(work_index),*self.output_shape)).cuda()*2-1
        #     TrvJOJvs,ETrAATs = vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0 ))(params, x[work_index],cotangents_variables)
        #     CorrelationTerm  = ETrAATs.mean(0) - TrvJOJvs.var(0)/2 #(B,)
        #     good_index = np.where(CorrelationTerm.detach().cpu().numpy()>0)[0].tolist()
        #     real_index = [work_index[idx] for idx in good_index]
        #     good_estimate[real_index]=CorrelationTerm[good_index]
        #     work_index = list(set(work_index) - set(real_index))
        good_estimate = []
        while len(good_estimate) < Batch_size:
            i = len(good_estimate)
            cotangents_variable = torch.randint(
                2, (self.sample_times, *self.output_shape)).cuda()*2-1
            TrvJOJvs, ETrAATs = vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0))(
                params, x[i], cotangents_variable)
            CorrelationTerm = ETrAATs.mean(0) - TrvJOJvs.var(0)/2  # (B,)
            assert not torch.isnan(CorrelationTerm)
            if CorrelationTerm.item() > 0:
                good_estimate.append(
                    CorrelationTerm/np.sqrt(np.prod(list(self.output_shape))))
        good_estimate = torch.stack(good_estimate)
        return good_estimate.mean()


class NGmode_estimate2_delta(Nodal_GradientModifier):
    def Normlization_Term_2(self, params, x):
        Batch_size = x.size(0)
        cotangents_variables = torch.randint(
            2, (self.sample_times, Batch_size, *self.output_shape)).cuda()*2-1
        TrvJOJvs, ETrAATs = vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0))(
            params, x, cotangents_variables)
        CorrelationTerm = ETrAATs.mean(0) - TrvJOJvs.var(0)/2  # (B,)
        CorrelationTerm = CorrelationTerm / \
            np.sqrt(np.prod(list(self.output_shape)))
        return CorrelationTerm.mean()



from scipy.sparse import identity
import sparse
import torch
from functorch._src.eager_transforms import *
from functorch._src.eager_transforms import _vjp_with_argnums,_construct_standard_basis_for,_slice_argnums,_safe_zero_index
def create_sparse_identity(flat_output, flat_output_numels):
    assert isinstance(flat_output_numels,int)
    assert isinstance(flat_output,(tuple,list))
    coo= sparse.COO.from_scipy_sparse(identity(flat_output_numels))
    coo= coo.reshape([flat_output_numels]+list(flat_output))
#     values  = coo.data
#     indices = coo.coords
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = coo.shape
#     tensor= torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return coo
def _construct_sparse_basis_for(flat_output, flat_output_numels):
    return [create_sparse_identity(t.shape,s) for t,s in zip(flat_output,flat_output_numels)]
def convert_to_torch_sparse(coo,device='cpu'):
    values  = coo.data
    indices = coo.coords
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
def convert_to_torch(coo): 
    return torch.FloatTensor(coo.todense())

class Nodal2_measure:
    def __init__(self,device,chunk_size=None):
        self.device     = device
        self.chunk_size = chunk_size
        self.basis_this_chunk_sparse = None
    def __call__(self,func: Callable, argnums: Union[int, Tuple[int]] = 0):
        @wraps(func)
        def wrapper_fn(*args):
            chunk_size=self.chunk_size
            has_aux = False
            vjp_out = _vjp_with_argnums(func,*args, argnums=1, has_aux=has_aux)
            output, vjp_fn = vjp_out
            flat_output, output_spec = tree_flatten(output)
            assert len(flat_output) == 1
            flat_output_numels = tuple(out.numel() for out in flat_output)
            
            sum_axis = list(range(1,len(output.shape)))
            sum_axis.reverse()
            sum_axis = [-t for t in sum_axis]
            
            #device= flat_output[0].device
            now = time.time()
            if self.basis_this_chunk_sparse is None:
                basis = _construct_sparse_basis_for(flat_output, flat_output_numels)[0]
                if chunk_size is None:
                    self.basis_this_chunk_sparse = convert_to_torch_sparse(basis,device=self.device)
                else:
                    start  = 0
                    self.basis_this_chunk_sparse={}
                    while start < len(basis):
                        end = min(start+chunk_size, len(basis))
                        self.basis_this_chunk_sparse[start,end] = convert_to_torch_sparse(basis[start:end],device=device)
                        start = end 
            print(f"create basis cost {time.time()-now}");now = time.time()
            if chunk_size is None:
                basis_this_chunk = self.basis_this_chunk_sparse.to_dense()
                print(f"sparse to dense cost {time.time()-now}");now = time.time()
                results = torch.sum((vmap(vjp_fn)(basis_this_chunk))**2,sum_axis)
                print(f"computing cost {time.time()-now}");now = time.time()
            else:
                results= None
                start  = 0
                while start < len(basis):
                    end = min(start+chunk_size, len(basis))
                    basis_this_chunk = self.basis_this_chunk_sparse[start,end].to_dense()
                    if results is None:
                        results = torch.sum(vmap(vjp_fn)(basis_this_chunk)**2,sum_axis)
                    else:
                        results = torch.cat([results,torch.sum(vmap(vjp_fn)(basis_this_chunk)**2,sum_axis)])
                    start = end
            return results
        return wrapper_fn
        

def the_Nodal_L2_meassure(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False, chunk_size=None,sum_axis=[-3,-2,-1]):
    @wraps(func)
    def wrapper_fn(*args):
        has_aux = False
        vjp_out = _vjp_with_argnums(func,*args, argnums=argnums, has_aux=has_aux)
        output, vjp_fn = vjp_out
        flat_output, output_spec = tree_flatten(output)
        assert len(flat_output) == 1
        flat_output_numels = tuple(out.numel() for out in flat_output)
        device= flat_output[0].device
        basis = _construct_sparse_basis_for(flat_output, flat_output_numels)[0]
        if chunk_size is None:
            basis_this_chunk = convert_to_torch_sparse(basis,device=device).to_dense()
            results = torch.sum((vmap(vjp_fn)(basis_this_chunk))**2,sum_axis)
        else:
            results= None
            start  = 0
            while start < len(basis):
                end = min(start+chunk_size, len(basis))
                #print(f"{start:4d} to {end:4d}")
                basis_this_chunk = convert_to_torch_sparse(basis[start:end],device=device).to_dense()
                #print("0")
                if results is None:
                    results = torch.sum(vmap(vjp_fn)(basis_this_chunk)**2,sum_axis)
                    #print("1")
                else:
                    results = torch.cat([results,torch.sum(vmap(vjp_fn)(basis_this_chunk)**2,sum_axis)])
                    #print("2")
                start = end
                #torch.cuda.empty_cache()
        #print(f"result_cost:{time.time()-now}");now = time.time()
        return results.reshape(flat_output[0].shape)
    return wrapper_fn

def the_Nodal_L1_meassure(func):
    @wraps(func)
    def wrapper_fn(x):
        cotangents_sum_along_x_dimension = torch.ones_like(x)
        tvalues= functorch.jvp(func,(x,), (cotangents_sum_along_x_dimension,))[1]
        return tvalues
    return wrapper_fn