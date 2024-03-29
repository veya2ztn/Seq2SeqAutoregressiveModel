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
    def get_inputs(self,modelfun,x,mean_path_length,path_length_mode='010'):
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
        return inputs
    def getPathLengthlossOld(self,modelfun,x,mean_path_length,path_length_mode='010',coef=None,decay=0.01):     
        x = self.get_inputs(modelfun,x,mean_path_length,path_length_mode=path_length_mode)
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
        x = self.get_inputs(modelfun,x,mean_path_length,path_length_mode=path_length_mode)
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


class NGmod_RotationDelta(Nodal_GradientModifier):
    def normed(self,a):
        shape = a.shape
        a   = a.reshape(a.size(0),-1)
        a   = a/(a.norm(dim=1,keepdim=True)+1e-7) 
        a   = a.reshape(shape)
        return a  
    def get_delta(self, modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
         
        if rotation_regular_mode =='0y0':
            delta = (self.normed(y - x),)
        elif rotation_regular_mode =='0v0':
            delta = (self.normed(y_no_grad - x),)
        elif rotation_regular_mode =='Yy0':
            delta = [self.normed(t - x),self.normed(y - x)]
        elif rotation_regular_mode =='Yv0':
            delta = [self.normed(t - x),self.normed(y_no_grad - x)]
        elif rotation_regular_mode =='YyN':
            delta = [self.normed(t - x),self.normed(y - x),self.normed(torch.rand_like(x))]
        else:
            raise NotImplementedError
        return torch.stack(delta)
    def getL2loss(self,modelfun,x,chunk=10,coef=None):
        raise
    def getL1loss(self,modelfun,x,chunk=10,coef=None):
        raise
    
class NGmod_RotationDeltaY(NGmod_RotationDelta): # fast then X mode
    def Estimate_vJ_once(self,vjpfunc,cotangents): 
        grad = vjpfunc(cotangents)[0] 
        penalty    = ((torch.sum(grad**2,dim=(1,2,3))-1)**2).mean()
        return penalty

    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0'):
        y, vjpfunc = functorch.vjp(modelfun, x)
        delta = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        #penalty= 0
        #for delta_cons in delta:
        #    grad = vjpfunc(delta_cons)[0] 
        #    position_range = list(range(1,len(grad.shape))) # (B,P, W,H ) --> (1,2,3)
        #    penalty       += ((torch.sum(grad**2,dim=position_range)-1)**2).mean()/len(delta)
        penalty = vmap(self.Estimate_vJ_once, (None,0),randomness='same')(vjpfunc,delta).mean()
        return penalty


class NGmod_RotationDeltaX(NGmod_RotationDelta):
    def Estimate_Jv_once(self, model,x,cotangents):
        grad = functorch.jvp(model, (x,), (cotangents,))[1] 
        penalty    = ((torch.sum(grad**2,dim=(1,2,3))-1)**2).mean()
        return penalty
    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0'):
        y=modelfun(x) if 'y' in rotation_regular_mode else None
        delta = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        penalty = vmap(self.Estimate_Jv_once, (None,None, 0),randomness='same')(modelfun,x,delta).mean()
        return penalty
    
class NGmod_RotationDeltaXE(NGmod_RotationDelta):
    def Estimate_Jv_once(self, model,x,cotangents):
        grad = functorch.jvp(model, (x,), (cotangents,))[1] 
        penalty    = ((torch.sum(grad**2,dim=(1,2,3))-1)**2).mean()
        return penalty
    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0'):
        y = None
        delta = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        penalty = vmap(self.Estimate_Jv_once, (None,None, 0),randomness='same')(modelfun,x,delta).mean()
        return penalty
    



class NGmod_RotationDeltaE(NGmod_RotationDelta):
    def get_delta(self, modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
        
        if "y" in rotation_regular_mode:
            delta = self.normed(t - y)
        elif "v" in rotation_regular_mode:
            delta = self.normed(t - y_no_grad) # actually the ||J|| is around 0.97~1.03, use normed value to check this
        else:
            raise NotImplementedError
        return delta
    
    def get_activate_x(self,modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
        if 'J' in rotation_regular_mode:
            activate_x = y
        elif 'm' in rotation_regular_mode:
            activate_x = (y_no_grad + t)/2
        elif 'M' in rotation_regular_mode:
            activate_x = (y + t)/2
        elif 'j' in rotation_regular_mode:
            activate_x = y_no_grad
        else:
            activate_x = t
        return activate_x
    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0'):
        y        = modelfun(x) if (('y' in rotation_regular_mode) or 
                                   ('J' in rotation_regular_mode) or 
                                   ('M' in rotation_regular_mode)) else None
        delta      = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        activate_x  = self.get_activate_x(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        Mdelta      = functorch.jvp(modelfun, (activate_x,), (delta,))[1] 
        penalty    = (torch.sum(Mdelta**2,dim=(1,2,3))).sqrt().mean()
        return penalty

class NGmod_RotationDeltaESet(NGmod_RotationDeltaE):
    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0', set_value=0.65):
        y        = modelfun(x) if (('y' in rotation_regular_mode) or 
                                   ('J' in rotation_regular_mode) or 
                                   ('M' in rotation_regular_mode)) else None
        delta      = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        activate_x  = self.get_activate_x(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        Mdelta      = functorch.jvp(modelfun, (activate_x,), (delta,))[1] 
        penalty    = (torch.sum(Mdelta**2,dim=(1,2,3))).sqrt().mean()
        penalty = (penalty - set_value)**2
        return penalty
class NGmod_RotationDeltaET(NGmod_RotationDeltaE):
    
    def get_delta(self, modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
        
        if "y" in rotation_regular_mode:
            delta = self.normed(t - y)
        elif "v" in rotation_regular_mode:
            delta = self.normed(t - y_no_grad) # actually the ||J|| is around 0.97~1.03, use normed value to check this
        else:
            raise NotImplementedError
        return delta
    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0y0'):
        y        = modelfun(x) if (('y' in rotation_regular_mode) or 
                                   ('J' in rotation_regular_mode) or 
                                   ('M' in rotation_regular_mode)) else None
        delta      = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        activate_x  = self.get_activate_x(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        
        Mdelta      = functorch.jvp(modelfun, (activate_x,), (delta,))[1] 
        penalty    = torch.mean(Mdelta**2)
        return penalty



class NGmod_RotationDeltaETwo(NGmod_RotationDelta):
    def get_delta(self, modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
        
        if "y" in rotation_regular_mode:
            delta = t - y
        elif "v" in rotation_regular_mode:
            delta = t - y_no_grad
        else:
            raise NotImplementedError
        return delta
    

    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = 'Myj'):
        y        = modelfun(x) if (('y' in rotation_regular_mode) or 
                           ('J' in rotation_regular_mode) or 
                           ('M' in rotation_regular_mode)) else None
        delta      = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        if 'm' in rotation_regular_mode:
            activate_x = (y_no_grad + t)/2
        elif 'M' in rotation_regular_mode:
            activate_x = (y + t)/2
        else:
            raise NotImplementedError
        if 'j' in rotation_regular_mode: 
            # principly, should use no graded y cause is you has already calculate the f[f(x)] then 
            # why not directly calculate the error between f[f(x_t)] and x_{t+2}
            # In short, the goal of this term is try to avoid calculate the backward of f[f(x_t)].
            activate_y = y_no_grad
        elif 'J' in rotation_regular_mode:
            activate_y = y 
        else:
            raise NotImplementedError

        Mdelta    = functorch.jvp(modelfun, (activate_x,), (delta,))[1] 
        Target_delta = modelfun(t) - modelfun(activate_y)
        penalty   = (Target_delta - Mdelta)
        penalty    = (torch.mean(penalty**2))
        return penalty


class NGmod_RotationDeltaNmin(NGmod_RotationDelta):
    def get_delta(self, modelfun, x, y, y_no_grad, t, rotation_regular_mode = '0y0'):
        
        if "y" in rotation_regular_mode:
            delta = t - y
        elif "v" in rotation_regular_mode:
            delta = t - y_no_grad
        else:
            raise NotImplementedError
        return delta
    

    def getRotationDeltaloss(self, modelfun, x, y_no_grad, t , rotation_regular_mode = '0yJ'): 
        y        = modelfun(x) if (('y' in rotation_regular_mode) or 
                           ('J' in rotation_regular_mode) or 
                           ('M' in rotation_regular_mode)) else None
        delta      = self.get_delta(modelfun, x, y, y_no_grad, t , rotation_regular_mode = rotation_regular_mode)
        if 'j' in rotation_regular_mode: 
            # principly, should use no graded y cause is you has already calculate the f[f(x)] then 
            # why not directly calculate the error between f[f(x_t)] and x_{t+2}
            # In short, the goal of this term is try to avoid calculate the backward of f[f(x_t)].
            activate_y = y_no_grad
        elif 'J' in rotation_regular_mode:
            activate_y = y 
        else:
            raise NotImplementedError

        Target_delta = modelfun(t) - modelfun(activate_y)
        amplifier  = (torch.log(torch.mean(Target_delta**2,dim=(1,2,3))) - 
                torch.log(torch.mean(delta**2,dim=(1,2,3)))
                )
        penalty   = amplifier.mean() #notice this value samller than 0, thus the loss_wall should set -100
        return penalty



class NGmod_RotationDeltaEThreeTwo(NGmod_RotationDeltaETwo):
    
    def getRotationDeltaloss(self, modelfun, x_t_and_x_tp1, x_tp1_and_x_tp2_pred, 
                          x_tp1_and_x_tp2, rotation_regular_mode = 'Myj'):
        
        Btimes2, P, W, H = x_t_and_x_tp1.shape
        Batch_size = Btimes2//2
        assert Batch_size*2 == Btimes2
        x_t,  x_tp1_pred, x_tp1 = x_t_and_x_tp1[:Batch_size], x_tp1_and_x_tp2_pred[:Batch_size] ,x_tp1_and_x_tp2[:Batch_size]
        x_tp1, x_tp2_pred, x_tp2 = x_t_and_x_tp1[Batch_size:], x_tp1_and_x_tp2_pred[Batch_size:] ,x_tp1_and_x_tp2[Batch_size:]
        x_tp1_pred_backward = x_tp2_pred_backward = None
        if ('y' in rotation_regular_mode) or ('J' in rotation_regular_mode) or ('M' in rotation_regular_mode):
            x_tp1_and_x_tp2_pred_backward = modelfun(x_t_and_x_tp1)
            x_tp1_pred_backward = x_tp1_and_x_tp2_pred_backward[:Batch_size]
            x_tp2_pred_backward = x_tp1_and_x_tp2_pred_backward[Batch_size:]
        
        delta      = self.get_delta(modelfun, x_t, x_tp1_pred_backward, x_tp1_pred, x_tp1 , rotation_regular_mode = rotation_regular_mode)
        # none normlized delta
        if 'm' in rotation_regular_mode:  activate_x = (x_tp1_pred      + x_tp1)/2
        elif 'M' in rotation_regular_mode: activate_x = (x_tp1_pred_backward + x_tp1)/2
        else:
            raise NotImplementedError
        Mdelta    = functorch.jvp(modelfun, (activate_x,), (delta,))[1] 

        
        if 'j' in rotation_regular_mode: 
            activate_y = x_tp2_pred
        elif 'J' in rotation_regular_mode:
            activate_y = x_tp2_pred_backward
        else:
            raise NotImplementedError

        Single_Delta = x_tp2 - activate_y
        penalty   = Single_Delta + Mdelta
        penalty    = (torch.mean(penalty**2))
        return penalty


class NGmod_RotationDeltaXS(NGmod_RotationDeltaX):
    def Estimate_Jv_once(self, model,x,cotangents):
        assert len(x.shape)==4
        grad = functorch.jvp(model, (x,), (cotangents,))[1] 
        penalty = ((torch.sum(grad**2,dim=(1,2,3)).sqrt()-1)**2).mean()
        return penalty




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
#from functorch._src.eager_transforms import _vjp_with_argnums,_construct_standard_basis_for,_slice_argnums,_safe_zero_index
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
    def __call__(self,func, argnums):
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
        

def the_Nodal_L2_meassure(func, argnums, *, has_aux=False, chunk_size=None,sum_axis=[-3,-2,-1]):
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