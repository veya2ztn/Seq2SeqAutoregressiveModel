import torch
import torch.nn.functional as F
import functorch
from functorch import jacrev
from functorch import make_functional, vmap
import torch

class Nodal_GradientModifier:
    def __init__(self,lambda1=1,lambda2=1,sample_times=10):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sample_times = sample_times
        self.cotangents_sum_along_x_dimension = None
    def Normlization_Term_1(self,params,x):
        if self.cotangents_sum_along_x_dimension is None or self.cotangents_sum_along_x_dimension.shape!=x.shape:
            self.cotangents_sum_along_x_dimension = torch.ones_like(x)
        return ((functorch.jvp(lambda x:self.func_model(params,x), (x,), (self.cotangents_sum_along_x_dimension,))[1]-1)**2).mean()
    def TrvJOJv_and_ETrAAT(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        dims = list(range(1,len(vJ.shape)))
        vJO  = vJ.sum(dims,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
        ETrAAT = functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1] # (B,Ouputdim)
        dims = list(range(1,len(ETrAAT.shape)))
        ETrAAT=ETrAAT.norm(dim=dims)
        return vJOJv, ETrAAT# DO NOT average the batch_size also
    def get_TrvJOJv(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        dims = list(range(1,len(vJ.shape)))
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
        return vJOJv
    def get_ETrAAT(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        ETrAAT = functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1] # (B,Ouputdim)
        dims = list(range(1,len(ETrAAT.shape)))
        ETrAAT=ETrAAT.norm(dim=dims)
        return ETrAAT
    def get_ETrAAT_times(self,params,x,cotangents_variables):
        return vmap(self.get_ETrAAT, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def get_TrvJOJv_times(self,params,x,cotangents_variables):
        return vmap(self.get_TrvJOJv, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def backward(self,model, x, y , return_Normlization_Term_1=False, return_Normlization_Term_2=False):
        
        self.func_model, params =  make_functional(model)
        shape = y.shape
        cotangents_variables = torch.randint(2,(self.sample_times,*shape)).cuda()*2-1
        with torch.no_grad():
            if self.lambda1 != 0:
                Derivation_Term_1 = jacrev(self.Normlization_Term_1, argnums=0)(params, x)
            if self.lambda2 != 0:
                Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params, x,cotangents_variables)
        for i, param in enumerate(model.parameters()):
            delta_p = 0
            if self.lambda1 != 0:delta_p += self.lambda1*Derivation_Term_1[i]
            if self.lambda2 != 0:delta_p += self.lambda2*Derivation_Term_2[i]
            if param.grad is not None:
                param.grad.data += delta_p
            else:
                param.grad = delta_p
        out=[]
        with torch.no_grad():
            if return_Normlization_Term_1: 
                out.append(self.Normlization_Term_1(params, x))
            if return_Normlization_Term_2:
                out.append(self.Normlization_Term_2(params, x,cotangents_variables))
        return out


class NGmod_absolute(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x,cotangents_variables):
        return (((vmap(jacrev(self.func_model, argnums=1), (None, 0))(params, x)**2).sum(-1)-1)**2).mean()

class NGmod_delta_mean(Nodal_GradientModifier):
    
    def Normlization_Term_2(self,params,x,cotangents_variable):
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

class NGmode_estimate():
    def Normlization_Term_2(self,params,x,cotangents_variables):
        while True:
            TrvJOJvs,ETrAATs =  vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0 ))(params, x,cotangents_variables)
            # (S,B) # (S,B)
            CorrelationTerm = ETrAATs.mean(0) - TrvJOJvs.var(0)/2 #(B,)
            # reject when CorrelationTerm < 0 
            if torch.all(CorrelationTerm>0):break
        return CorrelationTerm.mean()