import torch
import torch.nn.functional as F
import functorch
from functorch import jacrev
from functorch import make_functional, vmap
import torch

class Nodal_GradientModifier:
    def __init__(self,lambda1=100,lambda2=100,sample_times=10):
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
        vJOJv= (vJ*vJO).sum(dims)#should sum over all dimension except batch
        return vJOJv, functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1].norm()# average the batch_size also
    def get_TrvJOJv(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJ*vJO).sum(-1)#should sum over all dimension except batch
        return vJOJv
    def get_ETrAAT(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
        return functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1].norm()# average the batch_size also
    def get_ETrAAT_times(self,params,x,cotangents_variables):
        return vmap(get_ETrAAT, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def get_TrvJOJv_times(self,params,x,cotangents_variables):
        return vmap(self.get_TrvJOJv, (None, None, 0 ))(params, x,cotangents_variables).mean()
    def Normlization_Term_2(self,params,x,cotangents_variables):
        TrvJOJvs,ETrAATs =  vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0 ))(params, x,cotangents_variables)
        return ETrAATs.mean() - torch.var(TrvJOJvs,0).mean()
    def Normlization_Term_2_Full(model, params,x):
        return (((vmap(jacrev(self.func_model, argnums=1), (None, 0))(params, x)**2).sum(-1)-1)**2).mean()   
    
    def backward(self,model, x, y ):
        
        self.func_model, params =  make_functional(model)
        shape = y.shape
        cotangents_variables = torch.randint(2,(self.sample_times,*shape)).cuda()*2-1
        with torch.no_grad():
            Derivation_Term_1 = jacrev(self.Normlization_Term_1, argnums=0)(params, x)
            Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params, x,cotangents_variables)
            #torch.cuda.empty_cache() # usually not helpful to save memory
        for param, d1,d2 in zip(model.parameters(), Derivation_Term_1,Derivation_Term_2):
            if param.grad is not None:param.grad.data += d1
            elif d1 is not None:
                param.grad = d1 + d2
        