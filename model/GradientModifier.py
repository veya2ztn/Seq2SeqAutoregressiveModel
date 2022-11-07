import torch
import torch.nn.functional as F
import functorch
from functorch import jacrev
from functorch import make_functional, vmap, make_functional_with_buffers
import torch
import numpy as np

class Nodal_GradientModifier:
    def __init__(self, lambda1=1, lambda2=1, sample_times=10, do_unit_renormalize=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sample_times = sample_times 
        self.cotangents_sum_along_x_dimension = None
        self.do_unit_renormalize = do_unit_renormalize

    def Normlization_Term_1(self, params, x):
        if self.cotangents_sum_along_x_dimension is None or self.cotangents_sum_along_x_dimension.shape != x.shape:
            self.cotangents_sum_along_x_dimension = torch.ones_like(x)
        values = ((functorch.jvp(lambda x: self.func_model(params, x),
                  (x,), (self.cotangents_sum_along_x_dimension,))[1]-1)**2).mean()
        #(B, Outputdim) -> (1,)
        N = np.prod(x.shape[1:])
        if self.do_unit_renormalize:
            # the varation of L1 for iid normal distribution is around 2n^2 + 4n
            coef   = 2*np.power(N, 2) + 4*N
            values = values/np.sqrt(coef)
        return values
        
    def TrvJOJv_and_ETrAAT(self,params,x,cotangents_variable):
        _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,x), x)
        vJ   = vJ_fn(cotangents_variable)[0]
        dims = list(range(1,len(vJ.shape)))
        vJO  = vJ.sum(dims,keepdims=True)-vJ # <vJ|1-I|
        vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
        #vJOJv= vJOJv/np.sqrt(torch.prod(self.output_shape).item())
        ETrAAT = functorch.jvp(lambda x:self.func_model(params,x), (x,), (vJO,))[1] # (B,Ouputdim)
        dims = list(range(1,len(ETrAAT.shape)))
        ETrAAT=torch.sum(ETrAAT**2,dim=dims)
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
    
    def inference(self,model,x,y):
        self.func_model, params =  make_functional(model,disable_autograd_tracking=True)
        shape = y.shape
        cotangents_variables = torch.randint(2,(self.sample_times,*shape)).cuda()*2-1
        with torch.no_grad():
            L1=self.Normlization_Term_1(params, x).item()
            L2=self.Normlization_Term_2(params, x).item()
        return L1,L2
    
    
    def backward(self,model, x, y):
        self.func_model, params = make_functional(model,disable_autograd_tracking=True)
        self.output_shape       = y.shape[1:]
        with torch.no_grad():
            if self.lambda1 != 0:
                Derivation_Term_1 = jacrev(self.Normlization_Term_1, argnums=0)(params, x)
            if self.lambda2 != 0:
                Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params, x)
        for i, param in enumerate(model.parameters()):
            delta_p = 0
            if self.lambda1 != 0:delta_p += self.lambda1*Derivation_Term_1[i]
            if self.lambda2 != 0:delta_p += self.lambda2*Derivation_Term_2[i]
            if param.grad is not None:
                param.grad.data += delta_p
            else:
                param.grad = delta_p
    

class NGmod_absolute(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x,cotangents_variables):
        values = (((vmap(jacrev(self.func_model, argnums=1), (None, 0))(params, x)**2).sum(-1)-1)**2).mean()
        N      = np.prod(x.shape[1:])
        if self.do_unit_renormalize: 
            coef = 8*np.power(N,3) + 24*np.power(N,2) + 24*N
            values = values/np.sqrt(coef)
            # the varation of L2 for iid normal distribution is around 8n^3 + 24n^2 + 24n
        return values

class NGmod_absoluteNone(NGmod_absolute):
    def backward(self,model, x, y):
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


class NGmode_estimate(Nodal_GradientModifier):
    def Normlization_Term_2(self,params,x):
        Batch_size    = x.size(0)
        work_index    = list(range(Batch_size))
        good_estimate = torch.zeros(Batch_size).to(x.device)
        while len(work_index)>0:
            cotangents_variables = torch.randint(2,(self.sample_times,len(work_index),*self.output_shape)).cuda()*2-1
            TrvJOJvs,ETrAATs = vmap(self.TrvJOJv_and_ETrAAT, (None, None, 0 ))(params, x[work_index],cotangents_variables)
            CorrelationTerm  = ETrAATs.mean(0) - TrvJOJvs.var(0)/2 #(B,)
            good_index = torch.where(CorrelationTerm>0)[0].cpu().numpy().tolist()
            real_index = [work_index[idx] for idx in good_index]
            good_estimate[real_index]=CorrelationTerm[good_index]
            work_index = list(set(work_index) - set(real_index))
        return good_estimate.mean()/np.sqrt(np.prod(list(self.output_shape)))


# class Nodal_GradientModifierBuff:
#     def __init__(self,lambda1=1,lambda2=1,sample_times=10):
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.sample_times = sample_times
#         self.cotangents_sum_along_x_dimension = None
#     def Normlization_Term_1(self,params,buffers,x):
#         if self.cotangents_sum_along_x_dimension is None or self.cotangents_sum_along_x_dimension.shape!=x.shape:
#             self.cotangents_sum_along_x_dimension = torch.ones_like(x)
#         return ((functorch.jvp(lambda x:self.func_model(params,buffers,x), (x,), (self.cotangents_sum_along_x_dimension,))[1]-1)**2).mean()
#     def TrvJOJv_and_ETrAAT(self,params,buffers,x,cotangents_variable):
#         _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,buffers,x), x)
#         vJ   = vJ_fn(cotangents_variable)[0]
#         dims = list(range(1,len(vJ.shape)))
#         vJO  = vJ.sum(dims,keepdims=True)-vJ # <vJ|1-I|
#         vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
#         ETrAAT = functorch.jvp(lambda x:self.func_model(params,buffers,x), (x,), (vJO,))[1] # (B,Ouputdim)
#         dims = list(range(1,len(ETrAAT.shape)))
#         ETrAAT=ETrAAT.norm(dim=dims)
#         return vJOJv, ETrAAT# DO NOT average the batch_size also
#     def get_TrvJOJv(self,params,buffers,x,cotangents_variable):
#         _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,buffers,x), x)
#         vJ   = vJ_fn(cotangents_variable)[0]
#         dims = list(range(1,len(vJ.shape)))
#         vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
#         vJOJv= (vJO*vJ).sum(dim=dims)#should sum over all dimension except batch
#         return vJOJv
#     def get_ETrAAT(self,params,buffers,x,cotangents_variable):
#         _, vJ_fn = functorch.vjp(lambda x:self.func_model(params,buffers,x), x)
#         vJ   = vJ_fn(cotangents_variable)[0]
#         vJO  = vJ.sum(1,keepdims=True)-vJ # <vJ|1-I|
#         ETrAAT = functorch.jvp(lambda x:self.func_model(params,buffers,x), (x,), (vJO,))[1] # (B,Ouputdim)
#         dims = list(range(1,len(ETrAAT.shape)))
#         ETrAAT=ETrAAT.norm(dim=dims)
#         return ETrAAT
#     def get_ETrAAT_times(self,params,buffers,x,cotangents_variables):
#         return vmap(self.get_ETrAAT, (None, None, None,0 ), randomness='same')(params,buffers,x,cotangents_variables).mean()
#     def get_TrvJOJv_times(self,params,buffers,x,cotangents_variables):
#         return vmap(self.get_TrvJOJv, (None, None, None,0 ), randomness='same')(params,buffers,x,cotangents_variables).mean()
#     def backward(self,model, x, y , return_Normlization_Term_1=False, return_Normlization_Term_2=False):
#         self.func_model, params, buffers = make_functional_with_buffers(model)
#         shape = y.shape
#         cotangents_variables = torch.randint(2,(self.sample_times,*shape)).cuda()*2-1
#         with torch.no_grad():
#             if self.lambda1 != 0:
#                 Derivation_Term_1 = jacrev(self.Normlization_Term_1, argnums=0)(params,buffers,x)
#             if self.lambda2 != 0:
#                 Derivation_Term_2 = jacrev(self.Normlization_Term_2, argnums=0)(params,buffers,x,cotangents_variables)
#         for i, param in enumerate(model.parameters()):
#             delta_p = 0
#             if self.lambda1 != 0:delta_p += self.lambda1*Derivation_Term_1[i]
#             if self.lambda2 != 0:delta_p += self.lambda2*Derivation_Term_2[i]
#             if param.grad is not None:
#                 param.grad.data += delta_p
#             else:
#                 param.grad = delta_p
#         out=[]
#         with torch.no_grad():
#             if return_Normlization_Term_1: 
#                 out.append(self.Normlization_Term_1(params,buffers,x).item())
#             if return_Normlization_Term_2:
#                 out.append(self.Normlization_Term_2(
#                     params,buffers,x, cotangents_variables).item())
#         return out

# class NGmod_absoluteBuff(Nodal_GradientModifierBuff):
#     def Normlization_Term_2(self,params,buffers,x,cotangents_variables):
#         return (((vmap(jacrev(self.func_model, argnums=1), (None, None , 0), randomness='same')(params,buffers,x)**2).sum(-1)-1)**2).mean()