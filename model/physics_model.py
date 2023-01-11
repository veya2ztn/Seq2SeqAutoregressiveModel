import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch
from .afnonet import BaseModel
# $$
# \begin{align}
# \partial_t V_s &= F -  V_s \cdot \nabla_s   V_s  - \omega\partial_p V_s - \nabla_s\phi\\
# \partial_t T &= Q/C_v + \frac{RT}{C_pp}\omega  - V_s  \cdot \nabla _s   T -\omega \partial_pT\\
# \partial_t \phi&= wg  - V_s \cdot\nabla_s \phi-\omega\partial_p\phi\\
# 0&=\nabla_s\cdot V+ \frac{\partial \omega}{\partial p}
# \end{align}
# $$

class First_Derivative_Layer(torch.nn.Module):
    '''
    use high dimenstion conv is faster
        B=2
        P=4
        a=torch.randn(B,P,3,32,64).cuda()
        layer=  First_Derivative_Layer(dim=3).cuda()
        runtime_weight=layer.runtime_weight

        x = torch.conv3d(a.flatten(0,1).unsqueeze(1),runtime_weight).reshape(*a.shape[:-1],-1)
        --> 34.8 µs ± 130 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

        x2 = torch.conv1d(a.flatten(0,-2).unsqueeze(1),runtime_weight[0,0]).reshape(*a.shape[:-1],-1)
        --> 43.1 µs ± 1.63 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    def __init__(self,position=-1,dim=2,mode='five-point-stencil',pad_mode='circular',intervel=torch.Tensor([1])):
        super().__init__()
        self.postion=position
        self.dim    =dim
        self.mode   =mode
        self.pad_mode=pad_mode
        self.intervel= intervel
        self.conv_engine = [torch.conv1d, torch.conv2d, torch.conv3d][self.dim-1]
        if self.mode=='five-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([1/12,-8/12,0,8/12,-1/12]),requires_grad=False)
             self.pad_num = 2
        elif self.mode=='three-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([-1/2,0,1/2]),requires_grad=False)
             self.pad_num = 1
        elif isinstance(self.mode, int):
             self.weight = torch.nn.Parameter(torch.randn(self.mode)*0.01)
             self.pad_num     = self.mode
        else:
            raise NotImplementedError(f"the self.mode must be five-point-stencil or three-point-stencil or a int number to activate trainable derivate")
        padtuple = [0]*self.dim*2
        padtuple[(-self.postion-1)*2]   = self.pad_num
        padtuple[(-self.postion-1)*2+1] = self.pad_num
        self.padtuple = tuple(padtuple)

    @property
    def runtime_weight(self):
        if isinstance(self.mode, int):
            weight = torch.cat([-torch.flip(self.weight,(0,)),F.pad(self.weight,(1,0))])
        else:
            weight = self.weight
        return weight[(None,)*(self.dim + 1)].transpose(self.postion,-1)
    def forward(self, x):
        expandQ=False
        if len(x.shape) == self.dim + 1 and x.shape[1]!=1:
            x = x.unsqueeze(1)
            expandQ = True
        assert len(x.shape) == self.dim + 2
        # if only x dim, then the input should be (Batch, 1 , x)
        # if (x,y) pannel, then the input should be (Batch, 1 , x , y)
        # if (x,y,z) pannel, the the input should be (Batch, 1 , x , y , z)
        x = self.conv_engine(F.pad(x, self.padtuple, mode=self.pad_mode),self.runtime_weight)
        x = x/self.intervel.to(x.device)

        return x.squeeze(1) if expandQ else x

class Second_Derivative_Layer(torch.nn.Module):
    def __init__(self,position=(-2,-1),mode='nine-point-stencil',pad_mode='circular',intervel=torch.Tensor([1])):
        super().__init__()
        self.postion=position
        self.mode   =mode
        self.pad_mode=pad_mode
        self.intervel= intervel
        if self.mode=='nine-point-stencil':
            self.weight = torch.nn.Parameter(torch.Tensor([[1/3, 1/3,1/3],
                                                           [1/3,-8/3,1/3],
                                                           [1/3, 1/3,1/3]]),requires_grad=False)
            self.pad_num = 1
        else:
            raise NotImplementedError(f"the self.mode must be nine-point-stencil")

    @property
    def runtime_weight(self):
        return self.weight[None,None] #(1,1,3,3)
    def forward(self, x):
        x = x.transpose(-2,self.postion[0])
        x = x.transpose(-1,self.postion[1])
        
        oshape = x.shape
        x = x.flatten(0,-3).unsqueeze(1)
        x = F.pad(x, (self.pad_num,self.pad_num,self.pad_num,self.pad_num), mode=self.pad_mode)
        x = torch.conv2d(x,self.runtime_weight)
        x = x/self.intervel.to(x.device)
        x = x.reshape(*oshape)
        x = x.transpose(-1,self.postion[1])
        x = x.transpose(-2,self.postion[0])
        return x


class OnlineNormModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
    def forward(self, Field):
        mean = Field.mean(dim=(0,2,3),keepdim=True)
        std  = Field.std(dim=(0,2,3),keepdim=True)
        return self.backbone((Field-mean)/std)*std+mean

class DeltaModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
    def forward(self, Field):
        Delta = self.backbone(Field)
        return Delta + Field

class EulerEquationModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.Dx= First_Derivative_Layer(position=-1, dim=3)
        self.Dy= First_Derivative_Layer(position=-2, dim=3)
        self.Dz= First_Derivative_Layer(position=-3, dim=3, mode='three-point-stencil')
        self.thermal_factor = nn.Parameter(torch.randn(1))
        self.p_list         = nn.Parameter(torch.Tensor([10,8.5,5]).reshape(1,3,1,1),requires_grad=False)
        self.backbone =  backbone
        self.monitor = True
    
    
    def forward(self, Field):
        #Delta_u = Fx - u*u_dx - v*u_dy - o*u_dz - p_dx
        #Delta_v = Fy - u*v_dx - v*v_dy - o*v_dz - p_dy
        #Delta_T =  Q - u*T_dx - v*T_dy - o*T_dz + A*T/p*o
        #Delta_p =  W - u*P_dx - v*P_dy - o*P_dz
        #      0 = u_dx + v_dy + o_dz
        # input -> Field  = [u ,v, T, p] --> (Batch, 4, z, y ,x)
        # need generate unknown data [Fx, Fy , Q, W, o]
        b, si_z, i_y, i_x = Field.shape
        s=4
        i_z= si_z//4
        MachineLearningPart = self.backbone(Field).reshape(b, s+1, i_z, i_y, i_x) #(Batch, 5, z, y ,x)
        ExternalForce = MachineLearningPart[:,:4] #(Batch, 4, z, y ,x)
        o     = MachineLearningPart[:,4:5] #(Batch, 1, z, y ,x)
        Field = Field.reshape(b, s, i_z, i_y,  i_x) #(Batch, 5, z, y ,x)
        u = Field[:,0:1]
        v = Field[:,1:2]
        T = Field[:,2:3]
        p = Field[:,3:4]
        Field_dx = self.Dx(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Field_dy = self.Dy(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Field_dz = self.Dz(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        PhysicsPart  = torch.stack([-Field_dx[:,3], -Field_dy[:,3], self.thermal_factor*T[:,0]/self.p_list*o[:,0]],1)#(Batch,3,z, y ,x)
        PhysicsPart  = F.pad(PhysicsPart,(0,0,0,0,0,0,0,1)) #(Batch,4,z, y ,x)
        xydirection  = - u*Field_dx - v*Field_dy
        PhysicsPart  = xydirection - o*Field_dz + PhysicsPart
        Delta_Fd = ExternalForce + PhysicsPart
        Field    = Field+ Delta_Fd
        constrain= Field_dx[:,0] + Field_dy[:,1] + self.Dz(o[:,0])
        if not self.monitor:
            return Field.flatten(1,2),(constrain**2).mean()
        else:
            return Field.flatten(1,2),(constrain**2).mean(),{"ExternalForceFactor":(ExternalForce**2).mean().item(),
                                                             "PhysicsDrivenFactor":(PhysicsPart**2).mean().item(),
                                                             "xypannelDrivenFactor":(xydirection**2).mean().item()}

class EulerEquationModel2(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.Dx= First_Derivative_Layer(position=-1, dim=3, mode=2)
        self.Dy= First_Derivative_Layer(position=-2, dim=3, mode=2)
        self.Dz= First_Derivative_Layer(position=-3, dim=3, mode=1)
        self.thermal_factor = nn.Parameter(torch.randn(3).reshape(1,3,1,1))
        #self.p_list         = nn.Parameter(torch.Tensor([10,8.5,5]).reshape(1,3,1,1),requires_grad=False)
        self.backbone =  backbone
        self.monitor  = True
        self.physics_num= args.physics_num if hasattr(args,'physics_num') else 4
    def calculateEPI(self,Field):
        # u^{t+1} &= u^{t} + F_x - \nabla (Vu)  + u \nabla\cdot V - \partial_x\phi\\
        # v^{t+1} &= v^{t} + F_y - \nabla (Vv)  + v \nabla\cdot V - \partial_y\phi\\
        # T^{t+1} &= T^{t} + Q/C_v + \frac{RT}{C_pp}\omega  - \nabla (VT) +T \nabla\cdot V\\
        # \phi^{t+1}&=\phi^{t} + wg  - \nabla (V\phi)+ \phi \nabla\cdot V \\
        # 0&\approx \nabla\cdot V
        # input -> Field  = [u ,v, T, p] --> (Batch, 4, z, y ,x)
        # need generate unknown data [Fx, Fy , Q, W, o]
        #print(Field.shape)
        b, si_z, i_y, i_x = Field.shape
        s=self.physics_num
        i_z= si_z//self.physics_num
        MachineLearningPart = self.backbone(Field).reshape(b, s+1, i_z, i_y, i_x) #(Batch, 5, z, y ,x)
        ExternalForce = MachineLearningPart[:,:-1] #(Batch, 4, z, y ,x)
        o = MachineLearningPart[:,-1:] #(Batch, 1, z, y ,x)
        Field = Field.reshape(b, s, i_z, i_y,  i_x) #(Batch, 5, z, y ,x)
        u = Field[:,0:1]#(Batch, 1, z, y ,x)
        v = Field[:,1:2]#(Batch, 1, z, y ,x)
        T = Field[:,2:3]#(Batch, 1, z, y ,x)
        p = Field[:,3:4]#(Batch, 1, z, y ,x)
        V = torch.cat([u,v,o],1)#(Batch, 3, z, y ,x)
        Nabla_cdot_V = (self.Dx(u[:,0]) + self.Dy(v[:,0]) + self.Dz(o[:,0])).unsqueeze(1)#(Batch, 1, z, y ,x)
        Nabla_V_Field= Nabla_cdot_V*Field #(Batch, 4, z, y ,x)
        Vphysics     = torch.stack([V*u,V*v,V*T,V*p],1)#(Batch,4, 3, z, y ,x)
        
        Vphysics_dx  = self.Dx(Vphysics[:,:,0].flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Vphysics_dy  = self.Dy(Vphysics[:,:,1].flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Vphysics_dz  = self.Dz(Vphysics[:,:,2].flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        PhysicsPart  = -Vphysics_dx - Vphysics_dy - Vphysics_dz + Nabla_V_Field #(Batch,4,z, y ,x)

        InteractionPart= torch.stack([self.Dx(p[:,0]),
                                      self.Dy(p[:,0]),
                                      self.thermal_factor*T[:,0]*o[:,0]],1)#(Batch,3,z, y ,x)
        InteractionPart= F.pad(InteractionPart,(0,0,0,0,0,0,0,1)) #(Batch,4,z, y ,x)
        return Field, ExternalForce , PhysicsPart , InteractionPart, Nabla_cdot_V


    def forward(self, Field):
        Field, ExternalForce , PhysicsPart , InteractionPart, Nabla_cdot_V = self.calculateEPI(Field)
        
        Delta_Fd     = ExternalForce + PhysicsPart + InteractionPart
        Field        = Field+ Delta_Fd
        constrain    = Nabla_cdot_V
        if not self.monitor:
            return Field.flatten(1,2),(constrain**2).mean()
        else:
            return Field.flatten(1,2),(constrain**2).mean(),{"ExternalForceFactor":(ExternalForce**2).mean().item(),
                                                             "PhysicsDrivenFactor":(PhysicsPart**2).mean().item(),
                                                             "InteractionPart":(InteractionPart**2).mean().item()}

class EulerEquationModel3(EulerEquationModel2):
    def forward(self, Field):
        Field, ExternalForce , PhysicsPart , InteractionPart, Nabla_cdot_V = self.calculateEPI(Field)
        Delta_Fd     = ExternalForce + PhysicsPart + InteractionPart
        Field        = Field+ Delta_Fd
        constrain    = ((ExternalForce**2).mean() - ((PhysicsPart + InteractionPart)**2).mean())**2
        if not self.monitor:
            return Field.flatten(1,2),constrain
        else:
            return Field.flatten(1,2),constrain,{"ExternalForceFactor":(ExternalForce**2).mean().item(),
                                                             "PhysicsDrivenFactor":(PhysicsPart**2).mean().item(),
                                                             "InteractionPart":(InteractionPart**2).mean().item()}

class EulerEquationModel4(EulerEquationModel2):
    def forward(self, Field):
        Field, ExternalForce , PhysicsPart , InteractionPart, Nabla_cdot_V = self.calculateEPI(Field)
        Delta_Fd     = ExternalForce + PhysicsPart + InteractionPart
        Field        = Field+ Delta_Fd
        constrain    = torch.zeros_like(self.thermal_factor).mean()
        if not self.monitor:
            return Field.flatten(1,2),constrain
        else:
            return Field.flatten(1,2),constrain,{"ExternalForceFactor":(ExternalForce**2).mean().item(),
                                        "PhysicsDrivenFactor":(PhysicsPart**2).mean().item(),
                                        "InteractionPart":(InteractionPart**2).mean().item(),
                                        "Nabla_cdot_V":(Nabla_cdot_V**2).mean().item()}

import numpy as np
class ConVectionModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        h, w = backbone.img_size[-2:]
        print(f'h intervel: {h} , w intervel: {w}')
        if h in [51,49,47]:
            Hdx = 6371000*torch.sin(torch.linspace(0,720,49)/720*np.pi)*2*np.pi/w
            Hdx = Hdx.reshape(1,1,1,49,1)[...,1:-1,:] # (1,1,1,47,1)
            shape = Hdx.shape
            # the input will be (1,1,1,47,1)
            # if h == 51:Hdx = F.pad(Hdx.flatten(0,1),(0,0,2,2),mode='replicate').reshape(*shape[:-2],-1,shape[-1])# (1,1,1,51,1)
            # if h == 49:Hdx = F.pad(Hdx.flatten(0,1),(0,0,1,1),mode='replicate').reshape(*shape[:-2],-1,shape[-1])# (1,1,1,49,1)

            Hdy = torch.Tensor([6371000*np.pi/48])
            print(f"please notice we will using dt= 3600*6 as intertime")
            self.DT = 3600*6
            Hdx = Hdx/self.DT
            Hdy = Hdy/self.DT
            self.Hdx = Hdx
            self.Hdy = Hdy
        elif h in [32]:
            Hdx = 6371000*torch.sin(torch.linspace(0,1,34)*np.pi)*2*np.pi/w
            Hdx = Hdx[1:-1].reshape(1,1,1,32,1)
            shape = Hdx.shape
            Hdy = torch.Tensor([6371000*np.pi/32])
            print(f"please notice we will using dt= 3600*1 as intertime")
            self.DT = 3600*1 # notice for weatherbench module we have 1 hour dataset and 6 hour dataset. 
            Hdx = Hdx/self.DT
            Hdy = Hdy/self.DT
            self.Hdx = Hdx
            self.Hdy = Hdy
        else:
            raise NotImplementedError(f"for h not in [47,49,51], TODO.......")
        self.Dx= First_Derivative_Layer(position=-1, dim=3, mode=2, pad_mode='replicate', intervel=Hdx)
        self.Dy= First_Derivative_Layer(position=-2, dim=3, mode=2, pad_mode='replicate', intervel=Hdy)
        self.backbone =  backbone
        self.monitor = True

    def calculate_Advection(self,Field):
        u            = Field[:,0:1]
        v            = Field[:,1:2]
        Field_dx     = self.Dx(Field.flatten(0,1)).reshape(Field.shape)
        Field_dy     = self.Dy(Field.flatten(0,1)).reshape(Field.shape)

        Advection    = (u*Field_dx + v*Field_dy) #(B,P,z,y,x)
        return Advection


    def forward(self, normlized_Field):
        normlized_Dt = self.backbone(normlized_Field) #(B,P,z,y,x)
        return normlized_Dt

class DirectSpace_Feature_Model(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        h, w = backbone.img_size[-2:]
        print(f'h intervel: {h} , w intervel: {w}')
        if h in [51,49,47]:
            Hdx = 6371000*torch.sin(torch.linspace(0,720,49)/720*np.pi)*2*np.pi/w
            Hdx = Hdx.reshape(1,1,1,49,1)[...,1:-1,:] # (1,1,1,47,1)
            shape = Hdx.shape
            # the input will be (1,1,1,47,1)
            # if h == 51:Hdx = F.pad(Hdx.flatten(0,1),(0,0,2,2),mode='replicate').reshape(*shape[:-2],-1,shape[-1])# (1,1,1,51,1)
            # if h == 49:Hdx = F.pad(Hdx.flatten(0,1),(0,0,1,1),mode='replicate').reshape(*shape[:-2],-1,shape[-1])# (1,1,1,49,1)

            Hdy = torch.Tensor([6371000*np.pi/48])
            print(f"please notice we will using dt= 3600*6 as intertime")
            self.DT = 3600*6
            Hdx = Hdx/self.DT
            Hdy = Hdy/self.DT
            self.Hdx = Hdx
            self.Hdy = Hdy
        elif h in [32]:
            Hdx = 6371000*torch.sin(torch.linspace(0,1,34)*np.pi)*2*np.pi/w
            Hdx = Hdx[1:-1].reshape(1,1,1,32,1)
            shape = Hdx.shape
            Hdy = torch.Tensor([6371000*np.pi/32])
            print(f"please notice we will using dt= 3600*1 as intertime")
            self.DT = 3600*1 # notice for weatherbench module we have 1 hour dataset and 6 hour dataset. 
            Hdx = Hdx/self.DT
            Hdy = Hdy/self.DT
            self.Hdx = Hdx
            self.Hdy = Hdy
        else:
            raise NotImplementedError(f"for h not in [47,49,51], TODO.......")
        
        self.Dx= First_Derivative_Layer(position=-1, dim=2, mode='three-point-stencil', pad_mode='replicate', intervel=self.Hdx)
        self.Dy= First_Derivative_Layer(position=-2, dim=2, mode='three-point-stencil', pad_mode='replicate', intervel=self.Hdy)
        self.Dxy= Second_Derivative_Layer(position=(-2,-1), mode= 'nine-point-stencil', pad_mode='replicate', intervel=self.Hdy*self.Hdx)
        self.backbone    =  backbone
        self.physics_num = args.physics_num


    def forward(self, Field):
        # --> [Batch, z, P,  y, x] or --> [Batch, z, T, P, y, x]
        oshape = Field.shape
        if  Field.shape[1]!=self.physics_num:
            Field = Field.reshape(Field.shape[0],self.physics_num,-1,*Field.shape[2:])
        u  = Field[:,0:1] # --> [Batch, 1 ,P, y, x]
        v  = Field[:,1:2] # --> [Batch, 1 ,P, y, x]
        with torch.no_grad():
            Field_dx   =  self.Dx(Field.flatten(0,-3).unsqueeze(1)).reshape(Field.shape) # --> [Batch, z, P,  y, x]
            Field_dy   =  self.Dy(Field.flatten(0,-3).unsqueeze(1)).reshape(Field.shape) # --> [Batch, z, P,  y, x]
            Field_dxy  = self.Dxy(Field.flatten(0,-3).unsqueeze(1)).reshape(Field.shape) # --> [Batch, z, P,  y, x]
            
            Field = torch.cat([
                1*Field,u*v,
                1*Field_dx,  1*Field_dy,  1*Field_dxy, 
                u*Field_dx,  u*Field_dy,  u*Field_dxy, 
                v*Field_dx,  v*Field_dy,  v*Field_dxy, 
                u*v*Field_dx,u*v*Field_dy,u*v*Field_dxy, 
            ],dim=1)
        Field = self.backbone(Field)
        return Field.reshape(oshape)

class FeaturePickModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
    def forward(self, Field):
        return self.backbone(Field)

class OnlyPredSpeed(FeaturePickModel):
    train_for_part = list(range(28))
class WithoutSpeed(FeaturePickModel):
    train_for_part = list(range(28,70))    
class CrossSpeed(FeaturePickModel):
    train_for_part_extra = list(range(28))
    train_for_part = list(range(28,70))
class UVTP2p(FeaturePickModel):
    train_for_part = list(range(14*3,14*4-1))
class UVTPp2uvt(FeaturePickModel):
    train_for_part_extra = list(range(14*3,14*4-1))
    train_for_part = list(range(14*3))
class UVTP2uvt(FeaturePickModel):
    train_for_part = list(range(14*3))

class CombM_UVTP2p2uvt(BaseModel):
    def __init__(self,  args, backbone1, backbone2,ckpt1,ckpt2):
        super().__init__()
        self.UVTP2p  =  UVTP2p(args,backbone1)
        print(f"load UVTP2p model from {ckpt1}")
        self.UVTP2p.load_state_dict(torch.load(ckpt1, map_location='cpu')['model'])
        self.UVTPp2uvt = UVTPp2uvt(args,backbone2)
        print(f"load UVTPp2uvt model from {ckpt2}")
        self.UVTPp2uvt.load_state_dict(torch.load(ckpt2, map_location='cpu')['model'])
    def forward(self, UVTP):
        p = self.UVTP2p(UVTP)
        uvt= self.UVTPp2uvt(torch.cat([UVTP,p],1))
        return torch.cat([uvt,p],1)