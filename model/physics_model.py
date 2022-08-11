import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch

# $$
# \begin{align}
# \partial_t V_s &= F -  V_s \cdot \nabla_s   V_s  - \omega\partial_p V_s - \nabla_s\phi\\
# \partial_t T &= Q/C_v + \frac{RT}{C_pp}\omega  - V_s  \cdot \nabla _s   T -\omega \partial_pT\\
# \partial_t \phi&= wg  - V_s \cdot\nabla_s \phi-\omega\partial_p\phi\\
# 0&=\nabla_s\cdot V+ \frac{\partial \omega}{\partial p}
# \end{align}
# $$

class First_Derivative_Layer(torch.nn.Module):
    def __init__(self,position=-1,dim=2,mode='five-point-stencil'):
        super().__init__()
        self.postion=position
        self.dim    =dim
        self.mode   =mode
        self.conv_engine = [torch.conv1d, torch.conv2d, torch.conv3d][self.dim-1]
        if self.mode=='five-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([1/12,-8/12,0,8/12,-1/12]),requires_grad=False)
             self.pad_num = 2
        elif self.mode=='three-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([-1/4,0,1/4]),requires_grad=False)
             self.pad_num = 1
        elif isinstance(self.mode, int):
             self.weight = torch.nn.Parameter(torch.randn(self.mode))
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
            weight = torch.cat([torch.flip(self.weight,(0,)),F.pad(self.weight,(1,0))])
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
        x = self.conv_engine(F.pad(x, self.padtuple, mode='circular'),self.runtime_weight)
        return x.squeeze(1) if expandQ else x

class EulerEquationModel(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.Dx= First_Derivative_Layer(position=-1, dim=3)
        self.Dy= First_Derivative_Layer(position=-2, dim=3)
        self.Dz= First_Derivative_Layer(position=-3, dim=3, mode='three-point-stencil')
        self.thermal_factor = nn.Parameter(torch.randn(1))
        self.p_list         = nn.Parameter(torch.Tensor([10,8.5,5]).reshape(1,3,1,1),requires_grad=False)
        self.backbone =  backbone
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
        o = MachineLearningPart[:,4:5] #(Batch, 1, z, y ,x)
        Field = Field.reshape(b, s, i_z, i_y, i_x) #(Batch, 5, z, y ,x)
        u = Field[:,0:1]
        v = Field[:,1:2]
        T = Field[:,2:3]
        p = Field[:,3:4]
        Field_dx = self.Dx(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Field_dy = self.Dy(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        Field_dz = self.Dz(Field.flatten(0,1)).reshape(Field.shape)#(Batch, 4, z, y ,x)
        ResPart  = torch.stack([-Field_dx[:,3], -Field_dy[:,3], self.thermal_factor*T[:,0]/self.p_list*o[:,0]],1)#(Batch,3,z, y ,x)
        ResPart  = F.pad(ResPart,(0,0,0,0,0,0,0,1)) #(Batch,4,z, y ,x)
        Delta_Fd = ExternalForce -  u*Field_dx - v*Field_dy - o*Field_dz + ResPart
        Field    = Field+ Delta_Fd
        constrain= Field_dx[:,0] + Field_dy[:,1] + self.Dz(o[:,0])
        return Field.flatten(1,2),constrain.mean()
