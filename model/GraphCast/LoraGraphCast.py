
import torch.nn as nn
import torch
import torch.nn.functional as F
from .GraphCastDGL import GraphCastDGLBase

class LoRALinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.main = torch.nn.Linear(in_channel, out_channel, bias=False)
        self.lora = None
        # we will assign lora via outside function

    def forward(self, x):
        if self.lora is None:
            return self.main(x)
        else:
            assert not self.main.weight.requires_grad
            return self.main(x) + self.lora(x)

    
class LoRANode2Edge2NodeBlock(nn.Module):
    def __init__(self,embed_dim=128, do_source_update = False,**kargs):
        super().__init__()
        #initial_weight = fastinit1(3*embed_dim, embed_dim)
        #####--> notice the initial method changed
        #STE2E_S2E,STE2E_T2E,STE2E_E2E = torch.split(initial_weight,embed_dim)
        self.STE2E_S2E  = LoRALinear(embed_dim,embed_dim)
        self.STE2E_T2E  = LoRALinear(embed_dim,embed_dim)
        self.STE2E_E2E  = LoRALinear(embed_dim,embed_dim)
        self.ET2T_E2T  = LoRALinear(embed_dim,embed_dim)
        self.ET2T_T2T  = LoRALinear(embed_dim,embed_dim)
        self.activator1 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))
        self.activator2 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))

        self.S2S = None
        if do_source_update:
            self.S2S     = LoRALinear(embed_dim,embed_dim)
            self.activator3 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))

class LoRANode2Edge2NodeBlockDGLSymmetry(LoRANode2Edge2NodeBlock):
    def __init__(self, src_flag, edgetype, tgt_flag, embed_dim=128,do_source_update=False):
        super().__init__(embed_dim=embed_dim,do_source_update = do_source_update)
        self.src_flag = src_flag
        self.tgt_flag = tgt_flag
        self.edgetype = edgetype
        self.STE2E_T2E = None
    def forward(self,g):
        src_flag = self.src_flag 
        tgt_flag = self.tgt_flag 
        edgetype = self.edgetype 
        edgeflag = (src_flag,edgetype,tgt_flag)
        g.apply_edges(fn.u_add_v('feat', 'feat', 'node_to_edge'),etype=edgeflag)
        g.edges[edgeflag].data['add_feat'] = self.activator1(self.STE2E_S2E(g.edges[edgeflag].data['node_to_edge'])   + 
                                    self.STE2E_E2E(g.edges[edgeflag].data['feat'])     )
        g.edges[edgeflag].data['feat']  =   g.edges[edgeflag].data['feat'] + g.edges[edgeflag].data['add_feat']
        if 'coef' in g.edges[edgeflag].data:g.edges[edgeflag].data['add_feat']*= g.edges[edgeflag].data['coef']
        reduce_fun = fn.sum if 'coef' in g.edges[edgeflag].data else fn.mean
        g.update_all(fn.copy_e('add_feat','add_feat'),reduce_fun('add_feat', 'add_feat'),etype=edgeflag)
        g.nodes[tgt_flag].data['add_feat'] = self.activator2(self.ET2T_E2T(g.nodes[tgt_flag].data['add_feat']) + 
                                    self.ET2T_T2T(g.nodes[tgt_flag].data['feat']) )
        g.nodes[tgt_flag].data['feat']  = g.nodes[tgt_flag].data['feat'] + g.nodes[tgt_flag].data['add_feat']
        if self.S2S is not None:g.nodes[src_flag].data['feat'] = g.nodes[src_flag].data['feat']+ self.activator3(self.S2S(g.nodes[src_flag].data['feat']))
        return g

class LoRANode2Edge2NodeBlockDGL(LoRANode2Edge2NodeBlock):
    def __init__(self, src_flag, edgetype, tgt_flag, embed_dim=128,do_source_update=False):
        super().__init__(embed_dim=embed_dim,do_source_update = do_source_update)
        self.src_flag = src_flag
        self.tgt_flag = tgt_flag
        self.edgetype = edgetype
    def forward(self,g):
        src_flag = self.src_flag 
        tgt_flag = self.tgt_flag 
        edgetype = self.edgetype 
        edgeflag = (src_flag,edgetype,tgt_flag)
        g.nodes[src_flag].data['src']     = self.STE2E_S2E(g.nodes[src_flag].data['feat'])
        g.nodes[tgt_flag].data['dst']     = self.STE2E_T2E(g.nodes[tgt_flag].data['feat'])
        g.apply_edges(fn.u_add_v('src', 'dst', 'node_to_edge'),etype=edgeflag)
        g.edges[edgeflag].data['add_feat'] = self.activator1(g.edges[edgeflag].data['node_to_edge'] + self.STE2E_E2E(g.edges[edgeflag].data['feat']))
        g.edges[edgeflag].data['feat']  =   g.edges[edgeflag].data['feat'] + g.edges[edgeflag].data['add_feat']
        if 'coef' in g.edges[edgeflag].data:g.edges[edgeflag].data['add_feat']*= g.edges[edgeflag].data['coef']
        reduce_fun = fn.sum if 'coef' in g.edges[edgeflag].data else fn.mean
        g.update_all(fn.copy_e('add_feat','add_feat'),reduce_fun('add_feat', 'add_feat'),etype=edgeflag)
        g.nodes[tgt_flag].data['add_feat'] = self.activator2(self.ET2T_E2T(g.nodes[tgt_flag].data['add_feat'])  + 
                                    self.ET2T_T2T(g.nodes[tgt_flag].data['feat']) )
        g.nodes[tgt_flag].data['feat']  = g.nodes[tgt_flag].data['feat'] + g.nodes[tgt_flag].data['add_feat']
        if self.S2S is not None:g.nodes[src_flag].data['feat'] = g.nodes[src_flag].data['feat']+ self.activator3(self.S2S(g.nodes[src_flag].data['feat']) )
        return g

class LoRAGraphCastDGLSym(GraphCastDGLBase): 
    def __init__(self, img_size=(32,64),  in_chans=70, out_chans=70, depth=6, embed_dim=128, graphflag='mesh5', nonlinear='swish', **kargs):
        super().__init__()

        g = self.build_dgl(graphflag,img_size)

        #### build block ####
        edge_flag = ('mesh', 'M2M', 'mesh')
        self.grid2mesh = LoRANode2Edge2NodeBlockDGL('grid','G2M','mesh',embed_dim=embed_dim,do_source_update=True)
        self.mesh2mesh = nn.ModuleList()
        for i in range(depth):self.mesh2mesh.append(LoRANode2Edge2NodeBlockDGLSymmetry('mesh','M2M','mesh',embed_dim=embed_dim))        
        self.mesh2grid = LoRANode2Edge2NodeBlockDGL('mesh','M2G','grid',embed_dim=embed_dim)
        
        self.grid_rect_embedding_layer = LoRALinear(in_chans,embed_dim)
        self.projection         = LoRALinear(embed_dim,out_chans)
        self.northsouthembbed      = nn.Parameter(torch.randn(2,embed_dim))
        self.mesh_node_embedding    = nn.Parameter(torch.randn(g.num_nodes('mesh'),1, embed_dim))
        self.grid_mesh_bond_embedding  = nn.Parameter(torch.randn(g.num_edges('G2M'),1, embed_dim))
        #self.mesh_grid_bond_template   = torch.randn(g.num_edges('M2G'),1, embed_dim)
        self.g = g
        self.embed_dim = embed_dim

        M2Mweightorder1= g.edge_ids(self.M2M_edgeid2pair[:,0],self.M2M_edgeid2pair[:,1],etype='M2M')
        M2Mweightorder2= g.edge_ids(self.M2M_edgeid2pair[:,1],self.M2M_edgeid2pair[:,0],etype='M2M')
        mesh_mesh_bond_embedding = torch.randn(g.num_edges('M2M'),1, embed_dim)
        mesh_mesh_bond_embedding[M2Mweightorder2] = mesh_mesh_bond_embedding[M2Mweightorder2]    
        self.mesh_mesh_bond_embedding  = nn.Parameter(mesh_mesh_bond_embedding)

    @staticmethod
    def convertOLDweight(weigth):
        new_weight = {}
        for key, w in weigth.items():
            skip=False
            for activate_key in ['STE2E_S2E','STE2E_T2E','STE2E_E2E',
                                'ET2T_E2T','ET2T_T2T','S2S']:
                if activate_key in key:
                    key += ".main.weight"
                    new_weight[key] = w.transpose(1,0)
                    skip = True
                    continue
                    
            if skip:continue
            new_weight[key] = w
        return model2
