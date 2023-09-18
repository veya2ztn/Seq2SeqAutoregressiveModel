import torch.nn as nn
import numpy as np
import torch
import os
import dgl
import dgl.function as fn
from einops import rearrange
from .GraphCast import Node2Edge2NodeBlock, MeshCast

class Node2Edge2NodeBlockDGL(Node2Edge2NodeBlock):
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
        g.nodes[src_flag].data['src']     = g.nodes[src_flag].data['feat'] @ self.STE2E_S2E
        g.nodes[tgt_flag].data['dst']     = g.nodes[tgt_flag].data['feat'] @ self.STE2E_T2E
        g.apply_edges(fn.u_add_v('src', 'dst', 'node_to_edge'),etype=edgeflag)
        g.edges[edgeflag].data['add_feat'] = self.activator1(g.edges[edgeflag].data['node_to_edge'] + g.edges[edgeflag].data['feat'] @ self.STE2E_E2E)
        g.edges[edgeflag].data['feat']  =   g.edges[edgeflag].data['feat'] + g.edges[edgeflag].data['add_feat']
        if 'coef' in g.edges[edgeflag].data:g.edges[edgeflag].data['add_feat']*= g.edges[edgeflag].data['coef']
        reduce_fun = fn.sum if 'coef' in g.edges[edgeflag].data else fn.mean
        g.update_all(fn.copy_e('add_feat','add_feat'),reduce_fun('add_feat', 'add_feat'),etype=edgeflag)
        g.nodes[tgt_flag].data['add_feat'] = self.activator2(g.nodes[tgt_flag].data['add_feat'] @ self.ET2T_E2T + 
                                    g.nodes[tgt_flag].data['feat'] @ self.ET2T_T2T)
        g.nodes[tgt_flag].data['feat']  = g.nodes[tgt_flag].data['feat'] + g.nodes[tgt_flag].data['add_feat']
        if self.S2S is not None:g.nodes[src_flag].data['feat'] = g.nodes[src_flag].data['feat']+ self.activator3(g.nodes[src_flag].data['feat']@ self.S2S)
        return g

class GraphCastDGLBase(MeshCast):
    def build_dgl(self, graphflag, img_size):
        flag = graphflag
        resolution_flag = f'{img_size[0]}x{img_size[1]}'
        ROOTPATH = f"GraphCastStructure/{flag}"

        self.M2M_edgeid2pair = M2M_edgeid2pair = torch.LongTensor(
            np.load(os.path.join(ROOTPATH, f"M2M_edgeid2pair.npy")))
        ROOTPATH = f"GraphCastStructure/{flag}/{resolution_flag}"
        G2M_edge_id2pair_tensor = (
            np.load(os.path.join(ROOTPATH, f"G2M_edge_id2pair_tensor.npy")))
        # M2G_edge_id2pair_tensor = (np.load(os.path.join(ROOTPATH,f"M2G_id2edge_max_rank.npy")))
        M2G_edge_id2pair_tensor = (
            np.load(os.path.join(ROOTPATH, f"M2G_id2edge_max_rank.npy")))
        self.G2M_grid2LaLotudePos = np.load(
            os.path.join(ROOTPATH, f"G2M_grid2LaLotudePos.npy"))
        self.M2G_LaLotudePos2grid = np.load(os.path.join(
            ROOTPATH, f"M2G_LaLotudeGrid2rect_tensor.npy"))

        graph_data = {
            ('mesh', 'M2M', 'mesh'): (np.concatenate([M2M_edgeid2pair[:, 0], M2M_edgeid2pair[:, 1]]),
                                        np.concatenate([M2M_edgeid2pair[:, 1], M2M_edgeid2pair[:, 0]])),
            ('grid', 'G2M', 'mesh'): (G2M_edge_id2pair_tensor[:, 0], G2M_edge_id2pair_tensor[:, 1]),
            ('mesh', 'M2G', 'grid'): (M2G_edge_id2pair_tensor[:, 1], M2G_edge_id2pair_tensor[:, 0]),

        }
        g = dgl.heterograph(graph_data)

        total_mesh_node = g.num_nodes('mesh')
        total_mesh_edge = g.num_edges('M2M')
        total_grid_node = g.num_nodes('grid') - 2
        total_G2M_edges = g.num_edges('G2M')
        total_M2G_edges = g.num_edges('M2G')
        self.activated_gridn = activated_gridn = G2M_edge_id2pair_tensor[:, 0].max(
        )+1
        self.unactivated_grid = unactivated_grid = g.num_nodes(
            'grid') - activated_gridn

        # notice M2G_edge_id2pair_tensor still use (rect_id, node_id)
        # we firstly select those pair
        shared_index_in_M2G = np.where(
            M2G_edge_id2pair_tensor[:, 0] <= G2M_edge_id2pair_tensor[:, 0].max())[0]
        shared_pair_u = M2G_edge_id2pair_tensor[shared_index_in_M2G, 0]
        shared_pair_v = M2G_edge_id2pair_tensor[shared_index_in_M2G, 1]
        # now we need find the order of all the ('grid', 'G2M', 'mesh') edge
        # in ('mesh', 'M2G', 'grid') subgraph
        # then we should know the location for each pair in G2M edge list
        # one thing omit here is that the egde order in M2G is along the M2G_edge_id2pair_tensor index since we use it to create graph
        self.reorder_edge_id_of_M2G_from_G2M = g.edge_ids(
            shared_pair_u, shared_pair_v, etype=('grid', 'G2M', 'mesh'))
        self.reorder_edge_id_in_M2G = shared_index_in_M2G
        M2G_edge_from_node_to_activate_grid = len(
            self.reorder_edge_id_of_M2G_from_G2M)
        self.num_unactivated_edge = num_unactivated_edge = total_M2G_edges - \
            M2G_edge_from_node_to_activate_grid
        print(f'''
        This is ===> GraphCast Model(DGL) <===
        Information: 
            total mesh node:{total_mesh_node:5d} total unique mesh edge:{total_mesh_edge//2:5d}*2={total_mesh_edge:5d} 
            total grid node {total_grid_node}+2 = {total_grid_node+2} but activated grid {activated_gridn} 
            from activated grid to mesh, create 4*{total_mesh_node} - 6 = {total_G2M_edges} edges. (north and south pole repeat 4 times) 
            there are {unactivated_grid} unactivated grid node
            when mapping node to grid, 
            from node to activated grid, there are {M2G_edge_from_node_to_activate_grid} edges
            from node to unactivated grid, there are {num_unactivated_edge} edges
            thus, totally have {total_M2G_edges} edge. 
            #notice some grid only have 1-2 linked node but some grid may have 30 lined node
        ''')

        G2M_rect_of_node_tensor = np.load(os.path.join(
            ROOTPATH, f"G2M_rect_of_node_tensor.npy"))
        G2M_rect_distant_tensor = np.load(os.path.join(
            ROOTPATH, f"G2M_rect_distant_tensor.npy"))
        G2M_rect_node_tensor = torch.LongTensor(G2M_rect_of_node_tensor)
        G2M_rect_coef_tensor = torch.softmax(
            torch.FloatTensor(G2M_rect_distant_tensor), axis=-1)

        #### create_edge_coef_in_grid2mesh ######
        edge_flag = ('grid', 'G2M', 'mesh')
        NRC_tensor = torch.Tensor([(node_id, rect_id, coef) for node_id, (rect_list, coef_list) in enumerate(zip(
            G2M_rect_node_tensor, G2M_rect_coef_tensor)) for rect_id, coef in zip(rect_list, coef_list)])
        edge_idlist = g.edge_ids(NRC_tensor[:, 1].long(
        ), NRC_tensor[:, 0].long(), etype=edge_flag)
        edge_ids = {}
        for _id, coef in zip(edge_idlist, NRC_tensor[:, 2]):
            _id = _id.item()
            if _id not in edge_ids:
                edge_ids[_id] = 0
            edge_ids[_id] += coef
        edge_coef = torch.stack([edge_ids[i]
                                for i in range(len(edge_ids))])
        self.G2M_edge_coef = edge_coef.unsqueeze(-1).unsqueeze(-1)
        # g.edges[edge_flag].data['coef'] =

        #### create_edge_coef_in_mesh2grid ######
        edge_flag = ('mesh', 'M2G', 'grid')
        M2G_node_of_rect_tensor = np.load(os.path.join(
            ROOTPATH, f"M2G_node_of_rect_tensor.npy"))
        M2G_node_distant_tensor = np.load(os.path.join(
            ROOTPATH, f"M2G_node_distant_tensor.npy"))

        M2G_node_of_rect_tensor = torch.LongTensor(M2G_node_of_rect_tensor)
        M2G_node_distant_tensor = torch.softmax(
            torch.FloatTensor(M2G_node_distant_tensor), axis=-1)

        NRC_tensor = torch.Tensor([(node_id, rect_id, coef) for rect_id, (node_list, coef_list) in enumerate(zip(
            M2G_node_of_rect_tensor, M2G_node_distant_tensor)) for node_id, coef in zip(node_list, coef_list) if node_id >= 0])
        edge_idlist = g.edge_ids(NRC_tensor[:, 0].long(
        ), NRC_tensor[:, 1].long(), etype=edge_flag)

        edge_ids = {}
        for _id, coef in zip(edge_idlist, NRC_tensor[:, 2]):
            _id = _id.item()
            if _id not in edge_ids:
                edge_ids[_id] = 0
            edge_ids[_id] += coef
        edge_coef = torch.stack([edge_ids[i]
                                for i in range(len(edge_ids))])
        self.M2G_edge_coef = edge_coef.unsqueeze(-1).unsqueeze(-1)
        # g.edges[edge_flag].data['coef'] = torch.nn.Parameter(edge_coef ,requires_grad=False) # to automatively go into cuda
        self.device = None
        return g

    def forward(self, _input):
        B, P, W, H = _input.shape
        device = next(self.parameters()).device
        if self.device is None:
            self.device = device
            self.g = self.g.to(device)
        # (B,P,W,H) -> (B,W*H,P)
        feature_along_latlot = self.grid_rect_embedding_layer(
            rearrange(_input, "B P W H -> (W H) B P"))
        # (L,B,D)
        grid_rect_embedding = feature_along_latlot[self.G2M_grid2LaLotudePos]
        grid_rect_embedding = torch.cat([rearrange(self.northsouthembbed.repeat(B, 1, 1), "B L D -> L B D"),
                                            grid_rect_embedding])  # --> (L+2, B, D)
        L = len(grid_rect_embedding)
        g = self.g
        g.nodes['grid'].data['feat'] = torch.nn.functional.pad(
            grid_rect_embedding, (0, 0, 0, 0, 0, self.unactivated_grid))
        g.nodes['mesh'].data['feat'] = self.mesh_node_embedding
        g.edges['G2M'].data['feat'] = self.grid_mesh_bond_embedding
        g.edges['M2M'].data['feat'] = self.mesh_mesh_bond_embedding
        g.edges['G2M'].data['coef'] = self.G2M_edge_coef.to(device)
        g.edges['M2G'].data['coef'] = self.M2G_edge_coef.to(device)
        # checknan(g,'initial');
        g = self.grid2mesh(g)
        # checknan(g,'grid2mesh');
        for layer_idx, mesh2mesh in enumerate(self.mesh2mesh):
            g = mesh2mesh(g)  # checknan(g,f"mesh2mesh_{layer_idx}")
        g.edges['M2G'].data['feat'] = torch.nn.functional.pad(
            g.edges['G2M'].data['feat'][self.reorder_edge_id_of_M2G_from_G2M], (0, 0, 0, 0, 0, self.num_unactivated_edge))
        # luckly, the self.reorder_edge_id_in_M2G is just np.arange(....) thus, use padding rather than create one can fast 50%
        # self.mesh_grid_bond_template = torch.zeros(g.num_edges('M2G'),B,self.embed_dim).to(g.edges['G2M'].data['feat'].device)
        # self.mesh_grid_bond_template[self.reorder_edge_id_in_M2G] = g.edges['G2M'].data['feat'][self.reorder_edge_id_of_M2G_from_G2M]
        # g.edges['M2G'].data['feat'] = self.mesh_grid_bond_template
        # g.nodes['grid'].data['feat'][L:] = 0 <--- only to align with GraphCastFast
        g = self.mesh2grid(g)  # checknan(g,'mesh2grid')
        # (64,128,B,embed_dim)
        out = g.nodes['grid'].data['feat'][self.M2G_LaLotudePos2grid]
        return self.projection(out).permute(2, 3, 0, 1)

class GraphCastDGL(GraphCastDGLBase):    
    '''
    ====>  fastest in atom operation test, but slower than GraphCastFast in practice.
    ====>  still faster than normal GraphCast
    Repreduce of GraphCast in Pytorch.
    GraphCast has three part:
    - Grid to Mesh
    - Mesh to Mesh
    - Mesh to Grid
    -------------------------------------
    the input is a tensor (B, P, W, H), but the internal tensor all with shape (B, L ,P)
    where the L equal the node number or edge number.
    '''
    def __init__(self, img_size=(32,64),  in_chans=70, out_chans=70, depth=6, embed_dim=128, graphflag='mesh5', nonlinear='swish', **kargs):
        super().__init__()

        g = self.build_dgl(graphflag,img_size)

        #### build block ####
        edge_flag = ('mesh', 'M2M', 'mesh')
        self.grid2mesh = Node2Edge2NodeBlockDGL('grid','G2M','mesh',embed_dim=embed_dim,do_source_update=True)
        self.mesh2mesh = nn.ModuleList()
        for i in range(depth):self.mesh2mesh.append(Node2Edge2NodeBlockDGL('mesh','M2M','mesh',embed_dim=embed_dim))        
        self.mesh2grid = Node2Edge2NodeBlockDGL('mesh','M2G','grid',embed_dim=embed_dim)
        
        self.grid_rect_embedding_layer = nn.Linear(in_chans,embed_dim)
        self.projection        = nn.Linear(embed_dim,out_chans)
        self.northsouthembbed     = nn.Parameter(torch.randn(2,embed_dim))
        self.mesh_node_embedding   = nn.Parameter(torch.randn(g.num_nodes('mesh'),1, embed_dim))
        self.grid_mesh_bond_embedding  = nn.Parameter(torch.randn(g.num_edges('G2M'),1, embed_dim))
        self.mesh_grid_bond_template   = torch.randn(g.num_edges('M2G'),1, embed_dim)
        self.g = g
        self.embed_dim = embed_dim
        mesh_mesh_bond_embedding = torch.randn(g.num_edges('M2M'),1, embed_dim) 
        self.mesh_mesh_bond_embedding  = nn.Parameter(mesh_mesh_bond_embedding)
    
class Node2Edge2NodeBlockDGLSymmetry(Node2Edge2NodeBlock):
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
        g.edges[edgeflag].data['add_feat'] = self.activator1(g.edges[edgeflag].data['node_to_edge']@ self.STE2E_S2E + 
                                    g.edges[edgeflag].data['feat']    @ self.STE2E_E2E)
        g.edges[edgeflag].data['feat']  =   g.edges[edgeflag].data['feat'] + g.edges[edgeflag].data['add_feat']
        if 'coef' in g.edges[edgeflag].data:g.edges[edgeflag].data['add_feat']*= g.edges[edgeflag].data['coef']
        reduce_fun = fn.sum if 'coef' in g.edges[edgeflag].data else fn.mean
        g.update_all(fn.copy_e('add_feat','add_feat'),reduce_fun('add_feat', 'add_feat'),etype=edgeflag)
        g.nodes[tgt_flag].data['add_feat'] = self.activator2(g.nodes[tgt_flag].data['add_feat'] @ self.ET2T_E2T + 
                                    g.nodes[tgt_flag].data['feat'] @ self.ET2T_T2T)
        g.nodes[tgt_flag].data['feat']  = g.nodes[tgt_flag].data['feat'] + g.nodes[tgt_flag].data['add_feat']
        if self.S2S is not None:g.nodes[src_flag].data['feat'] = g.nodes[src_flag].data['feat']+ self.activator3(g.nodes[src_flag].data['feat']@ self.S2S)
        return g

class GraphCastDGLSym(GraphCastDGLBase):  
    def __init__(self, img_size=(32,64),  in_chans=70, out_chans=70, depth=6, embed_dim=128, graphflag='mesh5', nonlinear='swish', **kargs):
        super().__init__()

        g = self.build_dgl(graphflag,img_size)

        #### build block ####
        edge_flag = ('mesh', 'M2M', 'mesh')
        self.grid2mesh = Node2Edge2NodeBlockDGL('grid','G2M','mesh',embed_dim=embed_dim,do_source_update=True)
        self.mesh2mesh = nn.ModuleList()
        for i in range(depth):self.mesh2mesh.append(Node2Edge2NodeBlockDGLSymmetry('mesh','M2M','mesh',embed_dim=embed_dim))        
        self.mesh2grid = Node2Edge2NodeBlockDGL('mesh','M2G','grid',embed_dim=embed_dim)
        
        self.grid_rect_embedding_layer = nn.Linear(in_chans,embed_dim)
        self.projection        = nn.Linear(embed_dim,out_chans)
        self.northsouthembbed     = nn.Parameter(torch.randn(2,embed_dim))
        self.mesh_node_embedding   = nn.Parameter(torch.randn(g.num_nodes('mesh'),1, embed_dim))
        self.grid_mesh_bond_embedding  = nn.Parameter(torch.randn(g.num_edges('G2M'),1, embed_dim))
        #self.mesh_grid_bond_template   = torch.randn(g.num_edges('M2G'),1, embed_dim)
        self.g = g
        self.embed_dim = embed_dim

        M2Mweightorder1= g.edge_ids(self.M2M_edgeid2pair[:,0],self.M2M_edgeid2pair[:,1],etype='M2M')
        M2Mweightorder2= g.edge_ids(self.M2M_edgeid2pair[:,1],self.M2M_edgeid2pair[:,0],etype='M2M')
        mesh_mesh_bond_embedding = torch.randn(g.num_edges('M2M'),1, embed_dim)
        mesh_mesh_bond_embedding[M2Mweightorder2] = mesh_mesh_bond_embedding[M2Mweightorder2]    
        self.mesh_mesh_bond_embedding  = nn.Parameter(mesh_mesh_bond_embedding)


