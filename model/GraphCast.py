import torch.nn as nn
import numpy as np
import torch
import os
import csv
import copy

class MLP(nn.Module):
    def __init__(self, input_channel, output_cannel, bias=False, 
                 nonlinear='tanh',depth=1):
        super().__init__()
        self.linear    = torch.nn.Linear(input_channel,output_cannel,bias=bias)
        self.nonlinear = nonlinear
        self.depth     = depth
        if nonlinear == 'tanh':
            self.activator = torch.nn.Tanh()
        elif nonlinear == 'relu':
            self.activator = torch.nn.ReLU()
        elif nonlinear == 'sigmoid':
            self.activator = torch.nn.Sigmoid()
        elif nonlinear == 'swish':
            self.activator = torch.nn.SiLU()
        else:
            raise NotImplementedError
        self.norm  = torch.nn.LayerNorm(output_cannel)

    def forward(self,x):
        x = self.norm(self.activator(self.linear(x))) 
        return x

class Grid2Mesh(nn.Module):
    def __init__(self, G2M_edge_id2pair_tensor, G2M_edge_id_of_node_tensor, G2M_edge_coef_node_tensor, 
                 embed_dim=128,nonlinear='tanh',mlp_depth=1,mlp_bias = False):
        super().__init__()
        self.G2M_edge_id2pair_tensor   = G2M_edge_id2pair_tensor
        self.G2M_edge_id_of_node_tensor= G2M_edge_id_of_node_tensor
        self.G2M_edge_coef_node_tensor = G2M_edge_coef_node_tensor
        #self.register_buffer('G2M_edge_coef_node_tensor', G2M_edge_coef_node_tensor)

        self.MLP_G2M_GM2E= MLP(embed_dim*3,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)
        self.MLP_G2M_E2M = MLP(embed_dim*2,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)
        self.MLP_G2M_G2G = MLP(embed_dim  ,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)
        self.rect_index_limit = self.G2M_edge_id2pair_tensor[:,0].max() + 1
        self.node_index_limit = self.G2M_edge_id2pair_tensor[:,1].max() + 1
        self.edge_index_limit = len(self.G2M_edge_id2pair_tensor)
    def forward(self, grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding,edge_agg_fun=torch.mean):
        ### shape checking
        ### all the necessary rect of grid is recorded in G2M_grid2LaLotudePos
        #### we will plus north south point at the begining torch.cat([north_south_embedding,grid_rect_embedding],1)
        assert len(grid_rect_embedding.shape) == len(mesh_node_embedding.shape) == len(grid_mesh_bond_embedding.shape) == 3
        assert self.rect_index_limit == grid_rect_embedding.shape[1]
        assert self.node_index_limit == mesh_node_embedding.shape[1]
        assert self.edge_index_limit == grid_mesh_bond_embedding.shape[1] 
        device = self.MLP_G2M_GM2E.linear.weight.device
        
        delta_grid_mesh_bond_embedding = self.MLP_G2M_GM2E(torch.cat([grid_mesh_bond_embedding,
                                 grid_rect_embedding[:,self.G2M_edge_id2pair_tensor[:,0]],
                                 mesh_node_embedding[:,self.G2M_edge_id2pair_tensor[:,1]]],-1))

        delta_mesh_node_embedding      = self.MLP_G2M_E2M(torch.cat([mesh_node_embedding,
                edge_agg_fun(delta_grid_mesh_bond_embedding[:,self.G2M_edge_id_of_node_tensor]*self.G2M_edge_coef_node_tensor.to(device),
                                       -2)],-1))
        delta_grid_rect_embedding      = self.MLP_G2M_G2G(grid_rect_embedding)
        grid_mesh_bond_embedding      += delta_grid_mesh_bond_embedding
        grid_rect_embedding           += delta_grid_rect_embedding
        mesh_node_embedding           += delta_mesh_node_embedding
        return grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding

class Mesh2Grid(nn.Module):
    def __init__(self, M2G_edge_id2pair_tensor, M2G_edge_id_of_grid_tensor, M2G_edge_coef_grid_tensor, \
                 embed_dim=128,nonlinear='tanh',mlp_depth=1,mlp_bias = False):
        super().__init__()
        self.M2G_edge_id2pair_tensor      = M2G_edge_id2pair_tensor
        self.M2G_edge_id_of_grid_tensor   = M2G_edge_id_of_grid_tensor
        self.M2G_edge_coef_grid_tensor = M2G_edge_coef_grid_tensor
        #self.register_buffer('M2G_edge_coef_grid_tensor', M2G_edge_coef_grid_tensor)

        self.MLP_M2G_MG2E= MLP(embed_dim*3,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)
        self.MLP_M2G_E2G = MLP(embed_dim*2,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)


        self.rect_index_limit = self.M2G_edge_id2pair_tensor[:,0].max() + 1
        self.node_index_limit = self.M2G_edge_id2pair_tensor[:,1].max() + 1
        self.edge_index_limit = len(self.M2G_edge_id2pair_tensor)

    def forward(self, mesh_grid_bond_embedding,grid_allrect_embedding,mesh_node_embedding,edge_agg_fun=torch.mean):
        assert len(grid_allrect_embedding.shape) == len(mesh_node_embedding.shape) == len(mesh_grid_bond_embedding.shape) == 3
        assert self.rect_index_limit == grid_allrect_embedding.shape[1]
        assert self.node_index_limit == mesh_node_embedding.shape[1]
        assert self.edge_index_limit == mesh_grid_bond_embedding.shape[1] 
        device = self.MLP_M2G_MG2E.linear.weight.device
        
        delta_mesh_grid_bond_embedding = self.MLP_M2G_MG2E(torch.cat([mesh_grid_bond_embedding,
                                      mesh_node_embedding[:,self.M2G_edge_id2pair_tensor[:,1]],
                                   grid_allrect_embedding[:,self.M2G_edge_id2pair_tensor[:,0]]],-1))
        delta_grid_rect_embedding     = self.MLP_M2G_E2G(torch.cat([grid_allrect_embedding,
                            edge_agg_fun(delta_mesh_grid_bond_embedding[:,self.M2G_edge_id_of_grid_tensor]*self.M2G_edge_coef_grid_tensor.to(device),
                                       -2)],-1))  # notice  <-- this operation should be sum, but we use mean which cause a extra divide
        grid_allrect_embedding += delta_grid_rect_embedding
        return grid_allrect_embedding

class Mesh2Mesh(nn.Module):
    def __init__(self, M2M_edgeid2pair, key_nearbyedge_pair_per_level, num_of_linked_nodes, 
                 embed_dim=128,nonlinear='tanh',mlp_depth=1,mlp_bias = False):
        super().__init__()
        
        self.M2M_edgeid2pair = M2M_edgeid2pair
        self.key_nearbyedge_pair_per_level = key_nearbyedge_pair_per_level
        self.num_of_linked_nodes = num_of_linked_nodes
        #self.register_buffer('num_of_linked_nodes', num_of_linked_nodes)


        self.MLP_M2M_N2E= MLP(embed_dim*3,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)
        self.MLP_M2M_E2N= MLP(embed_dim*2,embed_dim,nonlinear=nonlinear,depth=mlp_depth,bias=mlp_bias)

        self.node_index_limit = self.M2M_edgeid2pair[:,1].max() + 1
        self.edge_index_limit = len(self.M2M_edgeid2pair)
        
    def forward(self, mesh_mesh_bond_embedding,mesh_node_embedding):
        assert len(mesh_node_embedding.shape) == len(mesh_mesh_bond_embedding.shape) == 3
        assert self.node_index_limit == mesh_node_embedding.shape[1]
        assert self.edge_index_limit == mesh_mesh_bond_embedding.shape[1] 
        device = self.MLP_M2M_N2E.linear.weight.device
        delta_mesh_mesh_bond_embedding = self.MLP_M2M_N2E(torch.cat([mesh_mesh_bond_embedding,
                                        mesh_node_embedding[:,self.M2M_edgeid2pair[:,0]],
                                        mesh_node_embedding[:,self.M2M_edgeid2pair[:,1]]],-1))
        delta_mesh_mesh_bond_embedding= torch.nn.functional.pad(delta_mesh_mesh_bond_embedding,(0,0,0,1))
        # notice the nearby node of each level either 5 or 6, then we use -1 as the padding number.
        mesh_node_aggregration = torch.zeros_like(mesh_node_embedding)
        for start_node, end_node_list in self.key_nearbyedge_pair_per_level:
            mesh_node_aggregration[:,start_node] += delta_mesh_mesh_bond_embedding[:,end_node_list].sum(-2)
        mesh_node_aggregration = mesh_node_aggregration/self.num_of_linked_nodes.to(device)
        
        delta_mesh_node_embedding = self.MLP_M2M_E2N(torch.cat([mesh_node_embedding,mesh_node_aggregration],-1))
        mesh_mesh_bond_embedding += delta_mesh_mesh_bond_embedding[:,:-1]
        mesh_node_embedding      += delta_mesh_node_embedding    
        return mesh_mesh_bond_embedding,mesh_node_embedding
    
class MeshCast(nn.Module):
    @staticmethod
    def generate_mesh2mesh_graph_static_file(flag):
        def readMx(path):
            pool = {}
            with open(path, 'r') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    key,val = row
                    key = eval(key.replace("{","(").replace("}",")"))
                    val = eval(val.replace("{","(").replace("}",")"))
                    pool[key]=val
            return pool
        ROOTPATH=f"GraphCastStructure/{flag}"
        if not os.path.exists(ROOTPATH):os.makedirs(ROOTPATH)
        print(f"creating Mesh static file, save in {ROOTPATH}")
        Mpoolist = [readMx(f'GraphCastStructure/M{i+1}.csv') for i in range(int(flag[-1]))]
        position2node = {}
        for pool in Mpoolist:
            for key in pool:
                if key not in position2node:
                    position2node[key] = len(position2node)
        node2position = np.zeros((len(position2node),3))
        for key,val in position2node.items():
            node2position[val]=np.array(key)
            
        torch.save(position2node,os.path.join(ROOTPATH,f"M2M_position2node.pt"   ))
        np.save(os.path.join(ROOTPATH,f"M2M_node2position.npy"   ),node2position)  


        node2nearby_por_level = []
        node_to_linked_nodes ={}
        for pool in Mpoolist:
            node2nearby = {}
            linked_nodes = []
            for pos, nearby in pool.items():
                node_id = position2node[pos]
                nearby_ids = [position2node[p] for p in nearby]
                if node_id not in node2nearby:node2nearby[node_id]=[]
                if node_id not in node_to_linked_nodes:node_to_linked_nodes[node_id] = []
                node2nearby[node_id]+=nearby_ids
                node_to_linked_nodes[node_id]+=nearby_ids
            node2nearby_por_level.append(node2nearby)
        key_nearby_pair_per_level = []
        for level, node2nearby in enumerate(node2nearby_por_level):
            max_length = max([len(t) for t in node2nearby.values()])
            key_nodes = []
            nearby_nodes_list = []
            for key_node,nearby_nodes in node2nearby.items():
                key_nodes.append(key_node)
                nearby_nodes_list.append(nearby_nodes if len(nearby_nodes)==max_length else nearby_nodes+[-1])
            key_nodes = np.array(key_nodes)
            nearby_nodes = np.array(nearby_nodes_list)
            key_nearby_pair_per_level.append([key_nodes,nearby_nodes])
            #print(f"{level}: {min(lenth_list)} - {max(lenth_list)}")

        edge2id = {}
        for key_nodes, node2nearby in key_nearby_pair_per_level:
            for key_node, nearby_nodes in zip(key_nodes, node2nearby):
                for nearby_node in nearby_nodes:
                    if key_node == -1:continue
                    if nearby_node == -1:continue
                    edge_id = [key_node,nearby_node]
                    edge_id.sort()
                    edge_id = tuple(edge_id) # unique edge number
                    if edge_id not in edge2id:
                        edge2id[edge_id] = len(edge2id)
                        
        edgeid2pair = np.zeros((len(edge2id),2),dtype='int')
        for pair, _id in edge2id.items():
            edgeid2pair[_id] = np.array(pair)

        key_nearbyedge_pair_per_level = []
        for key_nodes, node2nearby in key_nearby_pair_per_level:
            node2nearby_edge_list =[]
            for key_node, nearby_nodes in zip(key_nodes, node2nearby):
                nearby_edge_id = []
                for nearby_node in nearby_nodes:
                    if key_node == -1 or nearby_node == -1:
                        edge_id = -1
                    else:
                        edge_pair = [key_node,nearby_node]
                        edge_pair.sort()
                        edge_id = tuple(edge_pair)
                        edge_id = edge2id[edge_id]
                    nearby_edge_id.append(edge_id)
                node2nearby_edge_list.append(nearby_edge_id)
            key_nearbyedge_pair_per_level.append([key_nodes,np.array(node2nearby_edge_list)])
        np.save(os.path.join(ROOTPATH,f"M2M_edgeid2pair.npy"   ), edgeid2pair)
        torch.save(node_to_linked_nodes         ,os.path.join(ROOTPATH,f"M2M_node2lined_node.pt"   ))
        torch.save(edge2id                      ,os.path.join(ROOTPATH,f"M2M_edge2id.pt"   ))                
        torch.save(node2nearby_por_level        ,os.path.join(ROOTPATH,f"M2M_nearby_node_per_node_per_level.pt"   ))
        torch.save(key_nearbyedge_pair_per_level,os.path.join(ROOTPATH,f"M2M_nearby_edge_per_node_per_level.pt"   ))
        #torch.save(key_nearby_pair_per_level    ,os.path.join(ROOTPATH,f"M2M_pair_per_level.pt"   ))
        print("done~")

    @staticmethod
    def generate_grid2mesh_graph_static_file(flag,resolution):
        resolution_flag=f"{resolution}x{2*resolution}"
        ROOTPATH=f"GraphCastStructure/{flag}"
        if not os.path.exists(ROOTPATH):os.makedirs(ROOTPATH)
        node2position= np.load(os.path.join(ROOTPATH   ,f"M2M_node2position.npy"   ))
        position2node= torch.load(os.path.join(ROOTPATH,f"M2M_position2node.pt"   )                 )
        ROOTPATH=f"GraphCastStructure/{flag}/{resolution_flag}"
        if not os.path.exists(ROOTPATH):os.makedirs(ROOTPATH)
        print(f"creating Grid to Mesh and Mesh to Grid static file, save in {ROOTPATH}")
        theta_offset= (180/resolution/2)
        latitude   = (np.linspace(0,180,resolution+1) + theta_offset)[:resolution]
        longitude  = np.linspace(0,360,2*resolution+1)[:(2*resolution)]
        x, y           = np.meshgrid(latitude, longitude)
        LaLotude       = np.stack([y,x],-1).transpose(1,0,2)


        pos = node2position
        phi         = np.arctan2(pos[:,0],pos[:,1])
        phi[phi<0]  = 2*np.pi +  phi[phi<0]
        theta       = np.arctan2(np.sqrt(pos[:,0]**2+pos[:,1]**2),pos[:,2]) 
        phi   = phi*180/np.pi
        theta = theta*180/np.pi

        theta_index = (theta-theta_offset)//(180/resolution)
        phi_index   = phi//(180/resolution)
        rectangle_theta_idx = np.stack([theta_index,theta_index+1,theta_index+1,theta_index],axis=1)

        rectangle_phi_idx   = np.stack([phi_index,phi_index,phi_index+1,phi_index+1],axis=1)

        rectangle_theta_idx[position2node[0,0, 1]]= np.array([-1,-1,-1,-1])
        rectangle_theta_idx[position2node[0,0,-1]]= np.array([resolution,resolution,resolution,resolution])
        rectangle_phi_idx[position2node[0,0, 1]]= np.array([0,0,0,0])
        rectangle_phi_idx[position2node[0,0,-1]]= np.array([0,0,0,0])

        rectangle_theta= rectangle_theta_idx*(180/resolution)+theta_offset
        rectangle_theta[rectangle_theta_idx<0]           = 0
        rectangle_theta[rectangle_theta_idx>=resolution] = 180

        rectangle_phi   = rectangle_phi_idx*(180/resolution)
        rectangle_phi[rectangle_phi_idx>=2*resolution] = 0

        rectangle_angle_position = np.stack([rectangle_phi,rectangle_theta],-1)

        rectangle_theta=rectangle_theta/180*np.pi #(720,1440)
        rectangle_phi  =rectangle_phi  /180*np.pi #(720,1440) 

        rectangle_x = np.sin(rectangle_theta)*np.sin(rectangle_phi)
        rectangle_y = np.sin(rectangle_theta)*np.cos(rectangle_phi)
        rectangle_z = np.cos(rectangle_theta)
        rectangle_position_o = np.stack([rectangle_x,rectangle_y,rectangle_z],-1)

        LaLotudePI     = np.stack([y,x],-1).transpose(1,0,2)/180*np.pi #(720,1440)
        LaLotudeVector_o = np.stack([np.sin(LaLotudePI[...,1])*np.sin(LaLotudePI[...,0]),
                                    np.sin(LaLotudePI[...,1])*np.cos(LaLotudePI[...,0]),
                                    np.cos(LaLotudePI[...,1])],2)

        LaLotudeVector     = np.round(LaLotudeVector_o,8)
        rectangle_position = np.round(rectangle_position_o,8)
        LaLotudeVectorPool = {}
        for pos in LaLotudeVector.reshape(-1,3):  
            LaLotudeVectorPool[tuple(pos)] = len(LaLotudeVectorPool)
        print(len(LaLotudeVectorPool))
        print(len(rectangle_position))

        LaLotude2rect = {}
        rect2LaLotude = {}
        for pos in [(0,0,1),(0,0,-1)]:
            if pos not in LaLotude2rect:
                    node = len(LaLotude2rect)
                    rect2LaLotude[node]= pos
                    LaLotude2rect[pos] = node

        for pos_list in rectangle_position:
            for pos in pos_list:
                pos= tuple(pos)
                if pos not in LaLotude2rect:
                    node = len(LaLotude2rect)
                    rect2LaLotude[node]= pos
                    LaLotude2rect[pos] = node
        print(len(LaLotude2rect))
        G2M_rect_pool = copy.deepcopy(LaLotude2rect)
        # now we deal with the rest postion that won't appear in rect
        for pos in LaLotudeVectorPool.keys():
            pos= tuple(pos)
            if pos in LaLotude2rect:continue
            node = len(LaLotude2rect)
            rect2LaLotude[node]= pos
            LaLotude2rect[pos] = node

        grid2LaLotudePos = []
        for i,poss in enumerate(G2M_rect_pool.keys()):
            if (poss[-1]==1) or (poss[-1]==-1):continue
            grid2LaLotudePos.append(LaLotudeVectorPool[tuple(poss)])
        G2M_grid2LaLotudePos = np.array(grid2LaLotudePos)  
        np.save(os.path.join(ROOTPATH,f"G2M_grid2LaLotudePos.npy"   ),G2M_grid2LaLotudePos)    

        # should only be (0,0,1) and (0,0,-1)
        for key in LaLotude2rect.keys():
            if key not in LaLotudeVectorPool:
                print(key)

        # rectangle_position record which grid box the mesh node in.
        LaLotudePos2grid = []
        for i,poss in enumerate(LaLotudeVector.reshape(-1,3)):
            LaLotudePos2grid.append(LaLotude2rect[tuple(poss)])
        LaLotudePos2grid = np.array(LaLotudePos2grid)
        LaLotudePos2grid = LaLotudePos2grid.reshape((resolution),(2*resolution))  

        np.save(os.path.join(ROOTPATH,f"M2G_LaLotudeGrid2rect_tensor.npy"   ),LaLotudePos2grid)    

        G2M_edge_id2pair   = {} #np.zeros((len(node2position),4),dtype='float')
        G2M_edge_pos2_id   = {} #np.zeros((len(node2position),4),dtype='float')

        G2M_rect_of_node   = {} #np.zeros((len(node2position),4),dtype='int')
        G2M_rect_distant   = {} #np.zeros((len(node2position),4),dtype='float')
        G2M_edge_id_of_node= {} #np.zeros((len(node2position),4),dtype='float')
        G2M_edge_coef_node = {} #np.zeros((len(node2position),4),dtype='float')

        G2M_node_of_rect   = {} #np.zeros((len(node2position),4),dtype='int')
        G2M_node_distant   = {} #np.zeros((len(node2position),4),dtype='float')
        G2M_edge_id_of_rect= {} #np.zeros((len(node2position),4),dtype='float')
        G2M_edge_coef_rect = {} #np.zeros((len(node2position),4),dtype='float')


        for node_id, (node_pos,rec_poses) in enumerate(zip(node2position,rectangle_position)):
            for rec_pos in rec_poses:
                rect_id = LaLotude2rect[tuple(rec_pos)]
                distant = np.linalg.norm(node_pos - rec_pos)
                if (rect_id, node_id) not in G2M_edge_pos2_id:
                    G2M_edge_id = len(G2M_edge_pos2_id)
                    G2M_edge_pos2_id[rect_id, node_id]=G2M_edge_id
                    G2M_edge_id2pair[G2M_edge_id] = [rect_id, node_id]
                G2M_edge_id = G2M_edge_pos2_id[rect_id, node_id]
                if node_id not in G2M_edge_id_of_node:G2M_edge_id_of_node[node_id] = [] 
                if node_id not in G2M_edge_coef_node :G2M_edge_coef_node[node_id]  = []
                if node_id not in G2M_rect_of_node   :G2M_rect_of_node[node_id]    = []
                if node_id not in G2M_rect_distant   :G2M_rect_distant[node_id]    = []
                
                if rect_id not in G2M_node_of_rect   :G2M_node_of_rect[rect_id]    = [] 
                if rect_id not in G2M_node_distant   :G2M_node_distant[rect_id]    = []
                if rect_id not in G2M_edge_id_of_rect:G2M_edge_id_of_rect[rect_id] = []
                if rect_id not in G2M_edge_coef_rect :G2M_edge_coef_rect[rect_id]  = []
                    
                G2M_rect_of_node[node_id].append(rect_id)
                G2M_rect_distant[node_id].append(distant)
                G2M_edge_id_of_node[node_id].append(G2M_edge_id)
                G2M_edge_coef_node[node_id].append(distant)

                G2M_node_of_rect[rect_id].append(node_id)
                G2M_node_distant[rect_id].append(distant)
                G2M_edge_id_of_rect[rect_id].append(G2M_edge_id)
                G2M_edge_coef_rect[rect_id].append(distant)

        # build edge 
        G2M_edge_id2pair_tensor   = np.zeros((len(G2M_edge_id2pair),2),dtype='int')  #<-- save 
        for i in range(len(G2M_edge_id2pair)):
            G2M_edge_id2pair_tensor[i]    = np.array(G2M_edge_id2pair[i])
        G2M_edge_pos2_id = G2M_edge_pos2_id       
        np.save(os.path.join(ROOTPATH,f"G2M_edge_id2pair_tensor.npy"   ), G2M_edge_id2pair_tensor) 
        torch.save(G2M_edge_pos2_id    ,os.path.join(ROOTPATH,f"G2M_edge_pos2_id.pt"   ))  

        # build neighbor rect of ordered node
        G2M_rect_of_node_tensor = np.zeros((len(G2M_rect_of_node),4),dtype='int')    #<-- save  
        G2M_rect_distant_tensor = np.zeros((len(G2M_rect_of_node),4),dtype='float')  #<-- save 
        for i in range(len(G2M_rect_of_node)):
            G2M_rect_of_node_tensor[i] = np.array(G2M_rect_of_node[i])
            G2M_rect_distant_tensor[i] = np.array(G2M_rect_distant[i])
        # build neighbor egde of ordered node
        G2M_edge_id_of_node_tensor= np.zeros((len(G2M_rect_of_node),4),dtype='int')   #<-- save 
        G2M_edge_coef_node_tensor = np.zeros((len(G2M_rect_of_node),4),dtype='float') #<-- save 
        for i in range(len(G2M_rect_of_node)):
            G2M_edge_id_of_node_tensor[i] = np.array(G2M_edge_id_of_node[i] )
            G2M_edge_coef_node_tensor[i]  = np.array(G2M_edge_coef_node[i]  )
        np.save(os.path.join(ROOTPATH,f"G2M_rect_of_node_tensor.npy"   ), G2M_rect_of_node_tensor)    
        np.save(os.path.join(ROOTPATH,f"G2M_rect_distant_tensor.npy"   ), G2M_rect_distant_tensor)    
        np.save(os.path.join(ROOTPATH,f"G2M_edge_id_of_node_tensor.npy"), G2M_edge_id_of_node_tensor)    
        np.save(os.path.join(ROOTPATH,f"G2M_edge_coef_node_tensor.npy" ), G2M_edge_coef_node_tensor) 
                
        M2G_node_of_rect   = {} #np.zeros((len(node2position),4),dtype='int')
        M2G_node_distant   = {} #np.zeros((len(node2position),4),dtype='float')
        M2G_edge_id2pair   = G2M_edge_id2pair #np.zeros((len(node2position),4),dtype='float')
        M2G_edge_pos2_id   = G2M_edge_pos2_id #np.zeros((len(node2position),4),dtype='float')
        M2G_edge_id_of_rect= {} #np.zeros((len(node2position),4),dtype='float')
        M2G_edge_coef_rect = {} #np.zeros((len(node2position),4),dtype='float')

        # firstly we 
        for node_id, (node_pos,rec_poses) in enumerate(zip(node2position,rectangle_position)):
            for rec_pos in rec_poses:
                rect_id = LaLotude2rect[tuple(rec_pos)]
                distant = np.linalg.norm(node_pos - rec_pos)
                edgepair= (rect_id,node_id)
                assert edgepair in M2G_edge_pos2_id
                # if  edgepair not in M2G_edge_pos2_id:
                #     M2G_edge_id = len(M2G_edge_pos2_id)
                #     M2G_edge_pos2_id[edgepair]    = M2G_edge_id
                #     M2G_edge_id2pair[M2G_edge_id] = edgepair
                M2G_edge_id = M2G_edge_pos2_id[edgepair]
                if rect_id not in M2G_edge_id_of_rect:M2G_edge_id_of_rect[rect_id] = [] 
                if rect_id not in M2G_edge_coef_rect : M2G_edge_coef_rect[rect_id] = []
                if rect_id not in M2G_node_of_rect   :   M2G_node_of_rect[rect_id] = []
                if rect_id not in M2G_node_distant   :   M2G_node_distant[rect_id] = []
                M2G_node_of_rect[   rect_id].append(node_id)
                M2G_node_distant[   rect_id].append(distant)
                M2G_edge_id_of_rect[rect_id].append(M2G_edge_id)
                M2G_edge_coef_rect[ rect_id].append(distant)
        # now we deal with grid that won't appear in rectangle_position:

        for pos in LaLotudeVectorPool.keys():
            pos= tuple(pos)
            if pos in LaLotude2rect:continue
            node = len(LaLotude2rect)
            rect2LaLotude[node]= pos
            LaLotude2rect[pos] = node

        max_rank = 8
        for rec_pos in LaLotudeVectorPool.keys():
            rec_pos    = tuple(rec_pos)
            if rec_pos in G2M_rect_pool:continue
            rect_id    = LaLotude2rect[rec_pos]
            allnodes   = node2position
            alldists   = distant = np.linalg.norm(node2position - np.array([rec_pos]),axis=1)
            near_node  = np.argsort(alldists)[:max_rank]
            near_dist  = alldists[near_node]
            
            for node_id,distant in zip(near_node,near_dist):
                edgepair= (rect_id,node_id)
                if  edgepair not in M2G_edge_pos2_id:
                    M2G_edge_id = len(M2G_edge_pos2_id)
                    M2G_edge_pos2_id[edgepair]    = M2G_edge_id
                    M2G_edge_id2pair[M2G_edge_id] = edgepair
                if rect_id not in M2G_edge_id_of_rect: M2G_edge_id_of_rect[rect_id] = [] 
                if rect_id not in M2G_edge_coef_rect : M2G_edge_coef_rect[rect_id] = []
                if rect_id not in M2G_node_of_rect   :   M2G_node_of_rect[rect_id] = []
                if rect_id not in M2G_node_distant   :   M2G_node_distant[rect_id] = []
                M2G_edge_id = M2G_edge_pos2_id[edgepair]        
                M2G_node_of_rect[   rect_id].append(node_id)
                M2G_node_distant[   rect_id].append(distant)
                M2G_edge_id_of_rect[rect_id].append(M2G_edge_id)
                M2G_edge_coef_rect[ rect_id].append(distant)
        M2G_edge_id2pair_tensor = np.zeros((len(M2G_edge_id2pair),2),dtype='int')   #<-- save 
        for i in range(len(M2G_edge_id2pair)):
            M2G_edge_id2pair_tensor[i] = M2G_edge_id2pair[i]
        np.save(os.path.join(ROOTPATH,f"M2G_edge_id2pair_tensor.npy")      , M2G_edge_id2pair_tensor)

        # build neighbor egde of ordered node
        # this will only use the nearest `max_rank` nodes.
        M2G_edge_id_of_grid_tensor= np.zeros((len(M2G_node_of_rect),max_rank),dtype='int')   #<-- save 
        M2G_edge_coef_grid_tensor = np.zeros((len(M2G_node_of_rect),max_rank),dtype='float') #<-- save 
        for rect_id in range(len(M2G_node_distant)):
            nearby_node = M2G_edge_id_of_rect[rect_id]
            nearby_dist =  M2G_edge_coef_rect[rect_id]
            order    = np.argsort(nearby_dist)[:max_rank]
            nearby_node = [nearby_node[idx] for idx in order]
            nearby_dist = [nearby_dist[idx] for idx in order]
            if len(nearby_node)< max_rank:
                nearby_node = np.pad(nearby_node,(0,max_rank - len(nearby_node)),constant_values=-1)
                nearby_dist = np.pad(nearby_dist,(0,max_rank - len(nearby_dist)),constant_values=-100)
            M2G_edge_id_of_grid_tensor[rect_id]= np.array(nearby_node)
            M2G_edge_coef_grid_tensor[rect_id] = np.array(nearby_dist)

        np.save(os.path.join(ROOTPATH,f"M2G_edge_id_of_grid_tensor.npy"   ), M2G_edge_id_of_grid_tensor)    
        np.save(os.path.join(ROOTPATH,f"M2G_edge_coef_grid_tensor.npy"   ) , M2G_edge_coef_grid_tensor)    

        # above create a big graph contain all bond between mesh and grid
        # some bond would not be used in M2G but still saved and will be calculated at self-implement model
        ####################################################################################################################
        # we can also substract the M2G subgraph under the max_connected = max_rank = 8 for example.
        M2G_edge2id_max_rank = {}
        M2G_id2edge_max_rank = []
        M2G_node_of_rect_tensor = np.zeros((len(M2G_node_distant),max_rank),dtype='int')
        M2G_node_distant_tensor = np.zeros((len(M2G_node_distant),max_rank),dtype='float')
        for rect_id in range(len(M2G_node_distant)):
            nearby_node = M2G_node_of_rect[rect_id]
            nearby_dist = M2G_node_distant[rect_id]
            order    = np.argsort(nearby_dist)[:max_rank]
            nearby_node = [nearby_node[idx] for idx in order]
            nearby_dist = [nearby_dist[idx] for idx in order]
            if len(nearby_node)<max_rank:
                nearby_node = np.pad(nearby_node,(0,max_rank - len(nearby_node)),constant_values=-1)
                nearby_dist = np.pad(nearby_dist,(0,max_rank - len(nearby_dist)),constant_values=-100)
            for node_id in nearby_node:
                if node_id <0:continue
                saved_edge_relation = (rect_id,node_id)
                if saved_edge_relation not in M2G_edge2id_max_rank:
                    M2G_edge2id_max_rank[saved_edge_relation] = len(M2G_id2edge_max_rank)
                    M2G_id2edge_max_rank.append(saved_edge_relation)
            M2G_node_of_rect_tensor[rect_id]= np.array(nearby_node)
            M2G_node_distant_tensor[rect_id]= np.array(nearby_dist)

        np.save(os.path.join(ROOTPATH,f"M2G_id2edge_max_rank.npy"   ), M2G_id2edge_max_rank)    
        np.save(os.path.join(ROOTPATH,f"M2G_node_of_rect_tensor.npy"   ),M2G_node_of_rect_tensor)    
        np.save(os.path.join(ROOTPATH,f"M2G_node_distant_tensor.npy"   ),M2G_node_distant_tensor)
        
        

        
           
        print("done~")

class GraphCast(MeshCast):
    '''
    Repreduce of GraphCast in Pytorch.
    GraphCast has three part:
    - Grid to Mesh
    - Mesh to Mesh
    - Mesh to Grid
    -------------------------------------
    the input is a tensor (B, P, W, H), but the internal tensor all with shape (B, L ,P)
    where the L equal the node number or edge number.
    '''
    def __init__(self, img_size=(64,128),  in_chans=70, out_chans=70, depth=6, embed_dim=128, graphflag='mesh5', nonlinear='swish', **kargs):
        super().__init__()
        flag = graphflag
        resolution_flag=f'{img_size[0]}x{img_size[1]}'
        ROOTPATH=f"GraphCastStructure/{flag}"
        if not os.path.exists(ROOTPATH):
            self.generate_mesh2mesh_graph_static_file(flag)
        M2M_node2position                 = np.load(os.path.join(ROOTPATH   ,f"M2M_node2position.npy"   ))
        M2M_position2node                 = torch.load(os.path.join(ROOTPATH,f"M2M_position2node.pt"   )                 )
        M2M_node2lined_node               = torch.load(os.path.join(ROOTPATH,f"M2M_node2lined_node.pt"   )               )
        # M2M_edge2id                       = torch.load(os.path.join(ROOTPATH,f"M2M_edge2id.pt"   )                       )

        M2M_edgeid2pair                   = np.load(os.path.join(ROOTPATH   ,f"M2M_edgeid2pair.npy"   ))
        M2M_nearby_edge_per_node_per_level= torch.load(os.path.join(ROOTPATH,f"M2M_nearby_edge_per_node_per_level.pt"   ))
        #M2M_nearby_node_per_node_per_level= torch.load(os.path.join(ROOTPATH,f"M2M_nearby_node_per_node_per_level.pt"   ))

        ROOTPATH=f"GraphCastStructure/{flag}/{resolution_flag}"
        if not os.path.exists(ROOTPATH):self.generate_grid2mesh_graph_static_file(flag,img_size[0])
        G2M_grid2LaLotudePos = np.load(os.path.join(ROOTPATH,f"G2M_grid2LaLotudePos.npy"   )     )
        M2G_LaLotudePos2grid = np.load(os.path.join(ROOTPATH,f"M2G_LaLotudeGrid2rect_tensor.npy"))

        # G2M_rect_of_node_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_rect_of_node_tensor.npy"   ) )
        # G2M_rect_distant_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_rect_distant_tensor.npy"   ) )
        G2M_edge_id_of_node_tensor = np.load(os.path.join(ROOTPATH,f"G2M_edge_id_of_node_tensor.npy") )
        G2M_edge_coef_node_tensor  = np.load(os.path.join(ROOTPATH,f"G2M_edge_coef_node_tensor.npy" ) )
        G2M_edge_id2pair_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_edge_id2pair_tensor.npy"   ) )
        # G2M_edge_pos2_id           = torch.load(os.path.join(ROOTPATH,f"G2M_edge_pos2_id.pt"   ))
        # M2G_node_of_rect_tensor    = np.load(os.path.join(ROOTPATH,f"M2G_node_of_rect_tensor.npy"   ))    
        # M2G_node_distant_tensor    = np.load(os.path.join(ROOTPATH,f"M2G_node_distant_tensor.npy"   ))    
        M2G_edge_id_of_grid_tensor= np.load(os.path.join(ROOTPATH,f"M2G_edge_id_of_grid_tensor.npy"   ))  
        M2G_edge_coef_grid_tensor = np.load(os.path.join(ROOTPATH,f"M2G_edge_coef_grid_tensor.npy"   ) )  
        M2G_edge_id2pair_tensor   = np.load(os.path.join(ROOTPATH,f"M2G_edge_id2pair_tensor.npy")      )  
        
        embedding_dim = embed_dim
        self.num_unactivated_grid = num_unactivated_grid = len(M2G_edge_id_of_grid_tensor) - len(G2M_grid2LaLotudePos) - 2
        self.num_unactivated_edge = num_unactivated_edge = len(M2G_edge_id2pair_tensor) - len(G2M_edge_id2pair_tensor)
        print(f'''
        This is ===> GraphCast Model <===
        Information: 
            total mesh node:{len(M2M_node2position):5d} total unique mesh edge:{len(M2M_edgeid2pair):5d} 
            total grid node {np.prod(img_size)}+2 = {(len(M2G_edge_id_of_grid_tensor))} but activated grid {len(G2M_grid2LaLotudePos):5d} +  2
            from activated grid to mesh, create 4*{len(M2M_node2position)} = {len(G2M_edge_id2pair_tensor)} edge
            there are {num_unactivated_grid} unactivated grid node
            when mapping node to grid, 
            from node to activated grid, there are {len(G2M_edge_id2pair_tensor)} 
            from node to unactivated grid, there are {num_unactivated_edge} edge
            thus, totally have {len(M2G_edge_id2pair_tensor)} edge. 
            #notice some grid only have 1-2 linked node but some grid may have 30 lined node
        ''')
        G2M_edge_id2pair_tensor   = torch.LongTensor(G2M_edge_id2pair_tensor)
        G2M_edge_coef_node_tensor = torch.Tensor(G2M_edge_coef_node_tensor).softmax(-1).unsqueeze(-1)
        

        M2M_edgeid2pair                    = torch.LongTensor(M2M_edgeid2pair)
        M2M_nearby_edge_per_node_per_level = [[torch.LongTensor(start_node),torch.LongTensor(linked_edge_list)] for start_node,linked_edge_list in M2M_nearby_edge_per_node_per_level]
        M2M_num_of_linked_nodes            = torch.FloatTensor([len(M2M_node2lined_node[t]) for t in range(len(M2M_node2lined_node))]).unsqueeze(-1)
        
        M2G_edge_id2pair_tensor  = torch.LongTensor(M2G_edge_id2pair_tensor)
        M2G_node_of_rect_tensor  = torch.LongTensor(M2G_edge_id_of_grid_tensor)
        M2G_edge_coef_grid_tensor= torch.Tensor(M2G_edge_coef_grid_tensor).softmax(-1).unsqueeze(-1)


        M2G_LaLotudePos2grid = torch.LongTensor(M2G_LaLotudePos2grid)
        G2M_grid2LaLotudePos = torch.LongTensor(G2M_grid2LaLotudePos)
        

        

        self.M2G_LaLotudePos2grid = M2G_LaLotudePos2grid
        self.G2M_grid2LaLotudePos = G2M_grid2LaLotudePos
        
        self.layer_grid2mesh = Grid2Mesh(G2M_edge_id2pair_tensor,G2M_edge_id_of_node_tensor,G2M_edge_coef_node_tensor, 
                                         embed_dim=embedding_dim,nonlinear=nonlinear)
        
        self.layer_mesh2mesh = nn.ModuleList()
        for i in range(depth):
            self.layer_mesh2mesh.append(Mesh2Mesh(M2M_edgeid2pair, M2M_nearby_edge_per_node_per_level, M2M_num_of_linked_nodes, 
                                                  embed_dim=embedding_dim,nonlinear=nonlinear))
        self.layer_mesh2grid = Mesh2Grid(M2G_edge_id2pair_tensor,M2G_node_of_rect_tensor,M2G_edge_coef_grid_tensor, 
                                         embed_dim=embedding_dim,nonlinear=nonlinear)
        
        self.grid_rect_embedding_layer = nn.Linear(in_chans,embedding_dim)
        self.northsouthembbed = nn.Parameter(torch.randn(2,embedding_dim))
        
        self.mesh_node_embedding       = nn.Parameter(torch.randn(len(M2M_node2position),embedding_dim))
        self.mesh_mesh_bond_embedding  = nn.Parameter(torch.randn(len(M2M_edgeid2pair),embedding_dim))
        self.grid_mesh_bond_embedding  = nn.Parameter(torch.randn(len(G2M_edge_id2pair_tensor),embedding_dim))

        self.projection      = nn.Linear(embedding_dim,out_chans)

    def forward(self, _input):
        B, P , W, H =_input.shape
        feature_along_latlot     = self.grid_rect_embedding_layer(_input.permute(0,2,3,1).flatten(1,2))
        grid_rect_embedding      = feature_along_latlot[:,self.G2M_grid2LaLotudePos]
        grid_rect_embedding      = torch.cat([self.northsouthembbed.repeat(B,1,1),grid_rect_embedding],1)
        grid_mesh_bond_embedding = self.grid_mesh_bond_embedding.repeat(B,1,1)
        mesh_node_embedding      = self.mesh_node_embedding.repeat(B,1,1)
        mesh_mesh_bond_embedding = self.mesh_mesh_bond_embedding.repeat(B,1,1)
        grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding = self.layer_grid2mesh(
                                        grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding)
        for mesh2mesh in self.layer_mesh2mesh:
            mesh_mesh_bond_embedding, mesh_node_embedding  = mesh2mesh(mesh_mesh_bond_embedding, mesh_node_embedding)
        grid_mesh_bond_embedding = torch.nn.functional.pad(grid_mesh_bond_embedding,(0,0,0,self.num_unactivated_edge))
        grid_rect_embedding      = torch.nn.functional.pad(grid_rect_embedding,(0,0,0,self.num_unactivated_grid ))
        grid_rect_embedding      = self.layer_mesh2grid(grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding)
        grid_rect_embedding      = grid_rect_embedding[:,self.M2G_LaLotudePos2grid] #(B,64,128,embed_dim)
        return self.projection(grid_rect_embedding).permute(0,3,1,2)

import math
def fastinit1(embed_dim1,embed_dim2):
    # for tensor (a,b) `fan_in` deal with b `fan_out` deal with a
    # a linear weight is (out_dim, in_dim) and use `fan_in`
    # but here we direct do matrix matmul, thus fan_out
    return torch.nn.init.kaiming_uniform_(torch.randn(embed_dim1, embed_dim2), mode='fan_out', a=math.sqrt(5)) 

class Node2Edge2NodeBlock(nn.Module):
    def __init__(self,embed_dim=128, do_source_update = False,**kargs):
        super().__init__()
        initial_weight = fastinit1(3*embed_dim, embed_dim)
        STE2E_S2E,STE2E_T2E,STE2E_E2E = torch.split(initial_weight,embed_dim)
        self.STE2E_S2E = nn.Parameter(STE2E_S2E)
        self.STE2E_T2E = nn.Parameter(STE2E_T2E)
        self.STE2E_E2E = nn.Parameter(STE2E_E2E)
        self.activator1 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))
        
        initial_weight = fastinit1(2*embed_dim, embed_dim)
        ET2T_E2T,ET2T_T2T = torch.split(initial_weight,embed_dim)
        self.ET2T_E2T  = nn.Parameter(ET2T_E2T)
        self.ET2T_T2T  = nn.Parameter(ET2T_T2T)
        self.activator2 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))
        
        self.S2S = None
        if do_source_update:
            self.S2S     = nn.Parameter(fastinit1(embed_dim, embed_dim))
            self.activator3 = nn.Sequential(torch.nn.SiLU(),torch.nn.LayerNorm(embed_dim))

class Node2Edge2NodeBlockSingleLevel(Node2Edge2NodeBlock):
    def __init__(self, src_tgt_order, edge_order, bond_coef, 
                 embed_dim=128,do_source_update = False, agg_way='mean',
                 **kargs):
        super().__init__(embed_dim=embed_dim,do_source_update = do_source_update)
        self.src_order = src_tgt_order[:,0]
        self.tgt_order = src_tgt_order[:,1]
        self.edge_order= edge_order
        #self.bond_coef = bond_coef
        #self.register_buffer('G2M_edge_coef_node_tensor', G2M_edge_coef_node_tensor)
        self.agg_way  = agg_way
        self.bond_coef  = bond_coef
        # do not use nn.Parameter(bond_coef,requires_grad=False), since we may freeze or unfreeze model
        # or use regist('bond_coef',bond_coef) but will bug in multiGPU mode
        
        self.src_index_limit  = self.src_order.max() + 1
        self.tgt_index_limit  = self.tgt_order.max() + 1
        self.edge_index_limit = len(self.src_order)
    
    def node2bond(self,bond_embedding, src_embedding, tgt_embedding,activator):
        ## compute delta bond embedding
        delta_bond_embedding = (src_embedding @ self.STE2E_S2E)[:,self.src_order] #(B,L,D)
        delta_bond_embedding = delta_bond_embedding + (tgt_embedding @ self.STE2E_T2E)[:,self.tgt_order] #(B,L,D)
        delta_bond_embedding = delta_bond_embedding + bond_embedding @ self.STE2E_E2E  #(B,1,D)
        delta_bond_embedding = activator(delta_bond_embedding) #(B,L,D)
        return delta_bond_embedding
    
    def bond2node(self,bond_embedding, src_embedding, tgt_embedding,activator):
        device = self.STE2E_S2E.device
        self.bond_coef       = self.bond_coef.to(device)
        bond_reduce = bond_embedding
        if self.agg_way == 'mean':
            bond_reduce = (bond_reduce[:,self.edge_order]*self.bond_coef.unsqueeze(-1)).mean(-2)
        elif self.agg_way == 'sum':
            bond_reduce = (bond_reduce[:,self.edge_order]*self.bond_coef.unsqueeze(-1)).sum(-2)
        else:
            raise NotImplementedError
        # for some one try to realize via pyG, the only difference needed is use 
        # torch_scatter.scatter(delta_bond_embedding, layer.tgt_order, dim=1, reduce='sum') <<< this is slower than our implement. in inference at leat.
        delta_tgt_embedding = bond_reduce@self.ET2T_E2T + tgt_embedding@self.ET2T_T2T
        delta_tgt_embedding = activator(delta_tgt_embedding)
        return delta_tgt_embedding
    
    def forward(self, bond_embedding, src_embedding, tgt_embedding):
        ### shape checking
        ### all the necessary rect of grid is recorded in G2M_grid2LaLotudePos
        #### we will plus north south point at the begining torch.cat([north_south_embedding,grid_rect_embedding],1)
        assert len(bond_embedding.shape) == len(src_embedding.shape) == len(tgt_embedding.shape) == 3
        assert self.src_index_limit  ==  src_embedding.shape[1]
        assert self.tgt_index_limit  ==  tgt_embedding.shape[1]
        assert self.edge_index_limit == bond_embedding.shape[1] 
        device = self.STE2E_S2E.device
        self.bond_coef = self.bond_coef.to(device)
        delta_bond_embedding = self.node2bond(bond_embedding, src_embedding, tgt_embedding, self.activator1)
        delta_tgt_embedding  = self.bond2node(delta_bond_embedding, src_embedding, tgt_embedding, self.activator2)
        delta_src_embedding  = self.activator3(src_embedding@self.S2S) if self.S2S is not None else 0
        bond_embedding= bond_embedding  + delta_bond_embedding
        src_embedding = src_embedding   + delta_src_embedding
        tgt_embedding = tgt_embedding   + delta_tgt_embedding
        
        return bond_embedding,src_embedding,tgt_embedding

class Node2Edge2NodeBlockMultiLevel(Node2Edge2NodeBlockSingleLevel):
    def bond2node(self,bond_embedding, src_embedding, tgt_embedding,activator):
        device = self.STE2E_S2E.device
        self.bond_coef    = self.bond_coef.to(device)
        delta_bond_embedding = bond_embedding
        delta_bond_embedding = torch.nn.functional.pad(delta_bond_embedding,(0,0,0,1))
        # notice the nearby node of each level either 5 or 6, then we use -1 as the padding number.
        # notice some node has multi-level nearby,
        bond_reduce = torch.zeros_like(tgt_embedding)
        for start_node, end_node_list in self.edge_order:
            bond_reduce[:,start_node] += delta_bond_embedding[:,end_node_list].sum(-2)
        bond_reduce = bond_reduce/self.bond_coef
        delta_tgt_embedding = bond_reduce@self.ET2T_E2T + tgt_embedding@self.ET2T_T2T
        delta_tgt_embedding = activator(delta_tgt_embedding)
        return delta_tgt_embedding
    
    def forward(self, bond_embedding, src_embedding, tgt_embedding):
        ### shape checking
        ### all the necessary rect of grid is recorded in G2M_grid2LaLotudePos
        #### we will plus north south point at the begining torch.cat([north_south_embedding,grid_rect_embedding],1)
        assert len(bond_embedding.shape) == len(src_embedding.shape) == len(tgt_embedding.shape) == 3
        delta_bond_embedding = self.node2bond(bond_embedding, src_embedding, tgt_embedding, self.activator1)
        delta_tgt_embedding = self.bond2node(delta_bond_embedding, src_embedding, tgt_embedding, self.activator2)
        bond_embedding = bond_embedding + delta_bond_embedding
        tgt_embedding  = tgt_embedding  + delta_tgt_embedding
        return bond_embedding,None,tgt_embedding

class GraphCastFast(MeshCast):
    '''
    ---> 1.5 speed up
    Repreduce of GraphCast in Pytorch.
    GraphCast has three part:
    - Grid to Mesh
    - Mesh to Mesh
    - Mesh to Grid
    -------------------------------------
    the input is a tensor (B, P, W, H), but the internal tensor all with shape (B, L ,P)
    where the L equal the node number or edge number.
    '''
    def __init__(self, img_size=(64,128),  in_chans=70, out_chans=70, depth=6, embed_dim=128, graphflag='mesh5', agg_way='mean', nonlinear='swish', **kargs):
        super().__init__()
        flag = graphflag
        resolution_flag=f'{img_size[0]}x{img_size[1]}'
        ROOTPATH=f"GraphCastStructure/{flag}"
        if not os.path.exists(ROOTPATH):
            self.generate_mesh2mesh_graph_static_file(flag)
        M2M_node2position                 = np.load(os.path.join(ROOTPATH   ,f"M2M_node2position.npy"   ))
        M2M_position2node                 = torch.load(os.path.join(ROOTPATH,f"M2M_position2node.pt"   )                 )
        M2M_node2lined_node               = torch.load(os.path.join(ROOTPATH,f"M2M_node2lined_node.pt"   )               )
        # M2M_edge2id                       = torch.load(os.path.join(ROOTPATH,f"M2M_edge2id.pt"   )                       )

        M2M_edgeid2pair                   = np.load(os.path.join(ROOTPATH   ,f"M2M_edgeid2pair.npy"   ))
        M2M_nearby_edge_per_node_per_level= torch.load(os.path.join(ROOTPATH,f"M2M_nearby_edge_per_node_per_level.pt"   ))
        #M2M_nearby_node_per_node_per_level= torch.load(os.path.join(ROOTPATH,f"M2M_nearby_node_per_node_per_level.pt"   ))

        ROOTPATH=f"GraphCastStructure/{flag}/{resolution_flag}"
        if not os.path.exists(ROOTPATH):self.generate_grid2mesh_graph_static_file(flag,img_size[0])
        G2M_grid2LaLotudePos = np.load(os.path.join(ROOTPATH,f"G2M_grid2LaLotudePos.npy"   )     )
        M2G_LaLotudePos2grid = np.load(os.path.join(ROOTPATH,f"M2G_LaLotudeGrid2rect_tensor.npy"))

        # G2M_rect_of_node_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_rect_of_node_tensor.npy"   ) )
        # G2M_rect_distant_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_rect_distant_tensor.npy"   ) )
        G2M_edge_id_of_node_tensor = np.load(os.path.join(ROOTPATH,f"G2M_edge_id_of_node_tensor.npy") )
        G2M_edge_coef_node_tensor  = np.load(os.path.join(ROOTPATH,f"G2M_edge_coef_node_tensor.npy" ) )
        G2M_edge_id2pair_tensor    = np.load(os.path.join(ROOTPATH,f"G2M_edge_id2pair_tensor.npy"   ) )
        # G2M_edge_pos2_id           = torch.load(os.path.join(ROOTPATH,f"G2M_edge_pos2_id.pt"   ))
        # M2G_node_of_rect_tensor    = np.load(os.path.join(ROOTPATH,f"M2G_node_of_rect_tensor.npy"   ))    
        # M2G_node_distant_tensor    = np.load(os.path.join(ROOTPATH,f"M2G_node_distant_tensor.npy"   ))    
        M2G_edge_id_of_grid_tensor= np.load(os.path.join(ROOTPATH,f"M2G_edge_id_of_grid_tensor.npy"   ))  
        M2G_edge_coef_grid_tensor = np.load(os.path.join(ROOTPATH,f"M2G_edge_coef_grid_tensor.npy"   ) )  
        M2G_edge_id2pair_tensor   = np.load(os.path.join(ROOTPATH,f"M2G_edge_id2pair_tensor.npy")      )  
        
        embedding_dim = embed_dim
        self.num_unactivated_grid = num_unactivated_grid = len(M2G_edge_id_of_grid_tensor) - len(G2M_grid2LaLotudePos) - 2
        self.num_unactivated_edge = num_unactivated_edge = len(M2G_edge_id2pair_tensor) - len(G2M_edge_id2pair_tensor)
        print(f'''
        This is ===> GraphCast Model (Fast) <===
        Information: 
            total mesh node:{len(M2M_node2position):5d} total unique mesh edge:{len(M2M_edgeid2pair):5d} 
            total grid node {np.prod(img_size)}+2 = {(len(M2G_edge_id_of_grid_tensor))} but activated grid {len(G2M_grid2LaLotudePos):5d} +  2
            from activated grid to mesh, create 4*{len(M2M_node2position)} = {len(G2M_edge_id2pair_tensor)} edge
            there are {num_unactivated_grid} unactivated grid node
            when mapping node to grid, 
            from node to activated grid, there are {len(G2M_edge_id2pair_tensor)} 
            from node to unactivated grid, there are {num_unactivated_edge} edge
            thus, totally have {len(M2G_edge_id2pair_tensor)} edge. 
            #notice some grid only have 1-2 linked node but some grid may have 30 lined node
        ''')
        G2M_edge_id2pair_tensor   = torch.LongTensor(G2M_edge_id2pair_tensor)
        G2M_edge_coef_node_tensor = torch.Tensor(G2M_edge_coef_node_tensor).softmax(-1)
        

        M2M_edgeid2pair                    = torch.LongTensor(M2M_edgeid2pair)
        M2M_nearby_edge_per_node_per_level = [[torch.LongTensor(start_node),torch.LongTensor(linked_edge_list)] for start_node,linked_edge_list in M2M_nearby_edge_per_node_per_level]
        M2M_num_of_linked_nodes            = torch.FloatTensor([len(M2M_node2lined_node[t]) for t in range(len(M2M_node2lined_node))]).unsqueeze(-1)
        
        M2G_edge_id2pair_tensor  = torch.LongTensor(M2G_edge_id2pair_tensor) # (L,2)
        #notice the M2G_edge_id2pair_tensor record (rect_id , node_id) but in this implement
        # the order should be (src,tgt) which is node_id,rect_id
        # lets convert it . 
        M2G_edge_id2pair_tensor  = torch.stack([M2G_edge_id2pair_tensor[:,1],M2G_edge_id2pair_tensor[:,0]],-1)
        
        M2G_edge_of_rect_tensor  = torch.LongTensor(M2G_edge_id_of_grid_tensor)
        M2G_edge_coef_grid_tensor= torch.Tensor(M2G_edge_coef_grid_tensor).softmax(-1)


        M2G_LaLotudePos2grid = torch.LongTensor(M2G_LaLotudePos2grid)
        G2M_grid2LaLotudePos = torch.LongTensor(G2M_grid2LaLotudePos)
        

        

        self.M2G_LaLotudePos2grid = M2G_LaLotudePos2grid
        self.G2M_grid2LaLotudePos = G2M_grid2LaLotudePos
        
        self.layer_grid2mesh = Node2Edge2NodeBlockSingleLevel(G2M_edge_id2pair_tensor,G2M_edge_id_of_node_tensor,G2M_edge_coef_node_tensor, 
                                    embed_dim=embedding_dim,do_source_update=True,agg_way=agg_way)
        
        self.layer_mesh2mesh = nn.ModuleList()
        for i in range(depth):
            self.layer_mesh2mesh.append(Node2Edge2NodeBlockMultiLevel(M2M_edgeid2pair, M2M_nearby_edge_per_node_per_level, M2M_num_of_linked_nodes, 
                                                  embed_dim=embedding_dim))
        
        self.layer_mesh2grid = Node2Edge2NodeBlockSingleLevel(M2G_edge_id2pair_tensor,M2G_edge_of_rect_tensor,M2G_edge_coef_grid_tensor, 
                                         embed_dim=embedding_dim,agg_way=agg_way)
        
        self.grid_rect_embedding_layer = nn.Linear(in_chans,embedding_dim)
        self.northsouthembbed = nn.Parameter(torch.randn(2,embedding_dim))
        
        self.mesh_node_embedding       = nn.Parameter(torch.randn(1,len(M2M_node2position)      ,embedding_dim))
        self.mesh_mesh_bond_embedding  = nn.Parameter(torch.randn(1,len(M2M_edgeid2pair)        ,embedding_dim))
        self.grid_mesh_bond_embedding  = nn.Parameter(torch.randn(1,len(G2M_edge_id2pair_tensor),embedding_dim))

        self.projection      = nn.Linear(embedding_dim,out_chans)

    def forward(self, _input):
        B, P , W, H =_input.shape
        feature_along_latlot     = self.grid_rect_embedding_layer(_input.permute(0,2,3,1).flatten(1,2))
        grid_rect_embedding      = feature_along_latlot[:,self.G2M_grid2LaLotudePos]
        grid_rect_embedding      = torch.cat([self.northsouthembbed.repeat(B,1,1),grid_rect_embedding],1)
        grid_mesh_bond_embedding = self.grid_mesh_bond_embedding
        mesh_node_embedding      = self.mesh_node_embedding
        mesh_mesh_bond_embedding = self.mesh_mesh_bond_embedding
        
        grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding = self.layer_grid2mesh(
                                        grid_mesh_bond_embedding,grid_rect_embedding,mesh_node_embedding)
        for mesh2mesh in self.layer_mesh2mesh:
            mesh_mesh_bond_embedding, _, mesh_node_embedding  = mesh2mesh(mesh_mesh_bond_embedding, mesh_node_embedding, mesh_node_embedding)
        grid_mesh_bond_embedding = torch.nn.functional.pad(grid_mesh_bond_embedding,(0,0,0,self.num_unactivated_edge))
        grid_rect_embedding      = torch.nn.functional.pad(grid_rect_embedding,(0,0,0,self.num_unactivated_grid ))
        grid_mesh_bond_embedding,mesh_node_embedding,grid_rect_embedding      = self.layer_mesh2grid(grid_mesh_bond_embedding,mesh_node_embedding,grid_rect_embedding)
        grid_rect_embedding      = grid_rect_embedding[:,self.M2G_LaLotudePos2grid] #(B,64,128,embed_dim)
        return self.projection(grid_rect_embedding).permute(0,3,1,2)

try:
        
    import dgl
    import dgl.function as fn
    from einops import rearrange
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
            self.mesh_grid_bond_template   = torch.randn(g.num_edges('M2G'),1, embed_dim)
            self.g = g
            self.embed_dim = embed_dim

            M2Mweightorder1= g.edge_ids(self.M2M_edgeid2pair[:,0],self.M2M_edgeid2pair[:,1],etype='M2M')
            M2Mweightorder2= g.edge_ids(self.M2M_edgeid2pair[:,1],self.M2M_edgeid2pair[:,0],etype='M2M')
            mesh_mesh_bond_embedding = torch.randn(g.num_edges('M2M'),1, embed_dim)
            mesh_mesh_bond_embedding[M2Mweightorder2] = mesh_mesh_bond_embedding[M2Mweightorder2]    
            self.mesh_mesh_bond_embedding  = nn.Parameter(mesh_mesh_bond_embedding)

except:
    pass

