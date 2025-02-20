import json
import numpy as np
import os
import os.path as osp
import scipy.sparse as sp
import torch
import utils


MIT = utils.MIT
CMU = utils.CMU
Stanford = utils.Stanford
UCB = utils.UCB

# P here should be PT to calculate the equivalent ppr column vector
def calc_ppr_by_power_iteration_gpu(P: torch.Tensor, alpha: float, h: torch.Tensor, t: int) -> np.ndarray:
    alpha = 1-alpha
    alpha = 2*alpha/(1+alpha)
    iterated = (1 - alpha) * h
    result = iterated.clone()
    for iteration in range(t):
        iterated = (alpha * P).mv(iterated)
        result += iterated
    return result


def calc_set_volume(node_set: list, phi: np.ndarray):
    volume = 0
    for node in node_set:
        volume += phi[node]
    return volume


def calc_conductance(numnodes: int, node_set: list, P: sp.spmatrix, phi: np.ndarray):
    volume_boundary = 0
    bar_node_set = {}
    for node in range(numnodes):
        if node not in node_set:
            bar_node_set[node] = 0
    for u in node_set:
        for v in P.rows[u]:
            if v in bar_node_set:
                volume_boundary += phi[u] * P[u, v]
    set_volume = calc_set_volume(node_set, phi)
    return volume_boundary/(min(set_volume, 1-set_volume))


def main(args):

    np.random.seed(args.seed)

    if args.device == '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    dataset_file_name = 'dblp_v14_' +  args.conf.upper() + '.json'

    dataset_file_path = osp.join(os.getcwd(), 'datasets', dataset_file_name)
            
    # load data and construct the hypergraph
    numnodes = 0
    numedges = 0
    authorid_to_nodeid = {}
    authorid_to_org = {}
    authorid_to_name = {}
    org_to_authorids = {}
    org_to_authorids_set = {}
    edgeid_to_nodes = {}
    nodeid_to_edges = {}
    edge_weights = {}
    edvw = {}


    with open(dataset_file_path, encoding='utf-8') as f:
        while True:
            line = f.readline().strip()     
            if not line:
                break
            js_line = json.loads(line)
            authors = js_line['authors']
            edgeid_to_nodes[numedges] = []
            for author in authors:
                if author['id'] not in authorid_to_nodeid:
                    authorid_to_nodeid[author['id']] = numnodes
                    nodeid_to_edges[authorid_to_nodeid[author['id']]] = []
                    numnodes += 1

                edgeid_to_nodes[numedges].append(authorid_to_nodeid[author['id']])
                nodeid_to_edges[authorid_to_nodeid[author['id']]].append(numedges)

                authorid_to_org[author['id']] = author['org']

                authorid_to_name[author['id']] = author['name']

                if author['org'] not in org_to_authorids:
                    org_to_authorids[author['org']] = []
                    org_to_authorids_set[author['org']] = set()
                    org_to_authorids[author['org']].append(author['id'])
                    org_to_authorids_set[author['org']].add(author['id'])
                else:
                    if author['id'] not in org_to_authorids_set[author['org']]:
                        org_to_authorids[author['org']].append(author['id'])
                        org_to_authorids_set[author['org']].add(author['id'])
            
            utils.update_edvw(edvw, numedges, authors, authorid_to_nodeid)
                    
            edge_weights[numedges] = js_line['n_citation'] + 1  # +1 to avoid zero entries
            numedges += 1

    nodeid_to_authorid = {}
    for authorid in authorid_to_nodeid:
        nodeid_to_authorid[authorid_to_nodeid[authorid]] = authorid


    total_hyperedge_node = 0
    for edgeid in edgeid_to_nodes:
        for nodeids in edgeid_to_nodes[edgeid]:
            total_hyperedge_node += 1
    print("Total Number of Nodes: ", numnodes)
    print("Total Number of Hyperedges: ", numedges)
    print("Total Number of Hyperedge-node Connections: ", total_hyperedge_node)



    # compute the matrices R, W, D_V, D_E, P
    R = sp.lil_matrix((numedges, numnodes))
    W = sp.lil_matrix((numnodes, numedges))
    Dv = sp.lil_matrix((numnodes, numnodes))
    Dv_inv = sp.lil_matrix((numnodes, numnodes))
    De = sp.lil_matrix((numedges, numedges))
    De_inv = sp.lil_matrix((numedges, numedges))

    for edge in edvw:
        for node in edvw[edge]:
            R[edge, node] = edvw[edge][node]

    for edge in edge_weights:
        for node in edgeid_to_nodes[edge]:
            W[node, edge] = edge_weights[edge]

    for node in range(numnodes):
        node_degree = 0
        for edge in nodeid_to_edges[node]:
            node_degree += edge_weights[edge]
        Dv[node, node] = node_degree
        Dv_inv[node, node] = 1/node_degree

    for edge in range(numedges):
        edge_delta = 0
        for nodeid in edgeid_to_nodes[edge]:
            edge_delta += edvw[edge][nodeid]
        De[edge, edge] = edge_delta
        De_inv[edge, edge] = 1/edge_delta

    P = Dv_inv.tocsr() @ W.tocsr() @ De_inv.tocsr() @ R.tocsr()
    PT = P.T
    # tensor for faster calculation
    PT_coo = PT.tocoo()
    values = PT_coo.data
    indices = np.vstack((PT_coo.row, PT_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = PT_coo.shape
    PT_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)
    P = P.tolil()


    # compute the stationary distribution
    phi = torch.from_numpy(np.ones(numnodes)/numnodes).to(device).float()
    for iter in range(500):
        phi = torch.mv(PT_tensor, phi)
        phi = phi/torch.sum(phi)
    phi_gpu = phi.clone()
    phi = phi.cpu().numpy()


    MIT_authors = []
    CMU_authors = []
    Stanford_authors = []
    UCB_authors = []
    MIT_authors_nodeid = []
    CMU_authors_nodeid = []
    Stanford_authors_nodeid = []
    UCB_authors_nodeid = []

    for org in MIT:
        try:
            MIT_authors += org_to_authorids[org]
        except:
            pass

    for org in CMU:
        try:
            CMU_authors += org_to_authorids[org]
        except:
            pass

    for org in Stanford:
        try:
            Stanford_authors += org_to_authorids[org]
        except:
            pass

    for org in UCB:
        try:
            UCB_authors += org_to_authorids[org]
        except:
            pass

    for authorid in MIT_authors:
        MIT_authors_nodeid.append(authorid_to_nodeid[authorid])
    for authorid in CMU_authors:
        CMU_authors_nodeid.append(authorid_to_nodeid[authorid])
    for authorid in Stanford_authors:
        Stanford_authors_nodeid.append(authorid_to_nodeid[authorid])
    for authorid in UCB_authors:
        UCB_authors_nodeid.append(authorid_to_nodeid[authorid])


    conductances = np.zeros(args.obs)
    same_org_ratio = np.zeros(args.obs)
    F1s = np.zeros(args.obs)
    sizes = np.zeros(args.obs)
    precisions = np.zeros(args.obs)
    recalls = np.zeros(args.obs)
    alphas = np.zeros(args.obs)

    for observation in range(args.obs):
        if args.verbose:
            print("Observation: ", observation)
        if args.conf != 'ir':
            univ = np.random.randint(4)
        else:
            univ = np.random.randint(3)

        if univ == 0:
            # randomly sample 5 authors from MIT
            starting = np.random.choice(MIT_authors, 5, replace=False)
        elif univ == 1:
            # randomly sample 5 authors from CMU
            starting = np.random.choice(CMU_authors, 5, replace=False)
        elif univ == 2:
            # randomly sample 5 authors from Stanford
            starting = np.random.choice(Stanford_authors, 5, replace=False)
        elif univ == 3:
            # randomly sample 5 authors from UCB
            starting = np.random.choice(UCB_authors, 5, replace=False)

        starting_nodes = []
        for authorid in starting:
            starting_nodes.append(authorid_to_nodeid[authorid])

        # first iteration to get an approximate conductance
        # compute PageRank Vector
        total_volume = calc_set_volume(starting_nodes, phi)
        v = np.zeros(numnodes)
        conductance_of_the_starting_nodes_set = calc_conductance(numnodes, starting_nodes, P, phi)
        for starting_node in starting_nodes:
            v[starting_node] = phi[starting_node]/total_volume
        ppr_vector = calc_ppr_by_power_iteration_gpu(PT_tensor, conductance_of_the_starting_nodes_set, torch.from_numpy(v).to(device).float(), 200)

        ppr_vector = ppr_vector.cpu().numpy()
        p_over_phis = []
        for node in range(numnodes):
            if ppr_vector[node] != 0 and phi[node] != 0:
                p_over_phis.append(ppr_vector[node]/phi[node])
            else:
                p_over_phis.append(0)

        sweep_nodes = np.argsort(np.array(p_over_phis))[::-1]

        local_cluster = starting_nodes.copy()
        best_local_cluster = starting_nodes.copy()
        smallest_conductance = conductance_of_the_starting_nodes_set
        counter = 0
        patience = 0
        for sweep_node in list(sweep_nodes):
            if sweep_node not in starting_nodes:
                local_cluster.append(sweep_node)
                current_conductance = calc_conductance(numnodes, local_cluster, P, phi)
                if  current_conductance < smallest_conductance:
                    best_local_cluster = local_cluster.copy()
                    smallest_conductance = current_conductance

                else:
                    patience += 1
                    if patience == args.patience2:
                        if args.verbose:
                            print("conductance hasn't broken the smallest record for a long time. Early Stop.")
                        break

        next_conductance = smallest_conductance

        # additional iteration to get the best cluster
        for iter in range(args.additional_rounds):
            if iter == args.additional_rounds - 1:
                alphas[observation] = next_conductance
            v = np.zeros(numnodes)
            for starting_node in starting_nodes:
                v[starting_node] = phi[starting_node]/total_volume
            ppr_vector = calc_ppr_by_power_iteration_gpu(PT_tensor, next_conductance, torch.from_numpy(v).to(device).float(), 200)

            ppr_vector = ppr_vector.cpu().numpy()
            p_over_phis = []
            for node in range(numnodes):
                if ppr_vector[node] != 0 and phi[node] != 0:
                    p_over_phis.append(ppr_vector[node]/phi[node])
                else:
                    p_over_phis.append(0)

            sweep_nodes = np.argsort(np.array(p_over_phis))[::-1]

            local_cluster = starting_nodes.copy()
            best_local_cluster_iter = starting_nodes.copy()
            smallest_conductance_iter = conductance_of_the_starting_nodes_set
            counter = 0
            patience = 0
            for sweep_node in list(sweep_nodes):
                if sweep_node not in starting_nodes:
                    local_cluster.append(sweep_node)
                    current_conductance = calc_conductance(numnodes, local_cluster, P, phi)
                    if  current_conductance < smallest_conductance_iter:
                        best_local_cluster_iter = local_cluster.copy()
                        smallest_conductance_iter = current_conductance

                    else:
                        patience += 1
                        if patience == args.patience:
                            if args.verbose:
                                print("conductance hasn't broken the smallest record for a long time. Early Stop.")
                            break
            if smallest_conductance_iter < smallest_conductance:
                best_local_cluster = best_local_cluster_iter.copy()
                smallest_conductance = smallest_conductance_iter
            next_conductance = smallest_conductance_iter

        # compute the F1 score of the best local cluster
        if univ == 0:
            F1, precision, recall = utils.F1_score(best_local_cluster, MIT_authors_nodeid)
        elif univ == 1:
            F1, precision, recall = utils.F1_score(best_local_cluster, CMU_authors_nodeid)
        elif univ == 2:
            F1, precision, recall = utils.F1_score(best_local_cluster, Stanford_authors_nodeid)
        elif univ == 3:
            F1, precision, recall = utils.F1_score(best_local_cluster, UCB_authors_nodeid)

        if args.verbose:
            print("The smallest conductance achieved is: ", smallest_conductance)
            print("The F1 score the best local cluster is: ", F1)
            print("size of the best local cluster:", len(local_cluster))
        
        conductances[observation] = smallest_conductance
        orgs = utils.nodeids_to_orgs(best_local_cluster, nodeid_to_authorid, authorid_to_org)
        names = utils.nodeids_to_names(best_local_cluster, nodeid_to_authorid, authorid_to_name)
        org_hit = 0
        for org in orgs:
            if univ == 0:
                if org in MIT:
                    org_hit += 1
            elif univ == 1:
                if org in CMU:
                    org_hit += 1
            elif univ == 2:
                if org in Stanford:
                    org_hit += 1
            elif univ == 3:
                if org in UCB:
                    org_hit += 1
        same_org_ratio[observation] = org_hit/len(orgs)
        F1s[observation] = F1
        sizes[observation] = len(local_cluster)
        precisions[observation] = precision
        recalls[observation] = recall

        cumulative_conductance = np.sum(conductances)
        cumulative_F1 = np.sum(F1s)

        # wrap the log into one line of observation index, cumulative conductances and F1 scores
        print("Observation: ", observation, "; Cumulative Conductance: ", cumulative_conductance, "; Cumulative F1 Score: ", cumulative_F1)

    print("Average Conductance: ", np.mean(conductances))
    print("Average F1 Score: ", np.mean(F1s))

    



    

                


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('hyperclus_local_conductance')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='')   # cuda:0
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--patience2', type=int, default=40)
    parser.add_argument('--conf', type=str, default = 'ml')
    parser.add_argument('--obs', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--additional_rounds', type=int, default=1)
    args = parser.parse_args()

    if args.conf.lower() not in ['ml']:
        print('illegal conference name')
        exit(0)

    args.conf = args.conf.lower()

    main(args)








