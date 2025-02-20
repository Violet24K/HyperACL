import numpy as np
import scipy.sparse as sp


MIT = ['MIT, CSAIL, Cambridge, MA 02139 USA', 'Massachusetts Institute of Technology', 'MIT', 'MIT, Cambridge, MA 02139 USA', 'MIT, EECS, 77 Massachusetts Ave, Cambridge, MA 02139 USA', 'MIT, 77 Massachusetts Ave, Cambridge, MA 02139 USA',]
CMU = ['Carnegie Mellon University', 'Carnegie - Mellon University#TAB#', 'Carnegie Mellon Univ, Pittsburgh, PA 15213 USA', 'Carnegie Mellon Univ, Dept Comp Sci, Pittsburgh, PA 15213 USA', 'Carnegie-Mellon University', 'Carnegie Mellon Univ, Sch Comp Sci, Pittsburgh, PA 15213 USA',
       'Carnegie Mellon Univ, Robot Inst, Pittsburgh, PA 15213 USA', 'Carnegie Mellon University, Pittsburgh, PA, USA']
Stanford = ['Stanford University', 'Stanford', 'Stanford Univ, Dept Comp Sci, Stanford, CA 94305 USA', 'Stanford Univ, Stanford, CA 94305 USA', 'Stanford Univ, Comp Sci Dept, Stanford, CA 94305 USA', 'Stanford Universtiy',
            'Stanford University, Stanford, CA, USA', 'Stanford University, USA']
UCB = ['University of California, Berkeley', '#N#University of California Berkeley#N#', 'Univ Calif Berkeley, Berkeley Artificial Intelligence Res, Berkeley, CA 94720 USA', 'Univ Calif Berkeley, Berkeley, CA 94720 USA', 'UC Berkeley', 'UC Berkeley,#TAB#', 
       'Univ Calif Berkeley, Berkeley, CA USA', 'Univ Calif Berkeley, Dept Comp Sci, Berkeley, CA 94720 USA', 'EECS Department, University of California, Berkeley', 'Univ Calif Berkeley, Berkeley AI Res, Berkeley, CA 94720 USA', 
       'Univ Calif Berkeley, Dept Elect Engn & Comp Sci, Berkeley, CA 94720 USA', 'UC Berkeley)',
       'Univ Calif Berkeley, Dept EECS, Berkeley, CA 94720 USA']



def org_to_nodeid(org: str, org_to_authorids: dict, authorid_to_nodeid: dict):
    nodeids = []
    for authorid in org_to_authorids[org]:
        nodeids.append(authorid_to_nodeid[authorid])
    return nodeids


def nodeids_to_orgs(nodeids: list, nodeid_to_authorid: dict, authorid_to_org: dict):
    orgs = []
    for nodeid in nodeids:
        orgs.append(authorid_to_org[nodeid_to_authorid[nodeid]])
    return orgs

def nodeids_to_names(nodeids: list, nodeid_to_authorid: dict, authorid_to_name: dict):
    names = []
    for nodeid in nodeids:
        names.append(authorid_to_name[nodeid_to_authorid[nodeid]])
    return names

def calc_ppr_by_power_iteration(P: sp.spmatrix, alpha: float, h: np.ndarray, t: int) -> np.ndarray:
    iterated = (1 - alpha) * h
    result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        result += iterated
    return result


# list minus
def lm(list1, list2):
    result = [x for x in list1 if x not in list2]
    return result


def rpr_to_lazy(alpha):
    return 2*alpha/(1+alpha)




def F1_score(list1, list2):
    tp = 0
    for x in list1:
        if x in list2:
            tp += 1
    if tp == 0:
        return 0, 0, 0
    precision = tp/len(list1)
    recall = tp/len(list2)
    F1 = 2*precision*recall/(precision+recall)
    return F1, precision, recall



def update_edvw(edvw, numedges, authors, authorid_to_nodeid):
    edvw[numedges] = {}
    authorship_position = 1
    if len(authors) == 1:
        edvw[numedges][authorid_to_nodeid[authors[0]['id']]] = 1
    elif len(authors) == 2:
        edvw[numedges][authorid_to_nodeid[authors[0]['id']]] = 1
        edvw[numedges][authorid_to_nodeid[authors[1]['id']]] = 1
    else:
        if len(authors)%2 == 0:
            edvw[numedges][authorid_to_nodeid[authors[len(authors)//2]['id']]] = 1
            for i in range(len(authors)//2):
                edvw[numedges][authorid_to_nodeid[authors[i]['id']]] = 2**(len(authors)//2 - i)
            for i in range(len(authors)//2 + 1, len(authors)):
                edvw[numedges][authorid_to_nodeid[authors[i]['id']]] = 2**(i - len(authors)//2)
        else:
            edvw[numedges][authorid_to_nodeid[authors[len(authors)//2 - 1]['id']]] = 1
            edvw[numedges][authorid_to_nodeid[authors[len(authors)//2]['id']]] = 1
            for i in range(len(authors)//2-1):
                edvw[numedges][authorid_to_nodeid[authors[i]['id']]] = 2**(len(authors)//2 - 1 - i)
            for i in range(len(authors)//2 + 1, len(authors)):
                edvw[numedges][authorid_to_nodeid[authors[i]['id']]] = 2**(i - len(authors)//2)