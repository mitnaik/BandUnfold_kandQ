import numpy as np
import matplotlib.pyplot as plt


tol5 = 1.e-5

def k_range( kin):
    ktmp = np.copy(kin)
    for i in range(3):
        while ktmp[i] < -tol5:
            ktmp[i] = ktmp[i] + 1.0
        while ktmp[i] >= 1-tol5:
            ktmp[i] = ktmp[i] - 1.0
    return ktmp

def tpbatocrys_one(kpt_tpba, B):
    # Input: kpoints in tpba format
    # B: Reciprocal lattice vectors in tpba units
    kpt_tpba = np.array(kpt_tpba)
    A = np.linalg.inv(B)
    k_c = np.dot(kpt_tpba, A)
    return np.array(k_c)

def tpbatocrys(kpt_tpba, B):
    # Input: kpoints in tpba format
    # B: Reciprocal lattice vectors in tpba units
    kpt_tpba = np.array(kpt_tpba)
    kpt_c = []
    A = np.linalg.inv(B)
    for j in range(len(kpt_tpba)):
        tmp = []
        for i in range(len(kpt_tpba[0])):
            k_c = np.dot(kpt_tpba[j,i], A)
            tmp.append(k_c)
        kpt_c.append(tmp)
    return np.array(kpt_c)


def check_umk(Q1,Q2, B_uc):
    b1_uc = B_uc[0]
    b2_uc = B_uc[1]
    diff0 = np.linalg.norm(Q2 - Q1)
    umk1 = Q2 + b1_uc
    diff1 = np.linalg.norm(umk1 - Q1)
    umk2 = Q2 + b2_uc
    diff2 = np.linalg.norm(umk2 - Q1)
    umk3 = Q2 + b1_uc + b2_uc
    diff3 = np.linalg.norm(umk3 - Q1)
    umk4 = Q2 - b1_uc
    diff4 = np.linalg.norm(umk4 - Q1)
    umk5 = Q2 - b2_uc
    diff5 = np.linalg.norm(umk5 - Q1)
    umk6 = Q2 - b1_uc - b2_uc
    diff6 = np.linalg.norm(umk6 - Q1)
    umk7 = Q2 - b1_uc + b2_uc
    diff7 = np.linalg.norm(umk7 - Q1)
    umk8 = Q2 + b1_uc - b2_uc
    diff8 = np.linalg.norm(umk8 - Q1)
    if diff0 < 1.e-6 or diff1 < 1e-6 or diff2 < 1e-6 or diff3 < 1e-6 or diff4 < 1e-6 or diff5 < 1e-6 or diff6 < 1e-6 or diff7 < 1e-6 or diff8 < 1e-6 :
        return False
    return True

def buc_indices(Emap_k):
    # Fills an array same shape as Emap_k
    # with the k-point indices:
    # [[0,1,2,3,...], [0,1,2,3,...],...]
    i_bnd = np.zeros(np.shape(Emap_k))
    for ib in range(len(Emap_k[0])):
        i_bnd[:,ib] =  ib
    return i_bnd

def kuc_indices(Emap_k):
    # Fills an array same shape as Emap_k
    # with the k-point indices:
    # [[0,0,0,0,...], [1,1,1,1,...],...]
    i_kpt = np.zeros(np.shape(Emap_k))
    for ik in range(len(Emap_k[:,0])):
        i_kpt[ik] =  ik
    return i_kpt

def read_inp(f_name):
    fp = open(f_name)
    lines = fp.readlines()
    A_uc = np.zeros((3,3))
    sc = np.zeros((2,2), dtype = np.integer)
    for i in range(len(lines)):
        if "nv_sc" in lines[i]:
            w = lines[i+1].split()
            nv_sc, nc_sc, nF_sc = eval(w[0]), eval(w[1]), eval(w[2])
        if "alat" in lines[i]:
            w = lines[i+1].split()
            alat = eval(w[0])
        if "Unit-cell vectors" in lines[i]:
            for j in range(3):
                w = lines[i+j + 1].split()
                A_uc[j] = np.array([eval(w[0]), eval(w[1]), eval(w[2])])
        if "Super-cell vectors" in lines[i]:
            for j in range(2):
                w = lines[i+j + 1].split()
                sc[j] = np.array([eval(w[0]), eval(w[1])])
    return alat, A_uc, sc, nv_sc, nc_sc, nF_sc


alat, A_uc, sc, nv_sc, nc_sc, nF_sc = read_inp("map.inp")
B_uc = np.linalg.inv(A_uc)
B_uc = np.transpose(B_uc)
A_sc = np.zeros((3,3))
A_sc[0] = A_uc[0]*sc[0,0] + A_uc[1]*sc[0,1]
A_sc[1] = A_uc[0]*sc[1,0] + A_uc[1]*sc[1,1]
A_sc[2] = A_uc[2]

kuc_all = np.load("kuc_all.npy")
kuc_map = np.load("kuc_map.npy")
kuc_map_crys = tpbatocrys(kuc_map, B_uc)
np.save("kuc_map_crys.npy", kuc_map_crys)
#np.savetxt("kuc_map_tpba.dat", kuc_map)

kuc_G = kuc_all[0]
kmap_G = kuc_map[0]

print(np.shape(kuc_G), np.shape(kmap_G))
#plt.scatter(kuc_G[:,0], kuc_G[:,1], color = 'r')
#plt.scatter(kmap_G[:,0], kmap_G[:,1], alpha = 0.5)
#plt.title("G")
#plt.show()

# Emap[nk_sc, nk_uc, eig(nk_uc)]
Emap = np.load("E_map.npy")
nk_sc = len(Emap)
print(np.shape(kuc_map), np.shape(Emap))


Emap_sort = []
ikpt_sort = []
ibnd_sort = []

# Loop over all s.c. k-points in Emap
for iksc in range(nk_sc):
    # fill kpoint indices array (same shape as Emap[iksc])
    i_kpt = kuc_indices(Emap[iksc])
    i_bnd = buc_indices(Emap[iksc])
    Emap_fl = np.ndarray.flatten(Emap[iksc])
    #kuc_map_ = np.ndarray.flatten(kuc_map[iksc])
    i_kpt_fl = np.ndarray.flatten(i_kpt)
    i_bnd_fl = np.ndarray.flatten(i_bnd)
    isort = np.argsort(Emap_fl)
    # Sorted Emap, ikpt for a given s.c. k-point
    Etmp_sort = np.take_along_axis(Emap_fl, isort, axis = 0)
    iktmp_sort  = np.take_along_axis(i_kpt_fl, isort, axis = 0)
    ibtmp_sort  = np.take_along_axis(i_bnd_fl, isort, axis = 0)
    Emap_sort.append(Etmp_sort)
    ikpt_sort.append(iktmp_sort)
    ibnd_sort.append(ibtmp_sort)

print("i_bnd:",i_bnd)
Emap_sort = np.array(Emap_sort)
ikpt_sort = np.array(ikpt_sort)
ibnd_sort = np.array(ibnd_sort)
collect_kc = []
collect_ibc = []
collect_Ec = []
collect_kv = []
collect_ibv = []
collect_Ev = []

for ik_sc in range(nk_sc):
    #for i in range(len(Emap_sort[ik_sc])):
    tmp_kv = []
    tmp_bv = []
    tmp_Ev = []
    tmp_kc = []
    tmp_ibv = []
    tmp_ibc = []
    tmp_Ec = []
    for i in range(nF_sc - nv_sc, nF_sc):
        #if Emap_sort[ik_sc,i] > -0.30 and Emap_sort[ik_sc,i] < 0.0:
        kmap = np.copy(kuc_map[ik_sc])
        tmp_kv.append(kmap[int(ikpt_sort[ik_sc,i])])
        tmp_ibv.append(ibnd_sort[ik_sc,i])
        tmp_Ev.append(Emap_sort[ik_sc,i])
        print(Emap_sort[ik_sc,i], ibnd_sort[ik_sc,i], kmap[int(ikpt_sort[ik_sc,i])])
    for i in range(nF_sc, nF_sc + nc_sc):
        tmp_kc.append(kmap[int(ikpt_sort[ik_sc,i])])
        tmp_ibc.append(ibnd_sort[ik_sc,i])
        tmp_Ec.append(Emap_sort[ik_sc,i])
        print(Emap_sort[ik_sc,i], ibnd_sort[ik_sc,i], kmap[int(ikpt_sort[ik_sc,i])])
    collect_kv.append(np.flip(tmp_kv,0))
    collect_ibv.append(np.flip(tmp_ibv,0))
    collect_Ev.append(np.flip(tmp_Ev,0))
    collect_kc.append(tmp_kc)
    collect_ibc.append(tmp_ibc)
    collect_Ec.append(tmp_Ec)

print(collect_kc)
print(collect_ibc)
print(collect_kv)
print(collect_ibv)
collect_kv = np.array(collect_kv)# + collect_kc2)
collect_kv_crys = tpbatocrys(collect_kv, B_uc)
collect_ibv = np.array(collect_ibv)# + collect_kc2)
collect_ibc = np.array(collect_ibc)# + collect_kc2)
collect_kc = np.array(collect_kc)# + collect_kc2)
collect_kc_crys = tpbatocrys(collect_kc, B_uc)
collect_Ev = np.array(collect_Ev)# + collect_kc2)
collect_Ec = np.array(collect_Ec)# + collect_kc2)


print('Valence k list',collect_kv_crys)
print('Conduction k list',collect_kc_crys)

np.save("collect_kc", collect_kc)
np.save("collect_kv", collect_kv)
np.save("collect_ibc", collect_ibc)
np.save("collect_ibv", collect_ibv)
np.save("collect_Ec", collect_Ec)
np.save("collect_Ev", collect_Ev)


#nk_sc = 1
nQ0 = 0
Q = []
for ikp in range(nk_sc):
    for ik in range(nk_sc):
        for ic in range(nc_sc):
            for iv in range(nv_sc):
                for icp in range(nc_sc):
                    for ivp in range(nv_sc):
                        # BSE involves transition: ik_sc, ic, iv to ikp_sc, icp', ivp
                        # (ic, ik_sc) -> (ic_uc,ik1) 
                        # (iv, ik_sc) -> (iv_uc,ik2) 
                        # (icp, ik_sc) -> (icp_uc,ik3) 
                        # (ivp, ik_sc) -> (ivp_uc,ik4) 
                        k1 = collect_kc[ik,ic] 
                        ib1 = collect_ibc[ik,ic] 
                        k2 = collect_kv[ik,iv] 
                        ib2 = collect_ibv[ik,ic] 
                        k3 = collect_kc[ikp,icp] 
                        ib3 = collect_ibc[ikp,icp] 
                        k4 = collect_kv[ikp,ivp] 
                        ib4 = collect_ibv[ikp,ivp] 
                        k1_crys = collect_kc_crys[ik,ic]
                        k2_crys = collect_kv_crys[ik,iv]
                        k3_crys = collect_kc_crys[ikp,icp]
                        k4_crys = collect_kv_crys[ikp,ivp]
                        diff1 = k2_crys - k1_crys
                        diff2 = k4_crys - k3_crys
                        #Q1_crys = tpbatocrys_one(Q1, B_uc)
                        #Q2_crys = tpbatocrys_one(Q2, B_uc)
						#dQ = Q2_crys - Q1_crys
                        if np.linalg.norm(k_range(diff1)) < 1e-5 and np.linalg.norm(k_range(diff2)) < 1e-5:
                            Qq = 0
                            nQ0 += 1
                            #print(k1,k2,k3,k4,"Q = 0")
                            #print("Q = 0", k1,k2)
                        elif np.linalg.norm(k_range(diff1) - k_range(diff2)) < 1e-5:
                            Q.append(k2 - k1)
                            #print("Q = (%1.14f %1.14f ), (%1.14f %1.14f)"%(diff1[0],diff1[1],
                                #diff2[0],diff2[1]))
                            #print("k = (%1.6f %1.6f ), (%1.6f %1.6f), (%1.6f %1.6f), (%1.6f %1.6f)"%(k1[0],k1[1],
                            #    k2[0],k2[1], k3[0],k3[1], k4[0],k4[1]))
                        else:
                            Val = 0


Q = np.array(Q)
print("len Q", len(Q))
print("len Q0", nQ0)
#print(np.unique(Q.round(decimals = 9),axis = 0), len(np.unique(Q.round(decimals = 9),axis = 0)))
Q_uniq = np.unique(Q.round(decimals = 9),axis = 0)
Q_crys = tpbatocrys_one(Q_uniq, B_uc)
np.save("Q_crys_umk", Q_crys)
np.save("Q_all", Q)
np.save("Q_tpba_umk", Q_uniq)
np.savetxt("Q_crys_umk",Q_crys, fmt = '%1.10f')
print(Q_crys[0,0]*B_uc[0] + Q_crys[0,1]*B_uc[1])
print(Q_crys[17,0]*B_uc[0] + Q_crys[17,1]*B_uc[1])

print(B_uc)
