from NNLS_SPAR import *
from static_questions import * 
from matplotlib.pyplot import *


A = ozone_X2
b = ozone_Y1
qx = Q20_x[:,0]
true_index = where(qx>0)[0]
# s = len(true_index)
s = 8 
m,n = shape(A)

me = reshape(mean(A,axis = 0),[1,n])
A = A - ones([m,1]).dot(me)
se = std(A,axis = 0)
A = multiply(A,1/array(se))
q,r = linalg.qr(A)
lin_dep = where(abs(diag(r)) <= 1e-10)
lin_indep = list(range(n))
if len(lin_dep[0]) != 0:
    for i in range(len(lin_dep[0])):
        lin_indep.remove(lin_dep[0][i])
    A = A[:,lin_indep]
    n = n - len(lin_dep[0])


omega = NNLSSPAR(A,b,s)
omega1 = NNLSSPAR(A,b,s)
omega2 = NNLSSPAR(A,b,s)

# A_L = [ozone_X2,energy_x,conduct_x,A]
# b_L = [ozone_Y1,energy_y,conduct_y,online_y]
# for i in range(4):
#     t2 = time.time()
#     t0 = time.process_time()
#     q,r = linalg.qr(A_L[i])
#     qb = q.T.dot(b_L[i])
#     permanent_residual = norm(b) ** 2 - norm(qb) ** 2
#     t1 = time.process_time()
#     t3 = time.time()

#     print("CPU {:1.4f} Real {:1.4f}".format(t1-t0,t3-t2))

print("$s$ & MIP CPU  & NNLSSPAR CPU & MIP Real & NNLSSPAR \\\ ")

for s in range(2,3):
    omega.s = s
    omega1.s = s
    omega2.s = s
    
    
    omega.many = 1
    omega.verbose = False
    omega.solver = "GUROBI"
    t2 = time.time()
    t0 = time.process_time()
    # omega.gurobi_mip2(list(range(n)))
    t1 = time.process_time()
    t3 = time.time()
    
    omega1.many = 1
    omega1.verbose = False
    #for s in range(1,4):
    #random_break = input("continue ?")
    #omega1.s = s
    omega1.solver = "CPLEX"
    x = omega1.nonnegative_IHT(list(range(n)))
    ind = where(x > 0)[0]
    # f = omega1.qr_nnls(list(range(n)))
    # si = argsort(-1*f[0])
    # ns = omega.rtilde[:,si]
    # q,r = linalg.qr(ns)
    # omega1.rtilde = r
    # omega1.qb = q.T.dot(omega.qb)
    # omega1.arborescent(list(range(n)),n,f[1],f[0][si])
    t22 = time.time()
    t00 = time.process_time()
    omega1.solve_arborescent()
    t11 = time.process_time()
    t33 = time.time()
    # omega1.gurobi_mip3(list(range(n)),ind,x[ind])
    
    omega2.many = 1
    omega2.verbose = False
    #for s in range(1,4):
    #random_break = input("continue ?")
    #omega1.s = s
    omega2.solver = "GUROBI"
    t222 = time.time()
    t000 = time.process_time()
    omega2.solve_nnls(list(range(n)))
    t111 = time.process_time()
    t333 = time.time()
    #sys.stdout = omegalul
    # print("{:d}  & {:1.4f} & {:1.4f} \\\ ".format(s,t11-t00,t33-t22))
    print("{:d}  & {:1.8f} & {:1.8f} & {:1.8f} & {:1.8f} \\\ ".format(s,t11-t00,t111-t000,t33-t22,t333-t222))
    # print("{:d} & {:1.4f} & {:1.4f} & {:1.4f} & {:1.4f} & {:1.4f} & {:1.4f} \\\ ".format(s,t1-t0,t11-t00,t111-t000,t3-t2,t33-t22,t333-t222))
    # print("cpu time , first = {:1.4f} , second {:1.4f}".format(t1-t0,t11-t00))
    # print("real time, first = {:1.4f} , second {:1.4f}".format(t3-t2,t33-t22))

"""
m_cpu = zeros((10,8))
m_memory = zeros((10,8))
m_cpu_qr = zeros((10,8))
m_memory_qr = zeros((10,8))
m_node = zeros((10,8))

m_scpu = zeros((10,5))
m_smemory = zeros((10,5))
m_scpu_qr = zeros((10,5))
m_smemory_qr = zeros((10,5))
m_snode = zeros((10,5))

for j in range(10):
    cpu = []
    memory = []
    cpu_qr = []
    memory_qr = []
    node = []


    for i in range(1,9):
        A = random.randn(100*i,25*i)
        x = reshape(hstack((array([2,3,4,5,6,7,8,9,10,11]),zeros(15),zeros(25*(i-1)))),[25*(i),1])
        x = reshape(hstack((array([4,4,4,4,4,4,4,4,4,4]),zeros(15),zeros(25*(i-1)))),[25*(i),1])
        b = A.dot(x) + random.randn(100*i,1)
        m,n = shape(A)
        omega = NNLSSPAR(A,b,10)
        print(j,i)
        mem = hpy()
        mem.heap()
        mem.setrelheap()
        t2 = time.time()
        t0 = time.process_time()
        omega.solve_nnls(list(range(25*i)))
        t1 = time.process_time()
        t3 = time.time()
        m = mem.heap()
        
        cpu.append(t1-t0)
        memory.append(m.size + + (8*n+640)*omega.node)
        node.append(omega.node)
        

        mem1 = hpy()
        mem1.heap()
        mem1.setrelheap()
        t22 = time.time()
        t00 = time.process_time()
        omega.solve_qr_nnls(list(range(25*i)))
        t11 = time.process_time()
        t33 = time.time()
        m1 = mem.heap()
    
        cpu_qr.append(t11-t00)
        memory_qr.append(m1.size + + (8*n+640)*omega.node)
    
    m_cpu[j,:] = cpu
    m_memory[j,:] = memory
    m_cpu_qr[j,:] = cpu_qr
    m_memory_qr[j,:] = memory_qr 
    m_node[j,:] = node
    

for j in range(10):    
    A = random.randn(400,100)
    x = array([[2,3,4,5,6,7,8,9,10,11]])
    x = array([[4,4,4,4,4,4,4,4,4,4]])
    
    x = hstack((x,zeros([1,90])))
    b = A.dot(x.T) + random.randn(400,1)
    m,n = shape(A)
    
    scpu = []
    smemory = []
    scpu_qr = []
    smemory_qr = []
    snode = []
    
    omega = NNLSSPAR(A,b,8)
    
    for s in range(8,13):
        print(j,s)
        omega.s = s
        omega.node = 0
        omega.memory = 0
        mem = hpy()
        mem.heap()
        mem.setrelheap()
        t2 = time.time()
        t0 = time.process_time()
        omega.solve_nnls(list(range(100)))
        t1 = time.process_time()
        t3 = time.time()
        m = mem.heap()
        
        scpu.append(t1-t0)
        smemory.append(m.size + (8*n+640)*omega.node)
        snode.append(omega.node)
        
        mem1 = hpy()
        mem1.heap()
        mem1.setrelheap()
        t22 = time.time()
        t00 = time.process_time()
        omega.solve_qr_nnls(list(range(100)))
        t11 = time.process_time()
        t33 = time.time()
        m1 = mem.heap()
        
        scpu_qr.append(t11-t00)
        smemory_qr.append(m1.size + (8*n+640)*omega.node)

    
    m_scpu[j,:] = scpu
    m_smemory[j,:] = smemory
    m_scpu_qr[j,:] = scpu_qr
    m_smemory_qr[j,:] = smemory_qr
    m_snode[j,:] = snode


t1 = range(25,201,25)
t2 = range(8,13)
g = figure(1)

subplot(2,3,1)

plot(t1,mean(m_cpu_qr,axis = 0),"k")
plot(t1,m_cpu_qr.max(axis = 0),"r*")
plot(t1,m_cpu_qr.min(axis = 0),"g*")
title("cpu usage against varying size of matrix A")
xlabel("n,where A is m x n and m = 4 * n")
ylabel("cpu usage in seconds")
legend(["mean time","maximum time","minimum time"])

subplot(2,3,2)

plot(t1,mean(m_memory_qr,axis = 0),"k")
plot(t1,m_memory_qr.max(axis = 0),"r*")
plot(t1,m_memory_qr.min(axis = 0),"g*")
title("memory usage against varying size of matrix A")
xlabel("n,where A is m x n and m = 4 * n")
ylabel("memory usage in bytes")
legend(["mean size","maximum size","minimum size"])

subplot(2,3,3)

plot(t1,mean(m_node,axis = 0),"k")
plot(t1,m_node.max(axis = 0),"r*")
plot(t1,m_node.min(axis = 0),"g*")
title("number of nodes against varying size of matrix A")
xlabel("n,where A is m x n and m = 4 * n")
ylabel("number of nodes")
legend(["mean number","maximum number","minimum number"])

subplot(2,3,4)

plot(t2,mean(m_scpu_qr,axis = 0),"k")
plot(t2,m_scpu_qr.max(axis = 0),"r*")
plot(t2,m_scpu_qr.min(axis = 0),"g*")
title("cpu usage against changing sparsity level ")
xlabel("sparsity level s")
ylabel("cpu usage in seconds")
legend(["mean time","maximum time","minimum time"])

subplot(2,3,5)

plot(t2,mean(m_smemory_qr,axis = 0),"k")
plot(t2,m_smemory_qr.max(axis = 0),"r*")
plot(t2,m_smemory_qr.min(axis = 0),"g*")
title("memory usage against changing sparsity level")
xlabel("sparsity level s")
ylabel("memory usage in bytes")
legend(["mean size","maximum size","minimum size"])

subplot(2,3,6)

plot(t2,mean(m_snode,axis = 0),"k")
plot(t2,m_snode.max(axis = 0),"r*")
plot(t2,m_snode.min(axis = 0),"g*")
title("number of nodes against changing sparsity level")
xlabel("sparsity level s")
ylabel("number of nodes")
legend(["mean number","maximum number","minimum number"])
"""