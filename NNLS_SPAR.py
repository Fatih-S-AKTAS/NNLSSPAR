from scipy.linalg import *
from numpy import *
from queue import *
from statistics import *
from guppy import hpy
import time 
import os
import sys
import cvxpy as cp
from scipy.optimize import nnls
import gurobipy

class NNLSSPAR:
    def __init__(self,A,b,s):
        """ 
        initilization of parameters 
        -------------------------------------------------------------------------------------
        A       = m x n matrix consisting of m observations of n independent variables
        b       = m x 1 column vector consisting of m observed dependent variable
        s       = sparsity level
        
        Solver parameters
        -------------------------------------------------------------------------------------
        P       = Indexes of possible choices for independent variable
        C       = Chosen set of independent variables
        
        
        If you get a pythonic error, it will probably because of how unfriendly is python with usage of 
        vectors, like lists cannot be sliced with indeces but arrays can be done, hstack converts everything
        to floating points, registering a vector and randomly trying to make it 2 dimensional takes too much
        unnecessary work, some functions do inverse of hstack, randomly converting numbers to integers etc.
        
        Please contact selim.aktas@ug.bilkent.edu.tr for any bugs, errors and recommendations.
        """
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if math.isnan(A[i,j]) or abs(A[i,j]) == Inf:
                    print("Matrix A has NAN or Inf values, it will give linear algebra errors")
                    break
        for i in range(len(A)):
            for j in range(len(A[0])):
                if type(A[i,j]) != float64:
                    print("Matrix A should be registered as float64, otherwise computations will be wrong\
for example; Variance will be negative")
            
        if shape(shape(b)) == (1,):
            print("you did not register the vector b as a vector of size m x 1, you are making a pythonic\
                  error, please reshape the vector b")
        elif shape(A)[0] != shape(b)[0]:
            print("Dimensions of A and b must match, A is",shape(A)[0],"x",shape(A)[1],"matrix", \
                  "b is",shape(b)[0],"x 1 vector, make sure" ,shape(A)[0],"=",shape(b)[0])
        elif shape(A)[0] <= shape(A)[1]:
            print("The linear system is supposed to be overdetermined, given matrix A of size m x n,",\
                  "the condition m > n should be satisfied") 
        elif shape(b)[1] != 1:
            print("input argument b should be a vector of size m x 1, where m is the row size of the \
                  A")
        elif type(A) != ndarray:
            print("A should be a numpy ndarray")
        elif type(s) != int:
            print("s should be an integer")
        else:
            self.A = A
            self.b = b
            self.bv = b[:,0]
            self.s = s
            """ registering the matrix A independent variable values, vector b dependent variable values 
            and s sparsity level """
            self.m = shape(A)[0]
            self.n = shape(A)[1]
            """ saving the shape of matrix A explicitly not to repeat use of shape() function """
            self.node = 0
            """ initializing the number of end nodes or equivalently number of branching done """
            self.check = 0
            """ initializing number of nodes visited """
            self.means = abs(A.mean(axis = 0))
            """ storing the mean values of each independent variable which will be used for 
            selection criteria in solver algortihms"""
            self.sterror = std(A,axis = 0)
            """ storing the standard deviation and variance of each independent variable which will
            be used for selection criteria in solver algortihms"""
            self.rem_qsize = []
            """ initializing remaining queue size after the algorithm finishes """
            self.out = 1
            """ initializing the output choice of algorithm """
            self.SST = variance(b[:,0])*(self.m-1)
            self.original_stdout = sys.stdout
            """  saving original stdout, after manipulating it for changing values of out, we should be
            able to recover it """
            
            q,r = linalg.qr(A)
            self.q = q
            self.rtilde = r
            self.qb = self.q.T.dot(self.b)
            self.permanent_residual = norm(b) ** 2 - norm(self.qb) ** 2
            """ initial QR decomposition to shrink the system """
            self.tablelookup = {}
            self.tablelookup_nnls = {}
            """ lookup tables for solving all subset problem """
            self.many = 1
            """ initializing the number of solutions to be found"""
            self.residual_squared = []
            self.indexes = []
            self.coefficients = []
            """ initializing the arrays of solution parameters """
            
            self.bigm = max(abs(lstsq(A,b)[0]))*4 
            """ bigm value for integer programming """
            self.verbose = False
            """ initializing verbose to false, hence MIP solvers will not print out every step of the algorithm """
            self.solver = "GUROBI"
            """ initializing the CVXPY solver to GUROBI because it proved to be the fastest in our experience """
            self.memory = 0
            """ initializing memory usage of the algorithm """
            self.cpu = 0
            """ initializing cpu usage of the algorithm """
            
            self.rtilde2 = r.T.dot(r)
            self.A2 = self.A.T.dot(self.A)
            
            self.best_feasible = Inf
            self.searched = []
    
            
    def qp_mip1(self,ind,ce = []):
        l = len(ind)
        A_i = vstack((eye(l),-1*eye(l)))
        A_iy = vstack((-1*self.bigm*eye(l),-1*self.bigm*eye(l)))
        A_iy2 = ones([1,l])
        
        
        b_i = zeros([2*l,1])
        #b_i2 = vstack((self.s,-1*self.s))
        b_i2 = self.s
        
        x = cp.Variable((l,1))
        y = cp.Variable((l,1), boolean = True) 
        
        cost = cp.Minimize(cp.sum_squares(self.A[:,ind] @ x - self.b))
        if ce == []:
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(l) @ y <= ones([l,1]),-1*eye(l) @ x <= zeros([l,1]) ]
        else:
            k = zeros([l,1])
            k[ce] = 1
            CE = vstack((diag(k),-1*diag(k)))
            CEb = vstack((k,-1*k))
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(self.n) @ y <= ones([self.n,1]), CE @ y <= CEb,-1*eye(l) @ x <= zeros([l,1]) ]
        
        prob = cp.Problem(cost,constraints)
        prob.solve(solver = self.solver,verbose = self.verbose)
        
        self.residual_squared.append(prob.value)
        ind = where(abs(x.value[:self.n,0]) > 1e-6)[0]
        self.indexes.append(ind)
        self.coefficients.append(x.value[ind,0])
        return [x.value[:self.n,0],prob.value,y.value[:self.n,0]]

    def qp_mip2(self,ind,ce = []):
        l = len(ind)
        A_i = vstack((eye(l),-1*eye(l)))
        A_iy = vstack((-1*self.bigm*eye(l),-1*self.bigm*eye(l)))
        A_iy2 = ones([1,l])
        
        
        b_i = zeros([2*l,1])
        #b_i2 = vstack((self.s,-1*self.s))
        b_i2 = self.s
        
        x = cp.Variable((l,1))
        y = cp.Variable((l,1), boolean = True) 
        
        cost = cp.Minimize(cp.sum_squares(self.rtilde[:,ind] @ x - self.qb))
        if ce == []:
            # constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(l) @ y <= ones([l,1]),-1*eye(l) @ x <= zeros([l,1]) ]
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2,-1*eye(l) @ x <= zeros([l,1]) ]
        else:
            k = zeros([l,1])
            k[ce] = 1
            CE = vstack((diag(k),-1*diag(k)))
            CEb = vstack((k,-1*k))
            # constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, eye(l) @ y <= ones([l,1]), CE @ y <= CEb,-1*eye(l) @ x <= zeros([l,1]) ]
            constraints = [ A_i @ x + A_iy @ y <= b_i, A_iy2 @ y <= b_i2, CE @ y <= CEb,-1*eye(l) @ x <= zeros([l,1]) ]
        
        prob = cp.Problem(cost,constraints)
        prob.solve(solver = self.solver,verbose = self.verbose)
        
        self.residual_squared.append(prob.value)
        ind = list(where(x.value[:self.n,0] > 1e-6)[0])
        self.indexes.append(ind)
        self.coefficients.append(x.value[ind,0])
        return [x.value[:self.n,0],prob.value,y.value[:self.n,0]]
    
    def gurobi_mip1(self,ind):
        l = len(ind)
        
        prob = gurobipy.Model("gurobi_mip")
        prob.setParam( 'OutputFlag',0 )
        x = prob.addMVar(shape = l,vtype = "C",name = "x")
        y = prob.addMVar(shape = l,vtype = "B",name = "y")

        
        prob.setObjective(x @ self.A2[ind,:][:,ind] @ x - 2 * self.b[:,0].T.dot( self.A[:,ind]) @ x ,gurobipy.GRB.MINIMIZE)
        # prob.setObjective(x @ self.A[:,ind].T.dot(self.A[:,ind]) @ x - 2 * self.b[:,0].T @ self.A[:,ind] @ x ,gurobipy.GRB.MINIMIZE)
        prob.addConstr(eye(l) @ x <= self.bigm * eye(l) @ y)
        prob.addConstr(eye(l) @ x >= -1 * self.bigm *eye(l) @ y)
        prob.addConstr(ones(l) @ y <= self.s)
        
        prob.optimize()
        self.residual_squared.append(prob.objVal + norm(self.b,2) ** 2 - self.permanent_residual)
        ind = where( x.X > 1e-6)[0]
        self.indexes.append(list(ind))
        self.coefficients.append(x.X[ind])
        return [x.X,prob.objVal]  

    
    def gurobi_mip2(self,ind):
        l = len(ind)
        
        prob = gurobipy.Model("gurobi_mip")
        prob.setParam( 'OutputFlag',0 )
        x = prob.addMVar(shape = l,vtype = "C",name = "x")
        y = prob.addMVar(shape = l,vtype = "B",name = "y")

        
        prob.setObjective(x @ self.rtilde2[ind,:][:,ind] @ x - 2 * self.qb[:,0].T.dot(self.rtilde[:,ind]) @ x ,gurobipy.GRB.MINIMIZE)
        # prob.setObjective(x @ self.rtilde[:,ind].T.dot(self.rtilde[:,ind]) @ x - 2 * self.qb[:,0].T @ self.rtilde[:,ind] @ x ,gurobipy.GRB.MINIMIZE)
        prob.addConstr(eye(l) @ x <= self.bigm * eye(l) @ y)
        prob.addConstr(eye(l) @ x >= -1 * self.bigm *eye(l) @ y)
        prob.addConstr(ones(l) @ y <= self.s)
        
        prob.optimize()
        self.residual_squared.append(prob.objVal + norm(self.qb,2) ** 2)
        ind = where( x.X > 1e-6)[0]
        self.indexes.append(list(ind))
        self.coefficients.append(x.X[ind])
        return [x.X,prob.objVal]  

    def gurobi_mip3(self,ind,warm_x_index,warm_x):
        l = len(ind)
        
        prob = gurobipy.Model("gurobi_mip")
        prob.setParam( 'OutputFlag',0 )
        x = prob.addMVar(shape = l,vtype = "C",name = "x")
        y = prob.addMVar(shape = l,vtype = "B",name = "y")

        for index in range(len(warm_x_index)):
            x[warm_x_index[index]].start = warm_x[index]
            y[warm_x_index[index]].start = 1
        prob.setObjective(x @ self.rtilde2[ind,:][:,ind] @ x - 2 * self.qb[:,0].T @ self.rtilde[:,ind] @ x ,gurobipy.GRB.MINIMIZE)
        # prob.setObjective(x @ self.rtilde[:,ind].T.dot(self.rtilde[:,ind]) @ x - 2 * self.qb[:,0].T @ self.rtilde[:,ind] @ x ,gurobipy.GRB.MINIMIZE)
        prob.addConstr(eye(l) @ x <= self.bigm * eye(l) @ y)
        prob.addConstr(eye(l) @ x >= -1 * self.bigm *eye(l) @ y)
        prob.addConstr(ones(l) @ y <= self.s)
        t22 = time.time()
        t00 = time.process_time()
        prob.optimize()
        t11 = time.process_time()
        t33 = time.time()
        self.real_time = t33-t22
        self.cpu_time = t11-t00
        self.residual_squared.append(prob.objVal + norm(self.qb,2) ** 2)
        ind = where( x.X > 1e-6)[0]
        self.indexes.append(list(ind))
        self.coefficients.append(x.X[ind])
        return [x.X,prob.objVal]

    def nonnegative_IHT(self,ind,x = False):
        l = len(ind)
        A = self.rtilde[:,ind]
        u,d,v = svd(A)
        step = 1/d[0] ** 2
        if x is False:
            x1 = zeros(l)
            x2 = copy(x1)
        else:
            x1 = copy(x)
            x2 = copy(x1)
        diff = 1
        t = 0
        while diff >= 1e-6 and t <= 400:
            # print("dif",diff,"time",t)
            t += 1
            grad = self.rtilde[:,ind].T.dot(self.rtilde[:,ind].dot(x1) - self.qb[:,0])
            x1 = x1 - step * grad
            
            x1[where(x1 < 0)] = 0
            si = argsort(x1)
            x1[si[:-self.s]] = 0
            diff = norm(x2-x1,2)
            x2 = x1

        return x1

    def qr_nnls(self,ind):
        l = len(ind) 
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol = nnls(self.rtilde[:t,ind],self.qb[:t,0])
        res = sol[1] ** 2 + norm(self.qb[t:]) ** 2
        return [sol[0],res]
    
    def qr_nnlsl(self,ind):
        check = str(ind)
        if check in self.tablelookup_nnls:
            return self.tablelookup_nnls[check]
        l = len(ind) 
        t = max(ind)+1
        """ since R is upper triangular, rows beyond t are 0 """
        sol = nnls(self.rtilde[:t,ind],self.qb[:t,0])
        res = sol[1] ** 2 + norm(self.qb[t:]) ** 2
        """ using QR to solve a least squares problem """
        self.tablelookup_nnls[check] = [sol[0],res]
        return [sol[0],res]

                
    def solve_mk0(self,P ,C = []):
        """ 
        
        This is a function to solve the all subsets problem with exploiting previously solved problems with table look ups.
        
        """

        L = [0]*(self.s+1)
        for i in range(self.s+1):
            L[i] = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_nnls(C+P)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        for i in range(len(L)):
            L[i].put([f[1],[P,C,f[0][lenc:],lenc,lenp]])
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            # print("Started solving for sparsity level ",self.s)
            count_best = 0
            while L[i].qsize() >= 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = L[i].get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                # print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
                if len_c < self.s:
                    if  count_nonzero(coef) <= self.s:
                        self.residual_squared.append(low)
                        index = list(where(coef > 0)[0])
                        all_variables = P+C
                        real_indexes = [all_variables[i] for i in index]
                        self.indexes.append(real_indexes)
                        self.coefficients.append(coef[index])
                        count_best += 1
                        if count_best == self.many:
                            self.rem_qsize.append(L[i].qsize())
                            break
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = (sdx+xbar)*coef[:len_p]
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef1 = copy(coef)
                        coef1[l_index_bb:-1],coef1[-1] = coef1[l_index_bb+1:],coef1[l_index_bb]
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indexes"""
                        self.node += 1
                        """ update number of nodes visited """
                        lower2 = self.qr_nnlsl(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indexes ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_nnls(C1)
                            L[i].put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            L[i].put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            L[i].put([low ,[P1,C1,coef1,len_c1,len_p]])
                            L[i].put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(L[i].qsize())
                        break
                    
    def solve_nnls(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be NNLS coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This search rule basically finds the variable with the highest magnite in the scaled version of the non negative 
        least squares if the variable is not scaled prior. This is useful because it uses much less flops thansolving a 
        parallel non negative least squares with scaled variables.

        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.qr_nnls(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            # print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  count_nonzero(coef) <= self.s:
                    self.residual_squared.append(low)
                    index = list(where(coef > 0)[0])
                    all_variables = P+C
                    real_indexes = [all_variables[i] for i in index]
                    self.indexes.append(real_indexes)
                    self.coefficients.append(coef[index])
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = (sdx+xbar)*coef[:len_p]
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added""" 
                    P1 = P[:]
                    
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_nnls(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_nnls(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                
    def solve_nnls_mk2(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the least square
        
        let a(i) be NNLS coefficient of x(i),i'th independent variable
        let xbar(i) be mean of x(i), i'th independen variable
        let varx(i) be variance of x(i), i'th independent variable
        if (|xbar(j)| + varx(j))*a(j) is the highest then j'th variable has the highest impact
        
        xbar*a is justified because of unit selection, a unit maybe chosen for x(i) such that it is 
        large in quantity,but determines something relatively smaller to it in numbers
        then it will be adjusted smaller coefficientin LSE for accurate results, other way around 
        for when unit makes x(i) small 
        
        varx*a is justified because lets say independen variable x(j), takes values from -1000 to 1000
        values of x(j) are actually big, and also important because also high a(j), but by sheer chance,
        expectation, or statistically the mean is 0 for x(j) so xbar(j)*a(j) would not catch this variable
        
        This search rule basically finds the variable with the highest magnite in the scaled version of the non negative 
        least squares if the variable is not scaled prior. This is useful because it uses much less flops thansolving a 
        parallel non negative least squares with scaled variables.
        
        """
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.qr_nnls(P+C)
        q.put([f[1],P,C,f[0][0:len(P)]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,P,C,coef] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            #print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if count_nonzero(coef) <= self.s:
                self.residual_squared.append(low)
                if P == []:
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                else:
                    index = list(where(coef > 0)[0])
                    all_variables = P+C
                    real_indexes = [all_variables[i] for i in index]
                    self.indexes.append(real_indexes)
                    self.coefficients.append(coef[index])
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
            else:
                xbar = self.means[P]
                """ xbar is a vector length len(p), it retrieves the mean for each variable """
                sdx = self.sterror[P]
                """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                bb_dec = (sdx+xbar)*coef[:len(P)]
                """ bb_dec is the decision vector, logic is explained above"""
                l_index_bb = argmax(bb_dec)
                """ find index of the largest value in decision vector """
                r_index_bb = P[l_index_bb]
                """ find index of the variable by index of the largest value in decision vector 
                this is the chosen variable for this node """ 
                C1 = C + [r_index_bb]
                coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                """ add the new chosen variable to the solution 
                We also use old C, where chosen variable is not added""" 
                P1 = P[:]
                
                del P1[l_index_bb]
                """ erasing the chosen variable from the possible variables' list 
                reminder: this stores the variables by their indexes"""
                self.node += 1
                """ update number of nodes visited """
                lower2 = self.qr_nnls(P1+C)
                """ calculate lower bound of the second solution where C is the old chosen variable 
                list and p1 is the possible variable (indexes ) list where the chosen variable for 
                this node is erased """
                if len(C1) == self.s:
                    sol = self.qr_nnls(C1)
                    q.put([sol[1],[],C1,sol[0]])
                    q.put([lower2[1],P1,C,lower2[0]])
                else:
                    """ if the length of the chosen variable list is not equal to sparsity level, 
                    then it is lower than sparsity level. We create two new nodes where first node
                    is the node where chosen variable for this node is in the solution and the second 
                    where the chosen variable for this node is erased from the problem """
                    q.put([low ,P1,C1,coef])
                    q.put([lower2[1] ,P1,C,lower2[0]])

                                
    def solve_qr_mk0(self,P ,C = []):
        q,r = linalg.qr(self.A)
        # self.q2 = q
        # self.r2 = r
        # self.rtilde2 = r[0:self.n,:]
        # self.qb2 = self.q.T.dot(self.b)
        
        # self.permanent_residual2 = norm(self.b) ** 2 - norm(self.qb) ** 2
        
        self.q = q
        self.r = r
        self.rtilde = r[0:self.n,:]
        self.qb = self.q.T.dot(self.b)
        
        self.permanent_residual = norm(self.b) ** 2 - norm(self.qb) ** 2
        
        
        """ 
        
        All subsets function with tablelookups
        
        """

        L = [0]*(self.s+1)
        for i in range(self.s+1):
            L[i] = PriorityQueue()
        """ prioirty queue for best first search  """
        f = self.qr_nnls(C+P)
        self.tqsize = []
        lenp = len(P)
        lenc = len(C)
        for i in range(len(L)):
            L[i].put([f[1],[P,C,f[0][lenc:],lenc,lenp]])
        """ initialization of the first problem """
        s = self.s
        i = 0
        while i < s:
            i += 1
            self.s = i
            print("Started solving for sparsity level ",self.s)
            count_best = 0
            while L[i].qsize() >= 1:
                """ termination condition of the problem if we visit all the nodes then search is over """
                [low,[P,C,coef,len_c,len_p]] = L[i].get()
                """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
                #print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
                if len_c < self.s:
                    if  len(where(coef > 0)[0]) <= self.s:
                        self.residual_squared.append(low)
                        index = list(where(coef > 0)[0])
                        all_variables = P+C
                        real_indexes = [all_variables[i] for i in index]
                        self.indexes.append(real_indexes)
                        self.coefficients.append(coef[index])
                        count_best += 1
                        if count_best == self.many:
                            self.rem_qsize.append(L[i].qsize())
                            break
                    else:
                        xbar = self.means[P]
                        """ xbar is a vector length len(p), it retrieves the mean for each variable """
                        sdx = self.sterror[P]
                        """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                        bb_dec = (sdx+xbar)*coef[:len_p]
                        """ bb_dec is the decision vector, logic is explained above"""
                        l_index_bb = argmax(bb_dec)
                        """ find index of the largest value in decision vector """
                        r_index_bb = P[l_index_bb]
                        """ find index of the variable by index of the largest value in decision vector 
                        this is the chosen variable for this node """ 
                        C1 = C + [r_index_bb]
                        coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                        """ add the new chosen variable to the solution 
                        We also use old C, where chosen variable is not added""" 
                        P1 = P[:]
                        del P1[l_index_bb]
                        """ erasing the chosen variable from the possible variables' list 
                        reminder: this stores the variables by their indexes"""
                        self.node += 1
                        """ update number of nodes visited """
                        lower2 =self.qr_nnlsl(P1+C)
                        """ calculate lower bound of the second solution where C is the old chosen variable 
                        list and p1 is the possible variable (indexes ) list where the chosen variable for 
                        this node is erased """
                        len_p -= 1
                        len_c1 = len_c +1 
                        if len_c1 == self.s:
                            """ fix here """
                            sol = self.qr_nnls(C1)
                            L[i].put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                            L[i].put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                        else:
                            """ if the length of the chosen variable list is not equal to sparsity level, 
                            then it is lower than sparsity level. We create two new nodes where first node
                            is the node where chosen variable for this node is in the solution and the second 
                            where the chosen variable for this node is erased from the problem """
                            
                            L[i].put([low ,[P1,C1,coef,len_c1,len_p]])
                            L[i].put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
                else:
                    self.residual_squared.append(low)
                    self.indexes.append(C)
                    self.coefficients.append(coef)
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(L[i].qsize())
                        break

    def solve_qr_nnls(self,P ,C = []):
        """ This algortihm does best first search with Branch and Bound done according to their impact 
        on the non negative least squares subproblem

        """
        q,r = linalg.qr(self.A)
        # self.q2 = q
        # self.r2 = r
        # self.rtilde2 = r[0:self.n,:]
        # self.qb2 = self.q.T.dot(self.b)
        
        # self.permanent_residual2 = norm(self.b) ** 2 - norm(self.qb) ** 2
        
        self.q = q
        self.r = r
        self.rtilde = r[0:self.n,:]
        self.qb = self.q.T.dot(self.b)
        
        self.permanent_residual = norm(self.b) ** 2 - norm(self.qb) ** 2
            
        q = PriorityQueue()
        """ prioirty queue for best first search  """
        f =  self.qr_nnls(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        count_best = 0
        """ initialization of the first problem """
        while q.qsize() > 0:
            """ termination condition of the problem if we visit all the nodes then search is over """
            [low,[P,C,coef,len_c,len_p]] = q.get()
            """ get a node from the graph, with best SSE, sum of squared error, of unconstrained problem """
            #print("lowerbound for now",low,"len of chosen",len(C),"len of possible",len(P),"last chosen",C[-1:])
            if len_c < self.s:
                if  count_nonzero(coef) <= self.s:
                    self.residual_squared.append(low)
                    index = list(where(coef > 0)[0])
                    all_variables = P+C
                    real_indexes = [all_variables[i] for i in index]
                    self.indexes.append(real_indexes)
                    self.coefficients.append(coef[index])
                    count_best += 1
                    if count_best == self.many:
                        self.rem_qsize.append(q.qsize())
                        break
                else:
                    xbar = self.means[P]
                    """ xbar is a vector length len(p), it retrieves the mean for each variable """
                    sdx = self.sterror[P]
                    """ sd is a vector of length len(p), it retrieves the standard error for each variable """
                    bb_dec = (sdx+xbar)*coef[:len_p]
                    """ bb_dec is the decision vector, logic is explained above"""
                    l_index_bb = argmax(bb_dec)
                    """ find index of the largest value in decision vector """
                    r_index_bb = P[l_index_bb]
                    """ find index of the variable by index of the largest value in decision vector 
                    this is the chosen variable for this node """ 
                    C1 = C + [r_index_bb]
                    """ add the new chosen variable to the solution 
                    We also use old C, where chosen variable is not added"""
                    coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                    P1 = P[:]
                    del P1[l_index_bb]
                    """ erasing the chosen variable from the possible variables' list 
                    reminder: this stores the variables by their indexes"""
                    self.node += 1
                    """ update number of nodes visited """
                    lower2 = self.qr_nnls(P1+C)
                    """ calculate lower bound of the second solution where C is the old chosen variable 
                    list and p1 is the possible variable (indexes ) list where the chosen variable for 
                    this node is erased """
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_nnls(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        """ if the length of the chosen variable list is not equal to sparsity level, 
                        then it is lower than sparsity level. We create two new nodes where first node
                        is the node where chosen variable for this node is in the solution and the second 
                        where the chosen variable for this node is erased from the problem """
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                self.residual_squared.append(low)
                self.indexes.append(C)
                self.coefficients.append(coef)
                count_best += 1
                if count_best == self.many:
                    self.rem_qsize.append(q.qsize())
                    break
                
    def arborescent(self,P,len_p,low,coef):
        # print("lowerbound for now",low,"len of possible",len_p)
        if P in self.searched:
            return Inf
        else:
            self.searched.append(P)
        if low >= self.best_feasible:
            return Inf
        if len_p <= self.s:
            self.best_feasible = low
            self.sol_coef = coef
            self.sol_index = list(self.order[P])
            return low
        else:
            self.node += len_p
            for i in range(len(P)):
                Pnew = P[:]
                del Pnew[i]
                newsol = self.qr_nnls(Pnew)
                self.arborescent(Pnew,len_p-1,newsol[1],newsol[0])
                
    def solve_arborescent(self):
        f = self.qr_nnls(list(range(self.n)))
        si = argsort(-1*f[0])
        self.order = si
        ns = self.rtilde[:,si]
        q,r = linalg.qr(ns)
        self.rtilde = r
        self.qb = q.T.dot(self.qb)
        
        self.arborescent(list(range(self.n)),self.n,f[1],f[0][si])
    
    
    def solve(self,C = []):
            
        if self.out not in [0,1,2]:
            print("OUT parameter should be a integer >=0  and <= 2")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            return None
        elif self.many > math.factorial(self.n)/(math.factorial(self.s)*math.factorial(self.n-self.s)):
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        
        if not type(C) == list and C != []:
            print("C should be a list, C is taken empty list now ")
            C = []
        elif C != [] and (max(C) >= self.n or min(C) < 0):
            print("Values of C should be valid, in the range 0 <= n-1, C is taken empty list")
            C = []
        elif len(set(C)) != len(C):
            print("Values of C should be unique, C is taken empty list")
            C = []
        elif len(C) > self.s:
            print(" Length of C cannot be greater than spartsity level s,C is taken empty list")
            C = []
            
            
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
        
        P = list(range(self.n))
        if C != []:
            for i in range(len(C)):
                P.remove(C[i])
        """ Preparing the P and C that will be passed to the function """
        
        """ Another if list to find and call the right function """
        
        t2 = time.process_time()
        t3 = time.time()
        self.solve_nnls(P,C)
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        # print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout

        
    def solve_allsubsets(self):
        
        """ For performance concerns, only the best algorithm is offered to use for finding all best k subsets for sparsity level
        from 1 to s, """
        

        if self.many > self.n:
            print("Reduce number of best subsets you want to find, it is greater than  or equal to all possibilities")
            return None
        elif self.n < self.s:
            print("sparsity level is higher than number of variables")
            
            return None
        mem = hpy()
        """ memory object """
        mem.heap()
        """ check the objects that are in memory right now """
        mem.setrelheap()
        """ referencing this point, memory usage will be calculated """
        t0 = time.process_time()
        """ referencing this point, cpu usage will be calculated """
        
        if self.out != 2:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout    
        """ whether user wants to print every step of the algorithm or not """
            
        P = list(range(self.n))
        C = []
        """ Preparing the P and C that will be passed to the function """
        
        
        t2 = time.process_time()
        t3 = time.time()
        self.solve_mk0(P,C)
        """ mk0 also works, but this is in general faster """
        t4 = time.time()
        finish = time.process_time()
        duration = finish-t0
        self.cpu = duration
        if self.out == 0:
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = self.original_stdout   
        print("CPU time of the algorithm",duration,"seconds")
        m = mem.heap()
        print(m)
        self.memory = m.size
        """ real memory usage is different than the number we store here because we use guppy package 
        to track down memory usage of our algorithm, tracking down memory usage is also memory extensive
        process """
         
        sys.stdout = self.original_stdout

class NMF:
    def __init__(self,A,k,s):
        for i in range(len(A)):
            for j in range(len(A[0])):
                if math.isnan(A[i,j]) or abs(A[i,j]) == Inf:
                    print("Matrix A has NAN or Inf values, it will give linear algebra errors")
                    break
        for i in range(len(A)):
            for j in range(len(A[0])):
                if type(A[i,j]) != float64:
                    print("Matrix A should be registered as float64, otherwise computations will be wrong\
for example; Variance will be negative")
        if type(A) != ndarray:
            print("A should be a numpy ndarray")
        elif type(s) != int:
            print("s should be an integer")
        elif type(k) != int:
            print("s should be an integer")
        else:
            self.A = A
            self.k = k
            self.s = s

            self.m = shape(A)[0]
            self.n = shape(A)[1]

            self.means = abs(A.mean(axis = 0))
            self.sterror = std(A,axis = 0)
            
            q,r = linalg.qr(A)
            self.q = q
            self.rtilde = r

            self.residual_squared = []
            self.indexes = []
            self.coefficients = []
            
    def qr_nnls(self,ind):
        l = len(ind) 
        t = max(ind)+1
        
        sol = nnls(self.rtilde[:t,ind],self.qb[:t,0])
        res = sol[1] ** 2 + norm(self.qb[t:]) ** 2
        return [sol[0],res]
       
    def solve_nnls(self,P ,C = []):
        q = PriorityQueue()
        f =  self.qr_nnls(P+C)
        lenp = len(P)
        q.put([f[1],[P,C,f[0][0:lenp],len(C),lenp]])
        while q.qsize() > 0:
            [low,[P,C,coef,len_c,len_p]] = q.get()
            if len_c < self.s:
                if  count_nonzero(coef) <= self.s:
                    index = list(where(coef > 0)[0])
                    all_variables = P+C
                    real_indexes = [all_variables[i] for i in index]
                    self.indexes.append(real_indexes)
                    self.coefficients.append(coef[index])
                    return real_indexes,coef[index]
                else:
                    xbar = self.means[P]
                    sdx = self.sterror[P]
                    bb_dec = (sdx+xbar)*coef[:len_p]
                    l_index_bb = argmax(bb_dec)
                    r_index_bb = P[l_index_bb]

                    C1 = C + [r_index_bb]
                    coef[l_index_bb:-1],coef[-1] = coef[l_index_bb+1:],coef[l_index_bb]
                    P1 = P[:]
                    del P1[l_index_bb]
            
                    lower2 = self.qr_nnls(P1+C)
                    len_c1 = len_c +1 
                    len_p = len_p -1
                    if len_c1 == self.s:
                        sol = self.qr_nnls(C1)
                        q.put([sol[1],[P1,C1,sol[0],len_c1,len_p]])
                        q.put([lower2[1],[P1,C,lower2[0],len_c,len_p]])
                    else:
                        q.put([low ,[P1,C1,coef,len_c1,len_p]])
                        q.put([lower2[1] ,[P1,C,lower2[0],len_c,len_p]])
            else:
                return C,coef
                
    def solve(self):
        save_s = self.s
        
        R = copy(self.rtilde)
        H = zeros([self.k,self.n])
        p = 0
        P = []
        while norm(R) > 1e-4 and p < self.s:
            # print("first p",p)
            p += 1
            j = argmax(linalg.norm(R, axis = 0))
            P.append(j)
            
            self.s = p
            
            for i in range(self.n):
                # print("i",i)
                
                self.qb = self.rtilde[:,[i]]
                
                coef,res = nnls(self.rtilde[:,P],self.rtilde[:,i])
                
                R[:,i] = self.rtilde[:,i] - self.rtilde[:,P].dot(coef)
        self.s = save_s
        while norm(R) > 1e-4 and p < self.k:
            # print("second p",p)
            p += 1
            j = argmax(linalg.norm(R, axis = 0))
            P.append(j)
            
            for i in range(self.n):
                # print("i",i)
                
                self.qb = self.rtilde[:,[i]]
                
                index,coef = self.solve_nnls(P,[])
                
                R[:,i] = self.rtilde[:,i] - self.rtilde[:,index].dot(coef)
                
        M = self.A[:,P]
        for i in range(self.n):
            self.qb = self.rtilde[:,[i]]
            index,coef = self.solve_nnls(P,[])
            Hindex = [P.index(i) for i in index]
            H[Hindex,i] = coef
        return M,H,P
    
    def solve_mk2(self):
        R = copy(self.rtilde)
        H = zeros([self.k,self.n])
        q,r,p = qr(self.A,pivoting = True)
        
        P = list(p[:self.k])
        
        M = self.A[:,P]
        for i in range(self.n):
            self.qb = self.rtilde[:,[i]]
            index,coef = self.solve_nnls(P)
            Hindex = [P.index(i) for i in index]
            H[Hindex,i] = coef
        return M,H,P
    
    def solve_mk3(self):
        R = copy(self.rtilde)
        H = zeros([self.k,self.n])
        q,r,p = qr(self.A,pivoting = True)
        
        P = list(p[:self.k])
        
        M = self.A[:,P]
        for i in range(self.n):
            self.qb = self.rtilde[:,[i]]
            coef,res = nnls(self.rtilde[:,P],self.rtilde[:,i])
            H[:,i] = coef
        return M,H,P

            