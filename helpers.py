import sys
import dolfinx
import pyvista
from mpi4py import MPI #import parallel communicator
import numpy as np
import ufl
import matplotlib.pyplot as plt
import dolfinx.fem.petsc
import time
import pandas as pd
import basix
from petsc4py import PETSc

class DirectProblem:
  def __init__(self,N):
    """
    Params:
    - N: int, number of nodes in each square line
    """

    self.N = N
    self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
    Ve = basix.ufl.element('Lagrange', "triangle", degree=1, shape=())
    self.V = dolfinx.fem.functionspace(self.mesh, Ve) #Continuous Garlekin de grau 1 - funções afim em cada triângulo
    self.V_array_size = self.V.dofmap.index_map.size_global

    #for setting boundary conditions
    self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
    self.boundary_dofs = dolfinx.fem.locate_dofs_topological(self.V, self.mesh.topology.dim-1, boundary_facets)

    self.u_topology, self.u_cell_types, self.u_geometry = dolfinx.plot.vtk_mesh(self.V)

    #gera matriz de indices ordenados para coordenadas
    coord = abs(self.V.tabulate_dof_coordinates()[:,:2])
    idx = np.argsort(coord[:,0])
    idx_list = np.split(idx,N+1)
    idy_list = [np.argsort(coord[idx_list[i]][:,1]) for i in range(N+1)]
    self.M_id = np.array([idx_list[i][idy_list[i]] for i in range(N+1)])

    # KSP solver
    self.ksp = PETSc.KSP()
    self.ksp.create(comm=self.mesh.comm)
    self.ksp.setType(PETSc.KSP.Type.CG)
    self.ksp.getPC().setType(PETSc.PC.Type.LU)    
    #
    self.assembled = False

  def solveDirect(self, c, f, uB = None):
    if uB==None:
      uB = dolfinx.fem.Function(self.V)

    #colocando condições de Dirichlet
    bc = dolfinx.fem.dirichletbc(uB, self.boundary_dofs)

    u = ufl.TrialFunction(self.V)
    v = ufl.TestFunction(self.V)
    a = ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx + ufl.inner(c*u,v)*ufl.dx
    L = ufl.inner(f,v)*ufl.dx

    #Resolvendo
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg"},jit_options={'timeout':100})
    uh = problem.solve()
    return uh


  def solveAdjoint(self,h,c,u):
    psi = self.solveDirect(c,h)
    adj = dolfinx.fem.Function(self.V)
    adj.x.array[:] = - u.x.array * psi.x.array
    return adj

  def setProblem(self,c,bc):
    u = ufl.TrialFunction(self.V)
    v = ufl.TestFunction(self.V)

    a = ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx + ufl.inner(c*u,v)*ufl.dx
    a_form = dolfinx.fem.form(a)

    A = dolfinx.fem.petsc.assemble_matrix(a_form,bcs=[bc])
    self.ksp.setOperators(A)
    self.ksp.setFromOptions()
    A.assemble()

    return a_form
    
  def directOperator(self,c,f_list,uB=None):
    """
    Resolve o problema para cada fi na lista
    retorna lista com cada solução ui respectiva
    """
    if uB == None:
      uB = dolfinx.fem.Function(self.V)
    
    bc = dolfinx.fem.dirichletbc(uB, self.boundary_dofs)
    
    u_list = []

    v = ufl.TestFunction(self.V)
    a_form = self.setProblem(c,bc)

    for fi in f_list:
      #setting right hand side vector
      rhs = ufl.inner(fi,v)*ufl.dx
      b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(rhs))
      dolfinx.fem.petsc.apply_lifting(b, [a_form], [[bc]])
      dolfinx.fem.petsc.set_bc(b,bcs=[bc])
      
      #solving
      ui = dolfinx.fem.Function(self.V)
      self.ksp.solve(b,ui.vector)
      u_list.append(ui)

    return u_list

  def directOperatorDerivate(self,h,c,u_list):
    rhs_list = [-h*uk for uk in u_list]
    derivative_list = self.directOperator(c,rhs_list)
    return derivative_list

  def createFiList(self,n_grid_size, r, f_value=None):
    N = self.N
    index_array = np.linspace(0,(N+1),(n_grid_size)+2,dtype=int)[1:-1]
    index_grid = np.dstack(np.meshgrid(index_array, index_array)).reshape(-1, 2)

    f_list = [dolfinx.fem.Function(self.V) for i in range(n_grid_size**2)]
    # for i in range(n_grid_size**2):
    #   fi = f_list[i]
    #   index_i = self.M_id[index_grid[i][0]][index_grid[i][1]]
    #   fi.x.array[index_i] = f_value
    coord_array = self.V.tabulate_dof_coordinates().T

    for i in range(n_grid_size):
      for j in range(n_grid_size):
        circle_locator = get_circle_locator((1+i)/(n_grid_size+1),(1+j)/(n_grid_size+1), r)
        fi = f_list[i*n_grid_size+j]
        fi.x.array[:] = circle_locator(coord_array)
        if f_value==None:
          fi_norm = funcSquareNorm(fi)**0.5
          fi.x.array[:] = fi.x.array[:]/fi_norm
        else:
          fi.x.array[:] = fi.x.array * f_value

    return f_list

  def directOperatorAdjoint(self, sigma_list,c,u_list):
    #resolve a adjunta para cada direção sigma_i relativo ao u_i respectivo
    #armazena cada adunta numa lista
    psi_list = self.directOperator(c,sigma_list)
    adj_array = np.zeros(self.V_array_size)
    for ui,psi in zip(u_list,psi_list):      
      adj_array += - ui.x.array * psi.x.array
    adj = dolfinx.fem.Function(self.V)
    adj.x.array[:] = adj_array
    return adj

  def plotFunc(self, u, warped=False):
    pyvista.start_xvfb()
    u_grid = pyvista.UnstructuredGrid(self.u_topology, self.u_cell_types, self.u_geometry)
    u_grid.point_data["u"] = u.x.array
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter(notebook=True)

    if warped:
        warped = u_grid.warp_by_scalar()
        u_plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    else:
        u_plotter.add_mesh(u_grid, show_edges=True)
        u_plotter.view_xy()

    if not pyvista.OFF_SCREEN:
        u_plotter.show()
    if pyvista.OFF_SCREEN:
        figure = p.screenshot("disk.png")
      
    u_plotter.close()

class InverseProblem:

  def __init__(self, problem: DirectProblem):
    self.problem = problem
    self.V = self.problem.V

  def conjugateGradient(self, A: callable,b,x0=None,tol=1e-6,maxit=10):
    if x0==None:
      x0 = dolfinx.fem.Function(self.V)
    
    # Initialize xk vector
    xk = dolfinx.fem.Function(self.V)

    # Initialize residual vector
    residual = dolfinx.fem.Function(self.V)
    residual.x.array[:] = b.x.array - A(x0).x.array

    # Initialize search direction vector
    search_direction = residual

    # Compute initial squared residual norm
    old_resid_norm = funcSquareNorm(residual)**0.5
    xk.x.array[:] = x0.x.array
    # Iterate until convergence
    nit = 0
    while old_resid_norm > tol and nit<maxit:
      A_search_direction = A(search_direction)
      step_size = old_resid_norm**2 /(funcProduct(search_direction, A_search_direction))
      # Update solution
      xk.x.array[:] = xk.x.array + step_size * search_direction.x.array
      # Update residual
      residual.x.array[:] = residual.x.array - step_size * A_search_direction.x.array
      new_resid_norm = funcSquareNorm(residual)**0.5
      # Update search direction vector
      search_direction.x.array[:] = residual.x.array +  (new_resid_norm / old_resid_norm)**2 * search_direction.x.array

      # Update squared residual norm for next iteration
      old_resid_norm = new_resid_norm
      print(old_resid_norm)
      nit += 1
    return xk

  def tikhonovOp(self,xk,alpha,c,u_list):
    """
    Computes (A*A + a*I)x, where A = F'(c)
    """
    #Calcula Ax = F'(c)x
    derivative_list = self.problem.directOperatorDerivate(xk,c,u_list)

    #Calcula A*(Ax) = F'(c)^*(F'(c) x)
    adjoint = self.problem.directOperatorAdjoint(derivative_list, c, u_list)

    #soma A*(Ax) + alpha * x
    tx = dolfinx.fem.Function(self.V)
    tx.x.array[:] = adjoint.x.array + alpha*xk.x.array

    return tx

  # 1 lado direito
  def inLM(self, u_list, f_list, c, uB=None, tau=1.1,delta=0, lmbda=0.1, alpha=1, lmbda_mult=1, n_iter=100):
    """
    Função para médodo do Levenberg-Marquardt inercial

    Args:
      u_list:     list of dolfinx.fem.Function, with (u_i) = F(c).
      f_list:     list of dolfinx.fem.Function, with the rhs fi functions
      c:          dolfinx.fem.Function, solution
      tau:        float, discrepancy parameter
      delta:      float, noise level 
      lmbda:      float, step parameter (A^*A + lmbda * I) (default 0.1).
      alpha:      float, inertial wk = ck + alpha *(c_k - c_(k-1)) (default 1.0).
      lmbda_mult: float, update at each step lmbda *= lmbda_mult (default 1.0).
      n_iter:     int, max number of iterations (default 100).

    Returns:
      ck_sol:         dolfinx.fem.Function(V), approximated solution
      kdelta:     int, discrepancy index (if not reached, returns -1)
      err_norm_array:   error norm array at each iteration
      err_norm_array:   residual norm array at each iteration


    """
    V = self.problem.V

    L = len(u_list)

    c0 = dolfinx.fem.Function(V)
    residual_list = [dolfinx.fem.Function(V) for i in range(L)]
    c_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(c,c) * ufl.dx))**0.5
    err_norm_array = np.zeros(n_iter+1)
    res_norm_array = np.zeros(n_iter+1)
    ck = c0
    bk = [dolfinx.fem.Function(V) for i in range(L)]
    
    err_func = dolfinx.fem.Function(V)
    ck_old = dolfinx.fem.Function(V)
    wk = dolfinx.fem.Function(V)
    ck_sol = dolfinx.fem.Function(V)

    kdelta = -1

    for i in range(n_iter):
      err_func.x.array[:] = ck.x.array-c.x.array
      err_norm_array[i] = funcSquareNorm(err_func)**0.5
      uk_list = self.problem.directOperator(ck,f_list,uB)
      res_sum = 0

      wk.x.array[:] = ck.x.array + alpha*(ck.x.array - ck_old.x.array)

      Fwk_list = self.problem.directOperator(wk, f_list,uB)
      #calcula residuo e bk
      for k in range(L):
        residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
        res_sum += funcSquareNorm(residual_list[k])

        bk[k].x.array[:] = u_list[k].x.array - Fwk_list[k].x.array
      res_norm_array[i] = (res_sum/L)**0.5

      if kdelta==-1 and res_norm_array[i]<=tau*delta:
        ck_sol.x.array[:] = ck.x.array
        kdelta = i

      #calcula (Ak)^* bk
      adj_bk = self.problem.directOperatorAdjoint(bk, wk, Fwk_list)

      #define ((Ak)^*Ak + lmbda I)
      Ak = lambda x: self.tikhonovOp(x, lmbda, wk, Fwk_list)

      #calcula sk
      sk = self.conjugateGradient(Ak,adj_bk,dolfinx.fem.Function(V),tol=1e-10,maxit=3)

      #atualiza ck
      ck_old.x.array[:] = ck.x.array
      ck.x.array[:] = wk.x.array + sk.x.array

      lmbda *= lmbda_mult

    uk_list = self.problem.directOperator(ck,f_list,uB)
    res_sum = 0
    for k in range(L):
      residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
      res_sum += funcSquareNorm(residual_list[k])
    res_norm_array[-1] = (res_sum/L)**0.5

    err_func.x.array[:] = ck.x.array-c.x.array
    err_norm_array[-1] = funcSquareNorm(err_func)**0.5

    if kdelta==-1:
      ck_sol.x.array[:] = ck.x.array

    return ck_sol, kdelta, err_norm_array, res_norm_array

  def inLW(self, u_list, f_list, c, uB=None, tau=1.1,delta=0, alpha=0,lmbda=1,lmbda_mult=1,n_iter=100,eta=0.5):
      """
      Função para médodo do Landweber inercial

      Args:
        u_list:     list of dolfinx.fem.Function, with (u_i) = F(c).
        f_list:     list of dolfinx.fem.Function, with the rhs fi functions
        c:          dolfinx.fem.Function, solution
        tau:        float, discrepancy parameter
        delta:      float, noise level 
        lmbda:      float, step parameter (A^*A + lmbda * I) (default 0.1).
        alpha:      float, inertial wk = ck + alpha *(c_k - c_(k-1)) (default 1.0).
        lmbda_mult: float, update at each step lmbda *= lmbda_mult (default 1.0).
        n_iter:     int, max number of iterations (default 100).

      Returns:
        ck_sol:         dolfinx.fem.Function(V), approximated solution
        kdelta:     int, discrepancy index (if not reached, returns -1)
        err_norm_array:   error norm array at each iteration
        err_norm_array:   residual norm array at each iteration


      """
      V = self.problem.V
      L = len(u_list)

      c0 = dolfinx.fem.Function(V)
      residual_list = [dolfinx.fem.Function(V) for i in range(L)]
      c_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(c,c) * ufl.dx))**0.5
      u_norm = directOperatorNorm(u_list)
      err_norm_array = np.zeros(n_iter+1)
      res_norm_array = np.zeros(n_iter+1)
      ck = c0
      bk = [dolfinx.fem.Function(V) for i in range(L)]
      
      err_func = dolfinx.fem.Function(V)
      ck_old = dolfinx.fem.Function(V)
      wk = dolfinx.fem.Function(V)
      ck_sol = dolfinx.fem.Function(V)

      kdelta = -1

      for i in range(n_iter):
        err_func.x.array[:] = ck.x.array-c.x.array
        err_norm_array[i] = funcSquareNorm(err_func)**0.5
        uk_list = self.problem.directOperator(ck,f_list,uB)
        res_sum = 0

        wk.x.array[:] = ck.x.array + alpha*(ck.x.array - ck_old.x.array)

        Fwk_list = self.problem.directOperator(wk, f_list,uB)

        #calcula residuo e bk
        for k in range(L):
          residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
          res_sum += funcSquareNorm(residual_list[k])

          bk[k].x.array[:] = u_list[k].x.array - Fwk_list[k].x.array
        res_norm_array[i] = (res_sum/L)**0.5

        if kdelta==-1 and res_norm_array[i]<=tau*(delta/100)*u_norm:
          ck_sol.x.array[:] = ck.x.array
          kdelta = i
  
        #calcula (Ak)^* bk
        adj_bk = self.problem.directOperatorAdjoint(bk, wk, Fwk_list)
        #calcula sk
        if lmbda_mult=='ME':
          adj_norm_sq = funcSquareNorm(adj_bk)
          res_norm_sq = 0
          for bki in bk:
            res_norm_sq += funcSquareNorm(bki)
          step = ((1-eta)*res_norm_sq)/(adj_norm_sq) #regular o Eta
        else:
          step = lmbda
        sk_array = step*adj_bk.x.array

        #atualiza ck
        ck_old.x.array[:] = ck.x.array
        ck.x.array[:] = wk.x.array + sk_array


      uk_list = self.problem.directOperator(ck,f_list,uB)
      res_sum = 0
      for k in range(L):
        residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
        res_sum += funcSquareNorm(residual_list[k])
      res_norm_array[-1] = (res_sum/L)**0.5

      err_func.x.array[:] = ck.x.array-c.x.array
      err_norm_array[-1] = funcSquareNorm(err_func)**0.5

      if kdelta==-1:
        ck_sol.x.array[:] = ck.x.array

      return ck_sol, kdelta, err_norm_array, res_norm_array

  def stocLW(self, u_list, f_list, c, uB=None, tau=1.1,delta=0, alpha=0,n_iter=100):
      """
      Função para médodo do Landweber estocástico

      Args:
        u_list:     list of dolfinx.fem.Function, with (u_i) = F(c).
        f_list:     list of dolfinx.fem.Function, with the rhs fi functions
        c:          dolfinx.fem.Function, solution
        tau:        float, discrepancy parameter
        delta:      float, noise level 
        lmbda:      float, step parameter (A^*A + lmbda * I) (default 0.1).
        alpha:      float, inertial wk = ck + alpha *(c_k - c_(k-1)) (default 1.0).
        lmbda_mult: float, update at each step lmbda *= lmbda_mult (default 1.0).
        n_iter:     int, max number of iterations (default 100).

      Returns:
        ck_sol:         dolfinx.fem.Function(V), approximated solution
        kdelta:     int, discrepancy index (if not reached, returns -1)
        err_norm_array:   error norm array at each iteration
        err_norm_array:   residual norm array at each iteration


      """
      V = self.problem.V
      L = len(u_list)

      c0 = dolfinx.fem.Function(V)
      residual_list = [dolfinx.fem.Function(V) for i in range(L)]
      c_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(c,c) * ufl.dx))**0.5
      err_norm_array = np.zeros(n_iter+1)
      res_norm_array = np.zeros(n_iter+1)
      ck = c0
      bk = [dolfinx.fem.Function(V) for i in range(L)]
      
      err_func = dolfinx.fem.Function(V)
      ck_old = dolfinx.fem.Function(V)
      wk = dolfinx.fem.Function(V)
      ck_sol = dolfinx.fem.Function(V)

      kdelta = -1

      for i in range(n_iter):
        err_func.x.array[:] = ck.x.array-c.x.array
        err_norm_array[i] = funcSquareNorm(err_func)**0.5
        uk_list = self.problem.directOperator(ck,f_list,uB)
        res_sum = 0

        wk.x.array[:] = ck.x.array + alpha*(ck.x.array - ck_old.x.array)

        Fwk_list = self.problem.directOperator(wk, f_list,uB)

        #calcula residuo
        for k in range(L):
          residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
          res_sum += funcSquareNorm(residual_list[k])

          bk[k].x.array[:] = u_list[k].x.array - Fwk_list[k].x.array
        res_norm_array[i] = (res_sum/L)**0.5

        if kdelta==-1 and res_norm_array[i]<=tau*delta:
          ck_sol.x.array[:] = ck.x.array
          kdelta = i

        #escolhe indice j
        #calcula (Aj)^* bj
        j = np.random.randint(0,L)
        adj_bk = self.problem.directOperatorAdjoint(bk[j:j+1], wk, Fwk_list[j:j+1])

        #calcula sk
        adj_norm_sq = funcSquareNorm(adj_bk)
        res_norm_sq = 0
        for bki in bk[j:j+1]:
          res_norm_sq += funcSquareNorm(bki)
        
        if adj_norm_sq>0:
          step = ((1-0.9)*res_norm_sq)/(adj_norm_sq)
        else:
          step = 100
        sk_array = step*adj_bk.x.array

        #atualiza ck
        ck_old.x.array[:] = ck.x.array
        ck.x.array[:] = wk.x.array + sk_array


      uk_list = self.problem.directOperator(ck,f_list,uB)
      res_sum = 0
      for k in range(L):
        residual_list[k].x.array[:] = uk_list[k].x.array - u_list[k].x.array
        res_sum += funcSquareNorm(residual_list[k])
      res_norm_array[-1] = (res_sum/L)**0.5

      err_func.x.array[:] = ck.x.array-c.x.array
      err_norm_array[-1] = funcSquareNorm(err_func)**0.5

      if kdelta==-1:
        ck_sol.x.array[:] = ck.x.array

      return ck_sol, kdelta, err_norm_array, res_norm_array

  def addNoise(self,u_list,perc):
    u_delta_list = []
    for u in u_list:
        noise_vec = np.random.uniform(-1,1,self.problem.V_array_size)
        u_delta = dolfinx.fem.Function(self.problem.V)
        u_delta.x.array[:] = u.x.array + u.x.array*(perc/100)*noise_vec
        u_delta_list.append(u_delta)

    return u_delta_list

  def computeDelta(self, u_list, u_list_delta):
    L = len(u_list)
    err_func_list = [dolfinx.fem.Function(self.problem.V) for i in range(L)]
    for u,u_delta, err in zip(u_list,u_list_delta,err_func_list):
      err.x.array[:] = u.x.array - u_delta.x.array

    delta = directOperatorNorm(err_func_list)
    return delta

def directOperatorNorm(u_list):
  L = len(u_list)
  norm_sum = 0
  for k in range(L):
    norm_sum += funcSquareNorm(u_list[k])
  norm = (norm_sum/L)**0.5

  return norm

def funcProduct(u,v):
  product = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,v) * ufl.dx))
  return product

def funcSquareNorm(u):
  return dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,u) * ufl.dx))

def directOperatorProduct(list_u,list_v):
  sum = 0
  for uk, vk in zip(list_u,list_v):
    sum+= funcProduct(uk,vk)

  return sum

def err_residual_graph(err_array,res_array,c,u_list,tau=None,delta=None,kdelta=None):
  c_norm = funcSquareNorm(c)**0.5
  u_norm = directOperatorNorm(u_list)

  fig, ax = plt.subplots(1,2, figsize=(15,5))
  ax[0].plot(err_array/c_norm,label=f'L={len(u_list)}')
  if kdelta!=None and kdelta!=-1:
    ax[0].plot([kdelta],[err_array[kdelta]/c_norm],linestyle ='',marker='.',markersize=5,label='$k_\delta$')
  ax[0].set_title("Erro Relativo")
  ax[0].set_ylabel("$e_k = \|c_k - c\|_{L^2}$")
  ax[0].set_xlabel("k")
  ax[0].legend()

  ax[1].plot(res_array/u_norm,label=f'L={len(u_list)}')
  ax[1].set_title("Resíduo")
  ax[1].set_ylabel("$r_k = \|F(c_k) - F(c)\|_{L^2(\Omega)}$")
  ax[1].set_xlabel("k")
  if tau!=None and delta!=None:
    ax[1].axhline(y=tau*delta/u_norm,label='$\\tau \delta$',color='orange')
  ax[1].legend()
  plt.show()

def err_residual_graph_list(err_array_list,res_array_list,c,u_lists,tau=None,delta_list=[],kdelta_list=[]):
  c_norm = funcSquareNorm(c)**0.5

  fig, ax = plt.subplots(1,2, figsize=(15,5))

  for  err_array, res_array, u_list in zip(err_array_list,res_array_list,u_lists):
    u_norm = directOperatorNorm(u_list)
    ax[0].plot(err_array/c_norm,label=f'L={len(u_list)}')
    # if kdelta!=None and kdelta!=-1:
    #   ax[0].plot([kdelta],[err_array[kdelta]/c_norm],linestyle ='',marker='.',markersize=5,label='$k_\delta$')
    
    ax[1].plot(res_array/u_norm,label=f'L={len(u_list)}')

  ax[0].set_title("Erro Relativo")
  ax[0].set_ylabel("$e_k = \|c_k - c\|_{L^2}$")
  ax[0].set_xlabel("k")
  ax[0].legend()

  ax[1].set_title("Resíduo")
  ax[1].set_ylabel("$r_k = \|F(c_k) - F(c)\|_{L^2(\Omega)}$")
  ax[1].set_xlabel("k")
  ax[1].legend()
  plt.show()



def plotFunc(u,warped=False):
  V = u.function_space
  u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
  pyvista.start_xvfb()
  u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
  u_plotter = pyvista.Plotter(notebook=True)

  u_grid.point_data["u"] = u.x.array
  u_grid.set_active_scalars("u")
  
  if warped:
    warped = u_grid.warp_by_scalar()
    u_plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
  else:
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()

  # if not pyvista.OFF_SCREEN:
  u_plotter.show()
  # if pyvista.OFF_SCREEN:
  #     figure = p.screenshot("disk.png")

  u_plotter.close()


def plotFuncList(u_list,warped=False):
  V = u_list[0].function_space
  u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
  pyvista.start_xvfb()
  u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
  u_plotter = pyvista.Plotter(notebook=True)
  
  for u in u_list:
    u_grid.point_data["u"] = u.x.array
    u_grid.set_active_scalars("u")
    
    if warped:
      warped = u_grid.warp_by_scalar()
      u_plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    else:
      u_plotter.add_mesh(u_grid, show_edges=True)
      u_plotter.view_xy()

    # if not pyvista.OFF_SCREEN:
    u_plotter.show()
    # if pyvista.OFF_SCREEN:
    #     figure = p.screenshot("disk.png")

  u_plotter.close()

def get_circle_locator(cx,cy,r):
  return lambda x: ((x[0]-cx)**2 + (x[1]-cy)**2 <= r**2).astype(int)
