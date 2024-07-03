import sys
import dolfinx
import pyvista
from mpi4py import MPI #import parallel communicator
import numpy as np
import ufl
import matplotlib.pyplot as plt
import dolfinx.fem.petsc
from dolfinx.io import gmshio
import time
import pandas as pd
import basix

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
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "lu"},jit_options={'timeout':60})
    uh = problem.solve()
    return uh


  def solveAdjoint(self,h,c,u):
    psi = solveDirect(c,h)
    adj = dolfinx.fem.Function(self.V)
    adj.x.array[:] = - u.x.array * psi.x.array
    return adj

  def directOperator(self,c,f_list,uB=None):
    """
    Resolve o problema para cada fi na lista
    retorna lista com cada solução ui respectiva
    """
    if uB == None:
      uB = dolfinx.fem.Function(self.V)
    
    u_list = []

    for fi in f_list:
      u_list.append(self.solveDirect(c,fi,uB))

    return u_list

  def directOperatorDerivate(self,h,c,u_list):
    derivative_list = []

    for uk in u_list:
      etak = self.solveDirect(c,-h*uk)
      derivative_list.append(etak)

    return derivative_list

  def directOperatorAdjoint(self, sigma_list,c,u_list):
    adj_func_list = []

    #resolve a adjunta para cada direção sigma_i relativo ao u_i respectivo
    #armazena cada adunta numa lista
    for ui,sigma_i in zip(u_list,sigma_list):
      adj_i = self.solveAdjoint(sigma_i,c,ui)
      adj_func_list.append(adj_i)

    #cria lista com os vetores de coeficientes de cada adjunta
    adj_array_list = [adj.x.array for adj in adj_func_list]

    #cria função vazia para armazenar a adjunta
    adj_func = dolfinx.fem.Function(self.V)

    #armazena na função da adjunta a soma dos vetores de coeficientes
    adj_func.x.array[:] = np.sum(adj_array_list,axis=0)


    return adj_func


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

  def tikhonovOp(self, xk,alpha,c,u_list):
    """
    Computes (A*A + a*I)x, where A = F'(c)
    """
    #Calcula Ax = F'(c)x
    derivative_list = self.directOperatorDerivate(xk,c,u_list)

    #Calcula A*(Ax) = F'(c)^*(F'(c) x)
    adjoint = self.directOperatorAdjoint(derivative_list, c, u_list)

    #soma A*(Ax) + alpha * x
    tx = dolfinx.fem.Function(self.V)
    tx.x.array[:] = adjoint.x.array + alpha*xk.x.array

    return tx


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
