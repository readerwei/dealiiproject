/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2005 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2005, 2006
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/tensor_function.h>

namespace Step20
{
  using namespace dealii;


  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem (const unsigned int degree);
    void run ();

  private:
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve ();
    void compute_errors () const;
    void output_results () const;

    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;
    BlockVector<double>       system_rhs;
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &/*p*/,
                                    const unsigned int /*component*/) const
  {
    return 0;
  }



  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
    const double alpha = 0.3;
    const double beta = 1;
    return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
  }



  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

    const double alpha = 0.3;
    const double beta = 1;

    values(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
    values(1) = alpha*p[0]*p[1];
    values(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
  }




  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }





  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (const unsigned int degree)
    :
    degree (degree),
    fe (FE_RaviartThomas<dim>(degree), 1,
        FE_DGQ<dim>(degree), 1),
    dof_handler (triangulation)
  {}




  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs ()
  {
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (3);

    dof_handler.distribute_dofs (fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim];

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
              << std::endl;

    const unsigned int
    n_couplings = dof_handler.max_couplings_between_dofs();

    sparsity_pattern.reinit (2,2);
    sparsity_pattern.block(0,0).reinit (n_u, n_u, n_couplings);
    sparsity_pattern.block(1,0).reinit (n_p, n_u, n_couplings);
    sparsity_pattern.block(0,1).reinit (n_u, n_p, n_couplings);
    sparsity_pattern.block(1,1).reinit (n_p, n_p, n_couplings);
    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system ()
  {
    QGauss<dim>   quadrature_formula(degree+2);
    QGauss<dim-1> face_quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim>          right_hand_side;
    const PressureBoundaryValues<dim> pressure_boundary_values;
    const KInverse<dim>               k_inverse;

    std::vector<double> rhs_values (n_q_points);
    std::vector<double> boundary_values (n_face_q_points);
    std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        k_inverse.value_list (fe_values.get_quadrature_points(),
                              k_inverse_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
              const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = fe_values[pressure].value (i, q);

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = fe_values[pressure].value (j, q);

                  local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p *
                              rhs_values[q] *
                              fe_values.JxW(q);
            }

        for (unsigned int face_no=0;
             face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->at_boundary(face_no))
            {
              fe_face_values.reinit (cell, face_no);

              pressure_boundary_values
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);

              for (unsigned int q=0; q<n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  local_rhs(i) += -(fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));
            }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               local_matrix(i,j));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);
      }
  }






  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double> &A,
                     const IterativeInverse<Vector<double> > &Minv);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const IterativeInverse<Vector<double> > > m_inverse;

    mutable Vector<double> tmp1, tmp2;
  };


  SchurComplement::SchurComplement (const BlockSparseMatrix<double> &A,
                                    const IterativeInverse<Vector<double> > &Minv)
    :
    system_matrix (&A),
    m_inverse (&Minv),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void SchurComplement::vmult (Vector<double>       &dst,
                               const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    m_inverse->vmult (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }



  class ApproximateSchurComplement : public Subscriptor
  {
  public:
    ApproximateSchurComplement (const BlockSparseMatrix<double> &A);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
    void Tvmult (Vector<double>       &dst,
                 const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

    mutable Vector<double> tmp1, tmp2;
  };


  ApproximateSchurComplement::ApproximateSchurComplement (const BlockSparseMatrix<double> &A)
    :
    system_matrix (&A),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void ApproximateSchurComplement::vmult (Vector<double>       &dst,
                                          const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }


  void ApproximateSchurComplement::Tvmult (Vector<double>       &dst,
                                           const Vector<double> &src) const
  {
    vmult (dst, src);
  }




  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
    PreconditionIdentity identity;
    IterativeInverse<Vector<double> > m_inverse;
    m_inverse.initialize(system_matrix.block(0,0), identity);
    m_inverse.solver.select("cg");
    static ReductionControl inner_control(1000, 0., 1.e-13);
    m_inverse.solver.set_control(inner_control);

    Vector<double> tmp (solution.block(0).size());

    {
      Vector<double> schur_rhs (solution.block(1).size());

      m_inverse.vmult (tmp, system_rhs.block(0));
      system_matrix.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);


      SchurComplement
      schur_complement (system_matrix, m_inverse);

      ApproximateSchurComplement
      approximate_schur_complement (system_matrix);

      IterativeInverse<Vector<double> >
      preconditioner;
      preconditioner.initialize(approximate_schur_complement, identity);
      preconditioner.solver.select("cg");
      preconditioner.solver.set_control(inner_control);


      SolverControl solver_control (solution.block(1).size(),
                                    1e-12*schur_rhs.l2_norm());
      SolverCG<>    cg (solver_control);

      cg.solve (schur_complement, solution.block(1), schur_rhs,
                preconditioner);

      std::cout << solver_control.last_step()
                << " CG Schur complement iterations to obtain convergence."
                << std::endl;
    }

    {
      system_matrix.block(0,1).vmult (tmp, solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      m_inverse.vmult (solution.block(0), tmp);
    }
  }




  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors () const
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);

    ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, degree+2);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error
              << std::endl;
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
    std::vector<std::string> solution_names;
    switch (dim)
      {
      case 2:
        solution_names.push_back ("u");
        solution_names.push_back ("v");
        solution_names.push_back ("p");
        break;

      case 3:
        solution_names.push_back ("u");
        solution_names.push_back ("v");
        solution_names.push_back ("w");
        solution_names.push_back ("p");
        break;

      default:
        Assert (false, ExcNotImplemented());
      }


    DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names);

    data_out.build_patches (degree+1);

    std::ofstream output ("solution.gmv");
    data_out.write_gmv (output);
  }




  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {
    make_grid_and_dofs();
    assemble_system ();
    solve ();
    compute_errors ();
    output_results ();
  }
}



int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      deallog.depth_console (0);

      MixedLaplaceProblem<2> mixed_laplace_problem(0);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
