// Copyright (C) 2005-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "RectangleMesh.h"
#include <Eigen/Dense>
#include <cfloat>
#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Partitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

namespace
{
//-----------------------------------------------------------------------------
mesh::Mesh build_tri(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                     std::array<std::size_t, 2> n,
                     const mesh::GhostMode ghost_mode, std::string diagonal)
{
  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 2);
    EigenRowArrayXXi64 topo(0, 3);
    return mesh::Partitioning::build_distributed_mesh(
        comm, mesh::CellType::triangle, geom, topo, {}, ghost_mode);
  }

  // Check options
  if (diagonal != "left" && diagonal != "right" && diagonal != "right/left"
      && diagonal != "left/right" && diagonal != "crossed")
  {
    std::runtime_error("Unknown mesh diagonal definition.");
  }

  const Eigen::Vector3d& p0 = p[0];
  const Eigen::Vector3d& p1 = p[1];

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);

  if (std::abs(x0 - x1) < DBL_EPSILON || std::abs(y0 - y1) < DBL_EPSILON)
  {
    throw std::runtime_error("Rectangle seems to have zero width, height or "
                             "depth. Check dimensions");
  }

  if (nx < 1 || ny < 1)
  {
    throw std::runtime_error(
        "Rectangle has non-positive number of vertices in some dimension: "
        "number of vertices must be at least 1 in each dimension");
  }

  // Create vertices and cells
  std::size_t nv, nc;
  if (diagonal == "crossed")
  {
    nv = (nx + 1) * (ny + 1) + nx * ny;
    nc = 4 * nx * ny;
  }
  else
  {
    nv = (nx + 1) * (ny + 1);
    nc = 2 * nx * ny;
  }

  EigenRowArrayXXd geom(nv, 2);
  EigenRowArrayXXi64 topo(nc, 3);

  // Create main vertices
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    const double x1 = c + cd * static_cast<double>(iy);
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      geom(vertex, 0) = a + ab * static_cast<double>(ix);
      geom(vertex, 1) = x1;
      ++vertex;
    }
  }

  // Create midpoint vertices if the mesh type is crossed
  if (diagonal == "crossed")
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      const double x1 = c + cd * (static_cast<double>(iy) + 0.5);
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        geom(vertex, 0) = a + ab * (static_cast<double>(ix) + 0.5);
        geom(vertex, 1) = x1;
        ++vertex;
      }
    }
  }

  // Create triangles
  std::size_t cell = 0;
  if (diagonal == "crossed")
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t vmid = (nx + 1) * (ny + 1) + iy * nx + ix;

        // Note that v0 < v1 < v2 < v3 < vmid.
        topo.row(cell) << v0, v1, vmid;
        ++cell;
        topo.row(cell) << v0, v2, vmid;
        ++cell;
        topo.row(cell) << v1, v3, vmid;
        ++cell;
        topo.row(cell) << v2, v3, vmid;
        ++cell;
      }
    }
  }
  else if (diagonal == "left" || diagonal == "right" || diagonal == "right/left"
           || diagonal == "left/right")
  {
    std::string local_diagonal = diagonal;
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      // Set up alternating diagonal
      if (diagonal == "right/left")
      {
        if (iy % 2)
          local_diagonal = "right";
        else
          local_diagonal = "left";
      }
      if (diagonal == "left/right")
      {
        if (iy % 2)
          local_diagonal = "left";
        else
          local_diagonal = "right";
      }

      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        std::vector<std::size_t> cell_data;

        if (local_diagonal == "left")
        {
          topo.row(cell) << v0, v1, v2;
          ++cell;
          topo.row(cell) << v1, v2, v3;
          ++cell;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "right";
        }
        else
        {
          topo.row(cell) << v0, v1, v3;
          ++cell;
          topo.row(cell) << v0, v2, v3;
          ++cell;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "left";
        }
      }
    }
  }

  return mesh::Partitioning::build_distributed_mesh(
      comm, mesh::CellType::triangle, geom, topo, {}, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh build_quad(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                      std::array<std::size_t, 2> n,
                      const mesh::GhostMode ghost_mode)
{
  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 2);
    EigenRowArrayXXi64 topo(0, 4);
    return mesh::Partitioning::build_distributed_mesh(
        comm, mesh::CellType::quadrilateral, geom, topo, {}, ghost_mode);
  }

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  EigenRowArrayXXd geom((nx + 1) * (ny + 1), 2);
  EigenRowArrayXXi64 topo(nx * ny, 4);

  const double a = p[0][0];
  const double b = p[1][0];
  const double ab = (b - a) / static_cast<double>(nx);

  const double c = p[0][1];
  const double d = p[1][1];
  const double cd = (d - c) / static_cast<double>(ny);

  // Create vertices
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    double x1 = c + cd * static_cast<double>(iy);
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      geom(vertex, 0) = a + ab * static_cast<double>(ix);
      geom(vertex, 1) = x1;
      ++vertex;
    }
  }

  // Create rectangles
  std::size_t cell = 0;
  for (std::size_t iy = 0; iy < ny; iy++)
    for (std::size_t ix = 0; ix < nx; ix++)
    {
      const std::size_t i0 = iy * (nx + 1);
      topo(cell, 0) = i0 + ix;
      topo(cell, 1) = i0 + ix + 1;
      topo(cell, 2) = i0 + ix + nx + 1;
      topo(cell, 3) = i0 + ix + nx + 2;
      ++cell;
    }

  return mesh::Partitioning::build_distributed_mesh(
      comm, mesh::CellType::quadrilateral, geom, topo, {}, ghost_mode);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh RectangleMesh::create(MPI_Comm comm,
                                 const std::array<Eigen::Vector3d, 2>& p,
                                 std::array<std::size_t, 2> n,
                                 mesh::CellType cell_type,
                                 const mesh::GhostMode ghost_mode,
                                 std::string diagonal)
{
  if (cell_type == mesh::CellType::triangle)
    return build_tri(comm, p, n, ghost_mode, diagonal);
  else if (cell_type == mesh::CellType::quadrilateral)
    return build_quad(comm, p, n, ghost_mode);
  else
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");

  // Will never reach this point
  return build_quad(comm, p, n, ghost_mode);
}
//-----------------------------------------------------------------------------
