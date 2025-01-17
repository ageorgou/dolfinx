// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Connectivity.h"
#include "Mesh.h"
#include "Topology.h"

namespace dolfin
{

namespace mesh
{
/// A MeshEntity represents a mesh entity associated with a specific
/// topological dimension of some _Mesh_.

class MeshEntity
{
public:
  /// Constructor
  ///
  /// @param   mesh (_Mesh_)
  ///         The mesh.
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  /// @param     index (std::size_t)
  ///         The index.
  MeshEntity(const Mesh& mesh, int dim, std::int32_t index)
      : _mesh(&mesh), _dim(dim), _local_index(index)
  {
    // Do nothing
  }

  /// Copy constructor
  MeshEntity(const MeshEntity& e) = default;

  /// Move constructor
  MeshEntity(MeshEntity&& e) = default;

  /// Destructor
  ~MeshEntity() = default;

  /// Assignement operator
  MeshEntity& operator=(const MeshEntity& e) = default;

  /// Move assignement operator
  MeshEntity& operator=(MeshEntity&& e) = default;

  /// Comparison Operator
  ///
  /// @param e (MeshEntity)
  ///         Another mesh entity
  ///
  ///  @return    bool
  ///         True if the two mesh entities are equal.
  bool operator==(const MeshEntity& e) const
  {
    return (_mesh == e._mesh and _dim == e._dim
            and _local_index == e._local_index);
  }

  /// Comparison Operator
  ///
  /// @param e (MeshEntity)
  ///         Another mesh entity.
  ///
  /// @return     bool
  ///         True if the two mesh entities are NOT equal.
  bool operator!=(const MeshEntity& e) const { return !operator==(e); }

  /// Return mesh associated with mesh entity
  ///
  /// @return Mesh
  ///         The mesh.
  const Mesh& mesh() const { return *_mesh; }

  /// Return topological dimension
  ///
  /// @return     std::size_t
  ///         The dimension.
  int dim() const { return _dim; }

  /// Return index of mesh entity
  ///
  /// @return     std::size_t
  ///         The index.
  std::int32_t index() const { return _local_index; }

  /// Return array of indices for incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  ///         The index for incident mesh entities of given dimension.
  const std::int32_t* entities(int dim) const
  {
    if (dim == _dim)
      return &_local_index;
    else
    {
      assert(_mesh->topology().connectivity(_dim, dim));
      const std::int32_t* initialized_mesh_entities
          = _mesh->topology().connectivity(_dim, dim)->connections(
              _local_index);
      assert(initialized_mesh_entities);
      return initialized_mesh_entities;
    }
  }

  /// Compute local index of given incident entity (error if not
  /// found)
  ///
  /// @param     entity (_MeshEntity_)
  ///         The mesh entity.
  ///
  /// @return     std::size_t
  ///         The local index of given entity.
  int index(const MeshEntity& entity) const;

  /// Return informal string representation (pretty-print)
  ///
  /// @param      verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return      std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

protected:
  template <typename T>
  friend class MeshRange;
  template <typename T>
  friend class EntityRange;
  template <typename T>
  friend class MeshIterator;
  template <typename T>
  friend class MeshEntityIterator;

  // The mesh
  Mesh const* _mesh;

  // Topological dimension
  int _dim;

  // Local index of entity within topological dimension
  std::int32_t _local_index;
};

/// A Vertex is a MeshEntity of topological dimension 0.
class Vertex : public MeshEntity
{
public:
  /// Create vertex on given mesh
  Vertex(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 0, index) {}
};

/// An Edge is a MeshEntity of topological dimension 1.
class Edge : public MeshEntity
{
public:
  /// Create edge on given mesh
  ///
  /// @param    mesh (_Mesh_)
  ///         The mesh.
  /// @param    index (std::size_t)
  ///         Index of the edge.
  Edge(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 1, index) {}
};

/// A Face is a MeshEntity of topological dimension 2.
class Face : public MeshEntity
{
public:
  /// Create face on given mesh
  Face(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 2, index) {}
};

/// A Facet is a MeshEntity of topological codimension 1.
class Facet : public MeshEntity
{
public:
  /// Constructor
  Facet(const Mesh& mesh, std::int32_t index)
      : MeshEntity(mesh, mesh.topology().dim() - 1, index)
  {
  }
};

/// A Cell is a MeshEntity of topological codimension 0.
class Cell : public MeshEntity
{
public:
  /// Create cell on given mesh with given index
  ///
  /// @param    mesh
  ///         The mesh.
  /// @param    index
  ///         The index.
  Cell(const Mesh& mesh, std::int32_t index)
      : MeshEntity(mesh, mesh.topology().dim(), index)
  {
  }
};

} // namespace mesh
} // namespace dolfin
