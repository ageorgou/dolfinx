// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScVector;
}

namespace function
{
// class Function;
class FunctionSpace;
} // namespace function

namespace io
{

/// Interface to HDF5 files

class HDF5File
{

public:
  /// Constructor. file_mode should be "a" (append),
  /// "w" (write) or "r" (read).
  HDF5File(MPI_Comm comm, const std::string filename,
           const std::string file_mode);

  /// Destructor
  ~HDF5File();

  /// Close file
  void close();

  /// Flush buffered I/O to disk
  void flush();

  /// Write points to file
  void write(const std::vector<Eigen::Vector3d>& points,
             const std::string name);

  /// Write simple vector of double to file
  void write(const std::vector<double>& values, const std::string name);

  /// Write Vector to file in a format suitable for re-reading
  void write(const la::PETScVector& x, const std::string name);

  /// Read vector from file and optionally re-use any partitioning
  /// that is available in the file
  la::PETScVector read_vector(MPI_Comm comm, const std::string dataset_name,
                              const bool use_partition_from_file) const;

  /// Write Mesh to file in a format suitable for re-reading
  void write(const mesh::Mesh& mesh, const std::string name);

  /// Write Mesh of given cell dimension to file in a format
  /// suitable for re-reading
  void write(const mesh::Mesh& mesh, const int cell_dim,
             const std::string name);

  /// Write function::Function to file in a format suitable for re-reading
  void write(const function::Function& u, const std::string name);

  /// Write function::Function to file with a timestamp
  void write(const function::Function& u, const std::string name,
             double timestamp);

  /// Read function::Function from file and distribute data according to the
  /// mesh::Mesh and dofmap associated with the function::Function.  If the
  /// 'name'
  /// refers to a HDF5 group, then it is assumed that the function::Function
  /// data is stored in the datasets within that group.  If the
  /// 'name' refers to a HDF5 dataset within a group, then it is
  /// assumed that it is a Vector, and the function::Function will be filled
  /// from that Vector
  function::Function read(std::shared_ptr<const function::FunctionSpace> V,
                          const std::string name) const;

  /// Read mesh::Mesh from file, using attribute data (e.g., cell type)
  /// stored in the HDF5 file. Optionally re-use any partition data
  /// in the file. This function requires all necessary data for
  /// constructing a mesh::Mesh to be present in the HDF5 file.
  mesh::Mesh read_mesh(const std::string data_path,
                       bool use_partition_from_file,
                       const mesh::GhostMode ghost_mode) const;

  /// Construct mesh::Mesh with paths to topology and geometry datasets,
  /// and providing essential meta-data, e.g. geometric dimension
  /// and cell type. If this data is available in the HDF5 file, it
  /// will be checked for consistency. Set expected_num_global_cells
  /// to a negative value if not known.
  ///
  /// This function is typically called when using the XDMF format,
  /// in which case the meta data has already been read from an XML
  /// file
  mesh::Mesh read_mesh(const std::string topology_path,
                       const std::string geometry_path, const int gdim,
                       const mesh::CellType cell_type,
                       const std::int64_t expected_num_global_cells,
                       const std::int64_t expected_num_global_points,
                       bool use_partition_from_file,
                       const mesh::GhostMode ghost_mode) const;

  /// Write mesh::MeshFunction to file in a format suitable for re-reading
  void write(const mesh::MeshFunction<std::size_t>& meshfunction,
             const std::string name);

  /// Write mesh::MeshFunction to file in a format suitable for re-reading
  void write(const mesh::MeshFunction<int>& meshfunction,
             const std::string name);

  /// Write mesh::MeshFunction to file in a format suitable for re-reading
  void write(const mesh::MeshFunction<double>& meshfunction,
             const std::string name);

  /// Read mesh::MeshFunction from file
  mesh::MeshFunction<std::size_t>
  read_mf_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                 const std::string name) const;

  /// Read mesh::MeshFunction from file
  mesh::MeshFunction<int> read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                                      const std::string name) const;

  /// Read mesh::MeshFunction from file
  mesh::MeshFunction<double>
  read_mf_double(std::shared_ptr<const mesh::Mesh> mesh,
                 const std::string name) const;

  /// Write mesh::MeshValueCollection to file
  void write(const mesh::MeshValueCollection<std::size_t>& mesh_values,
             const std::string name);

  /// Write mesh::MeshValueCollection to file
  void write(const mesh::MeshValueCollection<double>& mesh_values,
             const std::string name);

  /// Write mesh::MeshValueCollection to file
  void write(const mesh::MeshValueCollection<bool>& mesh_values,
             const std::string name);

  /// Read mesh::MeshValueCollection from file
  mesh::MeshValueCollection<std::size_t>
  read_mvc_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                  const std::string name) const;

  /// Read mesh::MeshValueCollection from file
  mesh::MeshValueCollection<double>
  read_mvc_double(std::shared_ptr<const mesh::Mesh> mesh,
                  const std::string name) const;

  /// Read mesh::MeshValueCollection from file
  mesh::MeshValueCollection<bool>
  read_mvc_bool(std::shared_ptr<const mesh::Mesh> mesh,
                const std::string name) const;

  /// Check if dataset exists in HDF5 file
  bool has_dataset(const std::string dataset_name) const;

  /// Set the MPI atomicity
  void set_mpi_atomicity(bool atomic);

  /// Get the MPI atomicity
  bool get_mpi_atomicity() const;

  /// Get the file ID
  hid_t h5_id() const { return _hdf5_file_id; }

  // FIXME: document
  bool chunking = false;

private:
  // Friend
  friend class XDMFFile;

  // Write a mesh::MeshFunction to file
  template <typename T>
  void write_mesh_function(const mesh::MeshFunction<T>& meshfunction,
                           const std::string name);

  // Read a mesh::MeshFunction from file
  template <typename T>
  mesh::MeshFunction<T>
  read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
                     const std::string name) const;

  // Write a mesh::MeshValueCollection to file (new version using vertex
  // indices)
  template <typename T>
  void
  write_mesh_value_collection(const mesh::MeshValueCollection<T>& mesh_values,
                              const std::string name);

  // Read a mesh::MeshValueCollection from file
  template <typename T>
  mesh::MeshValueCollection<T>
  read_mesh_value_collection(std::shared_ptr<const mesh::Mesh> mesh,
                             const std::string name) const;

  // Write contiguous data to HDF5 data set. Data is flattened into
  // a 1D array, e.g. [x0, y0, z0, x1, y1, z1] for a vector in 3D
  template <typename T>
  void write_data(const std::string dataset_name, const std::vector<T>& data,
                  const std::vector<std::int64_t> global_size, bool use_mpi_io);

  // Write 2D dataset to HDF5. Eigen::Arrays on each process must have the same
  // number of columns.
  template <typename T>
  void write_data(
      const std::string dataset_name,
      Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data,
      bool use_mpi_io);

  // HDF5 file descriptor/handle
  hid_t _hdf5_file_id;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};

//---------------------------------------------------------------------------
// Needs to go here, because of use in XDMFFile.cpp
template <typename T>
void HDF5File::write_data(const std::string dataset_name,
                          const std::vector<T>& data,
                          const std::vector<std::int64_t> global_size,
                          bool use_mpi_io)
{
  assert(_hdf5_file_id > 0);
  assert(global_size.size() > 0);

  // Get number of 'items'
  std::int64_t num_local_items = 1;
  for (std::size_t i = 1; i < global_size.size(); ++i)
    num_local_items *= global_size[i];
  num_local_items = data.size() / num_local_items;

  // Compute offset
  const std::int64_t offset
      = MPI::global_offset(_mpi_comm.comm(), num_local_items, true);
  std::array<std::int64_t, 2> range = {{offset, offset + num_local_items}};

  // Write data to HDF5 file. Ensure dataset starts with '/'.
  std::string dset_name(dataset_name);
  if (dset_name[0] != '/')
    dset_name = "/" + dataset_name;

  HDF5Interface::write_dataset(_hdf5_file_id, dset_name, data.data(), range,
                               global_size, use_mpi_io, chunking);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_data(
    const std::string dataset_name,
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data,
    bool use_mpi_io)
{
  assert(_hdf5_file_id > 0);

  // Compute offset
  const std::int64_t offset
      = MPI::global_offset(_mpi_comm.comm(), data.rows(), true);
  std::array<std::int64_t, 2> range = {{offset, offset + data.rows()}};

  // Write data to HDF5 file. Ensure dataset starts with '/'.
  std::string dset_name(dataset_name);
  if (dset_name[0] != '/')
    dset_name = "/" + dataset_name;

  std::int64_t global_rows = MPI::sum(_mpi_comm.comm(), data.rows());
  std::vector<std::int64_t> global_size = {global_rows, data.cols()};
  if (data.cols() == 1)
    global_size = {global_rows};

  HDF5Interface::write_dataset(_hdf5_file_id, dset_name, data.data(), range,
                               global_size, use_mpi_io, chunking);
}
//---------------------------------------------------------------------------
} // namespace io
} // namespace dolfin
