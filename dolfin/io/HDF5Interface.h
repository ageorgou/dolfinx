// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-09-21
// Last changed:

#ifndef __DOLFIN_HDF5_INTERFACE_H
#define __DOLFIN_HDF5_INTERFACE_H

// Use 1.6 API for stability
// Could update to latest version, whichrequires adding a few extra
// arguments to calls for little obvious benefit
#define H5_USE_16_API

#include <vector>
#include <string>
#include <hdf5.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class HDF5File;

  // FIXME: Add class description

  class HDF5Interface
  {
  #define HDF5_FAIL -1
  public:

    /// FIXME: Add description
    static void create(const std::string filename);

    /// Write data to existing HDF file as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    template <typename T>
    static void write(const std::string filename,
                      const std::string dataset_name,
                      const std::vector<T>& data,
                      const std::pair<uint, uint> range,
                      const uint width);

    /// Read data from a HDF5 dataset as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    template <typename T>
    static void read(const std::string filename,
                     const std::string dataset_name,
                     std::vector<T>& data,
                     const std::pair<uint, uint> range,
                     const uint width);

    /// Check for existence of dataset in HDF5 file
    static bool dataset_exists(const HDF5File& hdf5_file,
                               const std::string dataset_name);

    /// Return list all datasets in named group of file
    static std::vector<std::string> dataset_list(const std::string filename,
                                                 const std::string group_name);

    // FIXME: Size of dimension?
    /// Get dimensions (NX, NY) of 2D dataset
    static std::pair<uint, uint> dataset_dimensions(const std::string filename,
                                               const std::string dataset_name);

    /// Get a named attribute of a dataset
    template <typename T>
    static void get_attribute(const std::string filename,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              T &attribute_value);

    /// Add attribute to dataset
    template <typename T>
    static void add_attribute(const std::string filename,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              const T& attribute_value);

  private:

    // HDF5 calls to open a file descriptor on multiple processes
    // Common file opening sequence
    static hid_t open_parallel_file(const std::string filename);

    template <typename T>
    static void add_attribute_value(const hid_t dset_id,
                                    const std::string attribute_name,
                                    const T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "add attribute data",
                   "No specialised function fot this data type");
    }

    template <typename T>
    static void get_attribute_value(const hid_t attr_type,
                                    const hid_t attr_id,
                                    T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "gate attribute data",
                   "No specialised function fot this data type");
    }

    // Return HDF5 data type
    template <typename T>
    static int hdf5_type()
    {
      dolfin_error("HDF5Interface.cpp",
                   "get HDF5 primitive data type",
                   "No specialised function for this data type");
      return 0;
    }

  };

  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::write(const std::string filename,
                                   const std::string dataset_name,
                                   const std::vector<T>& data,
                                   const std::pair<uint, uint> range,
                                   const uint width)
  {
    // Get HDF5 data type
    const int h5type = hdf5_type<T>();

    // Hyperslab selection parameters
    const hsize_t count[2]  = {range.second - range.first, width};
    const hsize_t offset[2] = {range.first, 0};

    // Dataset dimensions
    const hsize_t dimsf[2] = {MPI::sum(count[0]), width};

    // Generic status report
    herr_t status;

    // Open file
    const hid_t file_id = open_parallel_file(filename);

    // Create a global 2D data space
    const hid_t filespace0 = H5Screate_simple(2, dimsf, NULL);
    dolfin_assert(filespace0 != HDF5_FAIL);

    // Create global dataset (using dataset_name)
    const hid_t dset_id = H5Dcreate(file_id, dataset_name.c_str(), h5type,
                                    filespace0, H5P_DEFAULT);
    dolfin_assert(dset_id != HDF5_FAIL);

    // Close global data space
    status = H5Sclose(filespace0);
    dolfin_assert(status != HDF5_FAIL);

    // Create a local 2D data space
    const hid_t memspace = H5Screate_simple(2, count, NULL);
    dolfin_assert(memspace != HDF5_FAIL);

    // Create a file dataspace within the global space - a hyperslab
    const hid_t filespace1 = H5Dget_space(dset_id);
    status = H5Sselect_hyperslab(filespace1, H5S_SELECT_SET, offset, NULL,
                                 count, NULL);
    dolfin_assert(status != HDF5_FAIL);

    // Set parallel access with communicator
    const hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    dolfin_assert(status != HDF5_FAIL);

    // Write local dataset into selected hyperslab
    status = H5Dwrite(dset_id, h5type, memspace, filespace1, plist_id,
                      data.data());
    dolfin_assert(status != HDF5_FAIL);

    // close dataset collectively
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);

    // close hyperslab
    status = H5Sclose(filespace1);
    dolfin_assert(status != HDF5_FAIL);

    // close local dataset
    status = H5Sclose(memspace);
    dolfin_assert(status != HDF5_FAIL);

    // Release file-access template
    status = H5Pclose(plist_id);
    dolfin_assert(status != HDF5_FAIL);

    // Close file
    status = H5Fclose(file_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::read(const std::string filename,
                                  const std::string dataset_name,
                                  std::vector<T>& data,
                                  const std::pair<uint, uint> range,
                                  const uint width)
  {
    // Get HDF5 data type
    const int h5type = hdf5_type<T>();

    // Resize data
    data.resize(width*(range.second - range.first));

    // Generic return value
    herr_t status;

    // Hyperslab selection
    hsize_t offset[2] = {range.first, 0};
    hsize_t count[2]  = {range.second - range.first, width};

    // Open file descriptor in parallel
    const hid_t file_id = open_parallel_file(filename);

    // Open the dataset collectively
    const hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
    dolfin_assert(dset_id != HDF5_FAIL);

    // Create a file dataspace independently
    const hid_t filespace = H5Dget_space (dset_id);
    dolfin_assert(filespace != HDF5_FAIL);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                                 count, NULL);
    dolfin_assert(status != HDF5_FAIL);

    // Create a memory dataspace independently
    const hid_t memspace = H5Screate_simple (2, count, NULL);
    dolfin_assert (memspace != HDF5_FAIL);

    // Read data independently
    status = H5Dread(dset_id, h5type, memspace, filespace,
                     H5P_DEFAULT, data.data());
    dolfin_assert(status != HDF5_FAIL);

    // Close dataset collectively
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);

    // Release all IDs created
    status = H5Sclose(filespace);
    dolfin_assert(status != HDF5_FAIL);

    // Close the file collectively
    status = H5Fclose(file_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::add_attribute(const std::string filename,
                                           const std::string dataset_name,
                                           const std::string attribute_name,
                                           const T& attribute_value)
  {
    // Open file
    hid_t file_id = open_parallel_file(filename);

    // Open named dataset
    hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
    dolfin_assert(dset_id != HDF5_FAIL);

    // Add attribute of appropriate type
    add_attribute_value(dset_id, attribute_name, attribute_value);

    herr_t status;

    // Close dataset
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);

    // Close file
    status = H5Fclose(file_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const uint& attribute_value)
  {
    // Create a scalar dataspace
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create attribute of type uint
    hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_UINT,
                                   dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_UINT, &attribute_value);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                        const std::string attribute_name,
                                        const std::vector<uint>& attribute_value)
  {
    // Create a vector dataspace
    const hsize_t dimsf = attribute_value.size();
    const hid_t dataspace_id = H5Screate_simple(1, &dimsf, NULL);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create an attribute of type uint in the dataspace
    const hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_UINT, dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_UINT, &attribute_value[0]);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const std::string& attribute_value)
  {
    // Create a scalar dataspace
    const hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Copy basic string type from HDF5 types and set string length
    const hid_t datatype_id = H5Tcopy(H5T_C_S1);
    herr_t status = H5Tset_size(datatype_id, attribute_value.size());
    dolfin_assert(status != HDF5_FAIL);

    // Create attribute in the dataspace with the given string
    const hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(), datatype_id,
                                    dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    status = H5Awrite(attribute_id, datatype_id, attribute_value.c_str());
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          uint& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // Read value
    herr_t status = H5Aread(attr_id, H5T_NATIVE_UINT, &attribute_value);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          std::string& attribute_value)
  {
    // Check this attribute is a string
    dolfin_assert(H5Tget_class(attr_type) == H5T_STRING);

    // Copy string type from HDF5 types and set length accordingly
    const hid_t memtype = H5Tcopy(H5T_C_S1);
    const int string_length = H5Tget_size(attr_type) + 1;
    herr_t status = H5Tset_size(memtype,string_length);
    dolfin_assert(status != HDF5_FAIL);

    // FIXME: messy
    // Copy string value into temporary vector
    // std::vector::data can be copied into
    // (std::string::data cannot)
    std::vector<char> attribute_data(string_length);
    status = H5Aread(attr_id, memtype, attribute_data.data());
    dolfin_assert(status != HDF5_FAIL);

    attribute_value.assign(attribute_data.data());
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          std::vector<uint>& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // get dimensions of attribute array, check it is one-dimensional
    const hid_t dataspace = H5Aget_space(attr_id);
    dolfin_assert(dataspace != HDF5_FAIL);

    hsize_t cur_size[10];
    hsize_t max_size[10];
    const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);
    dolfin_assert(ndims == 1);

    attribute_value.resize(cur_size[0]);

    // Read value to vector
    herr_t status = H5Aread(attr_id, H5T_NATIVE_UINT, attribute_value.data());
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <>
  inline int HDF5Interface::hdf5_type<double>()
  { return H5T_NATIVE_DOUBLE; }
  //-----------------------------------------------------------------------------
  template <>
  inline int HDF5Interface::hdf5_type<int>()
  { return H5T_NATIVE_INT; }
  //-----------------------------------------------------------------------------
  template <>
  inline int HDF5Interface::hdf5_type<dolfin::uint>()
  { return H5T_NATIVE_UINT; }
  //-----------------------------------------------------------------------------

}

#endif