// Copyright (C) 2011-2014 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedMeshTools.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshIterator.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/Timer.h"
#include "dolfin/graph/Graph.h"
#include "dolfin/graph/SCOTCH.h"
#include <Eigen/Dense>
#include <complex>
#include <dolfin/common/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
namespace
{
typedef std::vector<std::size_t> Entity;

// Data structure for mesh entity data
struct EntityData
{
  // Constructor
  EntityData() : local_index(0) {}

  // Move constructor
  EntityData(EntityData&&) = default;

  // Move assignment
  EntityData& operator=(EntityData&&) = default;

  // Constructor  (index is local)
  explicit EntityData(std::int32_t index) : local_index(index) {}

  // Constructor (index is local)
  EntityData(std::int32_t index, const std::vector<std::int32_t>& procs)
      : local_index(index), processes(procs)
  {
    // Do nothing
  }

  // Constructor  (index is local)
  EntityData(std::int32_t index, std::int32_t process)
      : local_index(index), processes(1, process)
  {
    // Do nothing
  }

  // Local (this process) entity index
  std::int32_t local_index;

  // Processes on which entity resides
  std::vector<std::int32_t> processes;
};
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reorder_values_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  dolfin::common::Timer t("DistributedMeshTools: reorder values");

  // Number of items to redistribute
  const std::size_t num_local_indices = global_indices.size();
  assert(num_local_indices == (std::size_t)values.rows());

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
      = dolfin::MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                                     global_indices.end()))
        + 1;

  // Send unwanted values off process
  const std::size_t mpi_size = dolfin::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> indices_to_send(mpi_size);
  std::vector<std::vector<T>> values_to_send(mpi_size);

  // Go through local vector and append value to the appropriate list to
  // send to correct process
  for (std::size_t i = 0; i != num_local_indices; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i
        = dolfin::MPI::index_owner(mpi_comm, global_i, global_vector_size);
    indices_to_send[process_i].push_back(global_i);
    values_to_send[process_i].insert(values_to_send[process_i].end(),
                                     values.row(i).data(),
                                     values.row(i).data() + values.cols());
  }

  // Redistribute the values to the appropriate process - including
  // self. All values are "in the air" at this point. Receive into flat
  // arrays.
  std::vector<std::size_t> received_indices;
  std::vector<T> received_values;
  dolfin::MPI::all_to_all(mpi_comm, indices_to_send, received_indices);
  dolfin::MPI::all_to_all(mpi_comm, values_to_send, received_values);

  // Map over received values as Eigen array
  assert(received_indices.size() * values.cols() == received_values.size());
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      received_values_array(received_values.data(), received_indices.size(),
                            values.cols());

  // Create array for new data. Note that any indices which are not
  // received will be uninitialised.
  const std::array<std::int64_t, 2> range
      = dolfin::MPI::local_range(mpi_comm, global_vector_size);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_values(
      range[1] - range[0], values.cols());

  // Go through received data in descending order, and place in local
  // partition of the global vector. Any duplicate data (with same
  // index) will be overwritten by values from the lowest rank process.
  for (std::int32_t j = received_indices.size() - 1; j >= 0; --j)
  {
    const std::int64_t global_i = received_indices[j];
    assert(global_i >= range[0] && global_i < range[1]);
    new_values.row(global_i - range[0]) = received_values_array.row(j);
  }

  return new_values;
}
//-----------------------------------------------------------------------------
// Compute and return (number of global entities, process offset)
std::pair<std::size_t, std::size_t> compute_num_global_entities(
    const MPI_Comm mpi_comm, std::size_t num_local_entities,
    std::size_t num_processes, std::size_t process_number)
{
  // Communicate number of local entities
  std::vector<std::size_t> num_entities_to_number;
  dolfin::MPI::all_gather(mpi_comm, num_local_entities, num_entities_to_number);

  // Compute offset
  const std::size_t offset = std::accumulate(
      num_entities_to_number.begin(),
      num_entities_to_number.begin() + process_number, (std::size_t)0);

  // Compute number of global entities
  const std::size_t num_global
      = std::accumulate(num_entities_to_number.begin(),
                        num_entities_to_number.end(), (std::size_t)0);

  return {num_global, offset};
}
//-----------------------------------------------------------------------------
// Check if all entity vertices are the shared vertices in overlap
bool is_shared(
    const Entity& entity,
    const std::map<std::size_t, std::set<std::int32_t>>& shared_vertices)
{
  // Iterate over entity vertices
  for (auto e = entity.cbegin(); e != entity.cend(); ++e)
  {
    // Return false if an entity vertex is not in the list (map) of
    // shared entities
    if (shared_vertices.find(*e) == shared_vertices.end())
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
// Build preliminary 'guess' of shared entities. This function does not
// involve any inter-process communication. Returns (owned_entities,
// entity_ownership).
std::pair<std::vector<std::size_t>, std::array<std::map<Entity, EntityData>, 2>>
compute_preliminary_entity_ownership(
    const MPI_Comm mpi_comm,
    const std::map<std::size_t, std::set<std::int32_t>>& shared_vertices,
    const std::map<Entity, std::int32_t>& entities)
{
  // Create  maps
  std::vector<std::size_t> owned_entities;
  std::array<std::map<Entity, EntityData>, 2> shared_entities;

  // Entities
  std::map<Entity, EntityData>& owned_shared_entities = shared_entities[0];
  std::map<Entity, EntityData>& unowned_shared_entities = shared_entities[1];

  // Get my process number
  const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);

  // Iterate over all local entities
  for (auto it = entities.cbegin(); it != entities.cend(); ++it)
  {
    const Entity& entity = it->first;
    const std::size_t local_entity_index = it->second;

    // Compute which processes entity is shared with
    std::vector<std::int32_t> entity_processes;
    if (is_shared(entity, shared_vertices))
    {
      // Processes sharing first vertex of entity
      std::vector<std::size_t> intersection(
          shared_vertices.find(entity[0])->second.begin(),
          shared_vertices.find(entity[0])->second.end());
      std::vector<std::size_t>::iterator intersection_end = intersection.end();

      // Loop over entity vertices
      for (std::size_t i = 1; i < entity.size(); ++i)
      {
        // Global vertex index
        const std::size_t v = entity[i];

        // Sharing processes
        const std::set<std::int32_t>& shared_vertices_v
            = shared_vertices.find(v)->second;

        intersection_end = std::set_intersection(
            intersection.begin(), intersection_end, shared_vertices_v.begin(),
            shared_vertices_v.end(), intersection.begin());
      }
      entity_processes
          = std::vector<std::int32_t>(intersection.begin(), intersection_end);
    }

    // Check if entity is master, slave or shared but not owned (shared
    // with lower ranked process)
    bool shared_but_not_owned = false;
    for (std::size_t i = 0; i < entity_processes.size(); ++i)
    {
      if (entity_processes[i] < process_number)
      {
        shared_but_not_owned = true;
        break;
      }
    }

    if (entity_processes.empty())
    {
      owned_entities.push_back(local_entity_index);
    }
    else if (shared_but_not_owned)
    {
      unowned_shared_entities[entity]
          = EntityData(local_entity_index, entity_processes);
    }
    else
    {
      owned_shared_entities[entity]
          = EntityData(local_entity_index, entity_processes);
    }
  }

  return {std::move(owned_entities), std::move(shared_entities)};
}
//-----------------------------------------------------------------------------
// Communicate with other processes to finalise entity ownership
void compute_final_entity_ownership(
    const MPI_Comm mpi_comm, std::vector<std::size_t>& owned_entities,
    std::array<std::map<Entity, EntityData>, 2>& shared_entities)
{
  // Entities ([entity vertices], index) to be numbered
  std::map<Entity, EntityData>& owned_shared_entities = shared_entities[0];
  std::map<Entity, EntityData>& unowned_shared_entities = shared_entities[1];

  // Get MPI number of processes and process number
  const std::int32_t num_processes = dolfin::MPI::size(mpi_comm);
  const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);

  // Communicate common entities, starting with the entities we think
  // are shared but not owned
  std::vector<std::vector<std::size_t>> send_common_entity_values(
      num_processes);
  for (auto it = unowned_shared_entities.cbegin();
       it != unowned_shared_entities.cend(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it->first;

    // Get entity processes (processes that might share the entity)
    const std::vector<std::int32_t>& entity_processes = it->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const std::size_t p = entity_processes[j];
      send_common_entity_values[p].push_back(entity.size());
      send_common_entity_values[p].insert(send_common_entity_values[p].end(),
                                          entity.begin(), entity.end());
    }
  }

  // Communicate common entities, add the entities we think are owned
  // and shared
  for (auto it = owned_shared_entities.cbegin();
       it != owned_shared_entities.cend(); ++it)
  {
    // Get entity vertices (global vertex indices)
    const Entity& entity = it->first;

    // Get entity processes (processes that might share the entity)
    const std::vector<std::int32_t>& entity_processes = it->second.processes;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      const std::int32_t p = entity_processes[j];
      assert(process_number < p);
      send_common_entity_values[p].push_back(entity.size());
      send_common_entity_values[p].insert(send_common_entity_values[p].end(),
                                          entity.begin(), entity.end());
    }
  }

  // Communicate common entities
  std::vector<std::vector<std::size_t>> received_common_entity_values;
  dolfin::MPI::all_to_all(mpi_comm, send_common_entity_values,
                          received_common_entity_values);

  // Check if entities received are really entities
  std::vector<std::vector<std::size_t>> send_is_entity_values(num_processes);
  for (std::int32_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_common_entity_values[p].size();)
    {
      // Get entity
      const std::size_t entity_size = received_common_entity_values[p][i++];
      Entity entity;
      for (std::size_t j = 0; j < entity_size; ++j)
        entity.push_back(received_common_entity_values[p][i++]);

      // Check if received really is an entity on this process (in which
      // case it will be in owned or unowned entities)
      bool is_entity = false;
      if (unowned_shared_entities.find(entity) != unowned_shared_entities.end()
          || owned_shared_entities.find(entity) != owned_shared_entities.end())
      {
        is_entity = true;
      }

      // Add information about entity (whether it's actually an entity)
      // to send to other processes
      send_is_entity_values[p].push_back(entity_size);
      for (std::size_t j = 0; j < entity_size; ++j)
        send_is_entity_values[p].push_back(entity[j]);
      send_is_entity_values[p].push_back(is_entity);
    }
  }

  // Send data back (list of requested entities that are really
  // entities)
  std::vector<std::vector<std::size_t>> received_is_entity_values;
  dolfin::MPI::all_to_all(mpi_comm, send_is_entity_values,
                          received_is_entity_values);

  // Create map from entities to processes where it is an entity
  std::map<Entity, std::vector<std::int32_t>> entity_processes;
  for (std::int32_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_is_entity_values[p].size();)
    {
      const std::size_t entity_size = received_is_entity_values[p][i++];
      Entity entity;
      for (std::size_t j = 0; j < entity_size; ++j)
        entity.push_back(received_is_entity_values[p][i++]);
      const std::size_t is_entity = received_is_entity_values[p][i++];
      if (is_entity == 1)
      {
        // Add entity since it is actually an entity for process p
        entity_processes[entity].push_back(p);
      }
    }
  }

  // Fix the list of entities we do not own (numbered by lower ranked
  // process)
  std::vector<std::vector<std::size_t>> unignore_entities;
  std::map<Entity, EntityData>::iterator entity;
  for (entity = unowned_shared_entities.begin();
       entity != unowned_shared_entities.end(); ++entity)
  {
    const Entity& entity_vertices = entity->first;
    EntityData& entity_data = entity->second;
    const std::int32_t local_entity_index = entity_data.local_index;
    if (entity_processes.find(entity_vertices) != entity_processes.end())
    {
      const std::vector<std::int32_t>& common_processes
          = entity_processes[entity_vertices];
      assert(!common_processes.empty());
      const std::int32_t min_proc = *(
          std::min_element(common_processes.begin(), common_processes.end()));

      if (process_number < min_proc)
      {
        // Move from unowned to owned
        owned_shared_entities[entity_vertices]
            = EntityData(local_entity_index, common_processes);

        // Add entity to list of entities that should be removed from
        // the unowned entity list.
        unignore_entities.push_back(entity_vertices);
      }
      else
        entity_data.processes = common_processes;
    }
    else
    {
      // Move from unowned to owned exclusively
      owned_entities.push_back(local_entity_index);

      // Add entity to list of entities that should be removed from the
      // shared but not owned entity list
      unignore_entities.push_back(entity_vertices);
    }
  }

  // Remove unowned shared entities that should not be shared
  for (std::size_t i = 0; i < unignore_entities.size(); ++i)
    unowned_shared_entities.erase(unignore_entities[i]);

  // Fix the list of entities we share
  std::vector<std::vector<std::size_t>> unshare_entities;
  for (std::map<Entity, EntityData>::iterator it
       = owned_shared_entities.begin();
       it != owned_shared_entities.end(); ++it)
  {
    const Entity& e = it->first;
    const std::int32_t local_entity_index = it->second.local_index;
    if (entity_processes.find(e) == entity_processes.end())
    {
      // Move from shared to owned elusively
      owned_entities.push_back(local_entity_index);
      unshare_entities.push_back(e);
    }
    else
    {
      // Update processor list of shared entities
      it->second.processes = entity_processes[e];
    }
  }

  // Remove shared entities that should not be shared
  for (std::size_t i = 0; i < unshare_entities.size(); ++i)
    owned_shared_entities.erase(unshare_entities[i]);
}
//-----------------------------------------------------------------------------
// Compute ownership of entities ([entity vertices], data)
//  [0]: owned exclusively (will be numbered by this process)
//  [1]: owned and shared (will be numbered by this process, and number
//       communicated to other processes)
//  [2]: not owned but shared (will be numbered by another process,
//       and number communicated to this processes)
//  Returns (owned_entities,  shared_entities)
std::pair<std::vector<std::size_t>, std::array<std::map<Entity, EntityData>, 2>>
compute_entity_ownership(
    const MPI_Comm mpi_comm,
    const std::map<std::vector<std::size_t>, std::int32_t>& entities,
    const std::map<std::int32_t, std::set<std::int32_t>>& shared_vertices_local,
    const std::vector<std::int64_t>& global_vertex_indices, std::size_t d)

{
  LOG(INFO) << "Compute ownership for mesh entities of dimension " << d;
  common::Timer timer("Compute mesh entity ownership");

  // Build global-to-local indices map for shared vertices
  std::map<std::size_t, std::set<std::int32_t>> shared_vertices;
  for (auto v = shared_vertices_local.cbegin();
       v != shared_vertices_local.cend(); ++v)
  {
    //    assert(v->first < (int)global_vertex_indices.size());
    shared_vertices.insert({global_vertex_indices[v->first], v->second});
  }

  // Entity ownership list ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)

  // Compute preliminary ownership lists (shared_entities) without
  // communication
  std::vector<std::size_t> owned_entities;
  std::array<std::map<Entity, EntityData>, 2> shared_entities;
  std::tie(owned_entities, shared_entities)
      = compute_preliminary_entity_ownership(mpi_comm, shared_vertices,
                                             entities);

  // Qualify boundary entities. We need to find out if the shared
  // (shared with lower ranked process) entities are entities of a
  // lower ranked process.  If not, this process becomes the lower
  // ranked process for the entity in question, and is therefore
  // responsible for communicating values to the higher ranked
  // processes (if any).
  compute_final_entity_ownership(mpi_comm, owned_entities, shared_entities);

  return {std::move(owned_entities), std::move(shared_entities)};
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
void DistributedMeshTools::number_entities(const Mesh& mesh, int d)
{
  common::Timer timer("Number distributed mesh entities");

  // Return if global entity indices have already been calculated
  if (mesh.topology().have_global_indices(d))
    return;

  // Const-cast to allow data to be attached
  Mesh& _mesh = const_cast<Mesh&>(mesh);

  if (dolfin::MPI::size(mesh.mpi_comm()) == 1)
  {
    // Set global entity numbers in mesh
    mesh.create_entities(d);
    _mesh.topology().set_num_entities_global(d, mesh.num_entities(d));
    std::vector<std::int64_t> global_indices(mesh.num_entities(d), 0);
    std::iota(global_indices.begin(), global_indices.end(), 0);
    _mesh.topology().set_global_indices(d, global_indices);
    return;
  }

  // Get shared entities map
  std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = _mesh.topology().shared_entities(d);

  // Number entities
  // std::vector<std::int64_t> global_entity_indices;
  // const std::map<std::int32_t, std::pair<std::int32_t, std::int32_t>>
  //     slave_entities;
  // const std::size_t num_global_entities = number_entities(
  //     mesh, slave_entities, global_entity_indices, shared_entities, d);
  std::vector<std::int64_t> global_entity_indices;
  const std::map<std::int32_t, std::pair<std::int32_t, std::int32_t>>
      slave_entities;
  std::size_t num_global_entities;
  std::tie(global_entity_indices, shared_entities, num_global_entities)
      = number_entities(mesh, slave_entities, d);

  // Set global entity numbers in mesh
  _mesh.topology().set_num_entities_global(d, num_global_entities);
  _mesh.topology().set_global_indices(d, global_entity_indices);
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::int64_t>,
           std::map<std::int32_t, std::set<std::int32_t>>, std::size_t>
DistributedMeshTools::number_entities(
    const Mesh& mesh,
    const std::map<std::int32_t, std::pair<std::int32_t, std::int32_t>>&
        slave_entities,
    int d)
{
  // Developer note: This function should use global_vertex_indices for
  // the global mesh indices and *not* access these through the mesh. In
  // some cases special numbering is passed in which differs from mesh
  // global numbering, e.g. when computing mesh entity numbering for
  // problems with periodic boundary conditions.

  LOG(INFO)
      << "Number mesh entities for distributed mesh (for specified vertex ids)."
      << d;
  common::Timer timer(
      "Number mesh entities for distributed mesh (for specified vertex ids)");

  std::vector<std::int64_t> global_entity_indices;
  std::map<std::int32_t, std::set<std::int32_t>> shared_entities;

  // Check that we're not re-numbering vertices (these are fixed at mesh
  // construction)
  if (d == 0)
  {
    throw std::runtime_error(
        "Global vertex indices exist at input. Cannot be renumbered");
  }

  // Check that we're not re-numbering cells (these are fixed at mesh
  // construction)
  if (d == mesh.topology().dim())
  {
    shared_entities.clear();
    global_entity_indices = mesh.topology().global_indices(d);
    return std::make_tuple(std::move(global_entity_indices),
                           std::move(shared_entities),
                           mesh.num_entities_global(d));
  }

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get number of processes and process number
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);

  // Initialize entities of dimension d locally
  mesh.create_entities(d);

  // Build list of slave entities to exclude from ownership computation
  std::vector<bool> exclude(mesh.num_entities(d), false);
  for (auto s = slave_entities.cbegin(); s != slave_entities.cend(); ++s)
    exclude[s->first] = true;

  // Build entity global [vertex list]-to-[local entity index] map.
  // Exclude any slave entities.
  std::map<std::vector<std::size_t>, std::int32_t> entities;
  std::pair<std::vector<std::size_t>, std::int32_t> entity;
  const auto& global_vertices = mesh.topology().global_indices(0);
  for (auto& e : mesh::MeshRange<MeshEntity>(mesh, d, mesh::MeshRangeType::ALL))
  {
    const std::size_t local_index = e.index();
    if (!exclude[local_index])
    {
      entity.second = local_index;
      entity.first = std::vector<std::size_t>();
      for (auto& vertex : EntityRange<Vertex>(e))
        entity.first.push_back(global_vertices[vertex.index()]);
      std::sort(entity.first.begin(), entity.first.end());
      entities.insert(entity);
    }
  }

  // Get vertex global indices
  const std::vector<std::int64_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  // Get shared vertices (local index, [sharing processes])
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_vertices_local
      = mesh.topology().shared_entities(0);

  // Compute ownership of entities of dimension d ([entity vertices], data):
  //  [0]: owned and shared (will be numbered by this process, and number
  //       communicated to other processes)
  //  [1]: not owned but shared (will be numbered by another process, and number
  //       communicated to this processes)
  std::array<std::map<Entity, EntityData>, 2> entity_ownership;
  std::vector<std::size_t> owned_entities;
  std::tie(owned_entities, entity_ownership) = compute_entity_ownership(
      mpi_comm, entities, shared_vertices_local, global_vertex_indices, d);

  // Split shared entities for convenience
  const std::map<Entity, EntityData>& owned_shared_entities
      = entity_ownership[0];
  std::map<Entity, EntityData>& unowned_shared_entities = entity_ownership[1];

  // Number of entities 'owned' by this process
  const std::size_t num_local_entities
      = owned_entities.size() + owned_shared_entities.size();

  // Compute global number of entities and local process offset
  const std::pair<std::size_t, std::size_t> num_global_entities
      = compute_num_global_entities(mpi_comm, num_local_entities, num_processes,
                                    process_number);

  // Extract offset
  std::size_t offset = num_global_entities.second;

  // Prepare list of global entity numbers. Check later that nothing is
  // equal to -1
  global_entity_indices = std::vector<std::int64_t>(mesh.num_entities(d), -1);

  // Number exclusively owned entities
  for (std::size_t i = 0; i < owned_entities.size(); ++i)
    global_entity_indices[owned_entities[i]] = offset++;

  // Number shared entities that this process is responsible for
  // numbering
  for (auto it1 = owned_shared_entities.cbegin();
       it1 != owned_shared_entities.cend(); ++it1)
  {
    global_entity_indices[it1->second.local_index] = offset++;
  }

  // Communicate indices for shared entities (owned by this process) and
  // get indices for shared but not owned entities
  std::vector<std::vector<std::size_t>> send_values(num_processes);
  std::vector<std::size_t> destinations;
  for (auto it1 = owned_shared_entities.cbegin();
       it1 != owned_shared_entities.cend(); ++it1)
  {
    // Get entity index
    const std::int32_t local_entity_index = it1->second.local_index;
    const std::int64_t global_entity_index
        = global_entity_indices[local_entity_index];
    assert(global_entity_index != -1);

    // Get entity processes (processes sharing the entity)
    const std::vector<std::int32_t>& entity_processes = it1->second.processes;

    // Get entity vertices (global vertex indices)
    const Entity& e = it1->first;

    // Prepare data for sending
    for (std::size_t j = 0; j < entity_processes.size(); ++j)
    {
      // Store interleaved: entity index, number of vertices, global
      // vertex indices
      std::size_t p = entity_processes[j];
      send_values[p].push_back(global_entity_index);
      send_values[p].push_back(e.size());
      send_values[p].insert(send_values[p].end(), e.begin(), e.end());
    }
  }

  // Send data
  std::vector<std::vector<std::size_t>> received_values;
  MPI::all_to_all(mpi_comm, send_values, received_values);

  // Fill in global entity indices received from lower ranked processes
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_values[p].size();)
    {
      const std::size_t global_index = received_values[p][i++];
      const std::size_t entity_size = received_values[p][i++];
      Entity e;
      for (std::size_t j = 0; j < entity_size; ++j)
        e.push_back(received_values[p][i++]);

      // Access unowned entity data
      std::map<Entity, EntityData>::const_iterator recv_entity
          = unowned_shared_entities.find(e);

      // Sanity check, should not receive an entity we don't need
      if (recv_entity == unowned_shared_entities.end())
      {
        std::stringstream msg;
        msg << "Process " << MPI::rank(mpi_comm)
            << " received illegal entity given by ";
        msg << " with global index " << global_index;
        msg << " from process " << p;
        throw std::runtime_error(msg.str());
      }

      const std::size_t local_entity_index = recv_entity->second.local_index;
      assert(global_entity_indices[local_entity_index] == -1);
      global_entity_indices[local_entity_index] = global_index;
    }
  }

  // Get slave indices from master
  {
    std::vector<std::vector<std::size_t>> slave_send_buffer(
        MPI::size(mpi_comm));
    std::vector<std::vector<std::size_t>> local_slave_index(
        MPI::size(mpi_comm));
    for (auto s = slave_entities.cbegin(); s != slave_entities.cend(); ++s)
    {
      // Local index on remote process
      slave_send_buffer[s->second.first].push_back(s->second.second);

      // Local index on this
      local_slave_index[s->second.first].push_back(s->first);
    }
    std::vector<std::vector<std::size_t>> slave_receive_buffer;
    MPI::all_to_all(mpi_comm, slave_send_buffer, slave_receive_buffer);

    // Send back master indices
    for (std::size_t p = 0; p < slave_receive_buffer.size(); ++p)
    {
      slave_send_buffer[p].clear();
      for (std::size_t i = 0; i < slave_receive_buffer[p].size(); ++i)
      {
        const std::size_t local_master = slave_receive_buffer[p][i];
        slave_send_buffer[p].push_back(global_entity_indices[local_master]);
      }
    }
    MPI::all_to_all(mpi_comm, slave_send_buffer, slave_receive_buffer);

    // Set slave indices to received master indices
    for (std::size_t p = 0; p < slave_receive_buffer.size(); ++p)
    {
      for (std::size_t i = 0; i < slave_receive_buffer[p].size(); ++i)
      {
        const std::size_t slave_index = local_slave_index[p][i];
        global_entity_indices[slave_index] = slave_receive_buffer[p][i];
      }
    }
  }

  // Sanity check
  for (std::size_t i = 0; i < global_entity_indices.size(); ++i)
  {
    assert(global_entity_indices[i] != -1);
  }

  // Build shared_entities (global index, [sharing processes])
  shared_entities.clear();
  for (auto e = owned_shared_entities.cbegin();
       e != owned_shared_entities.cend(); ++e)
  {
    const EntityData& ed = e->second;
    shared_entities[ed.local_index]
        = std::set<std::int32_t>(ed.processes.begin(), ed.processes.end());
  }
  for (auto e = unowned_shared_entities.cbegin();
       e != unowned_shared_entities.cend(); ++e)
  {
    const EntityData& ed = e->second;
    shared_entities[ed.local_index]
        = std::set<std::int32_t>(ed.processes.begin(), ed.processes.end());
  }

  // Return
  return std::make_tuple(std::move(global_entity_indices),
                         std::move(shared_entities), num_global_entities.first);
}
//-----------------------------------------------------------------------------
std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
DistributedMeshTools::locate_off_process_entities(
    const std::vector<std::size_t>& entity_indices, std::size_t dim,
    const Mesh& mesh)
{
  common::Timer timer("Locate off-process entities");

  if (dim == 0)
  {
    LOG(WARNING)
        << "DistributedMeshTools::host_processes has not been tested for "
           "vertices.";
  }

  // Mesh topology dim
  const std::size_t D = mesh.topology().dim();

  // Check that entity is a vertex or a cell
  if (dim != 0 && dim != D)
  {
    throw std::runtime_error(
        "This version of DistributedMeshTools::host_processes is only for "
        "vertices or cells");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(dim)
      or !mesh.topology().have_global_indices(D))
  {
    throw std::runtime_error(
        "Global mesh entity numbers have not been computed");
  }

  // Get global cell entity indices on this process
  const std::vector<std::int64_t> global_entity_indices
      = mesh.topology().global_indices(dim);

  assert((std::int64_t)global_entity_indices.size() == mesh.num_entities(D));

  // Prepare map to hold process numbers
  std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
      processes;

  // FIXME: work on optimising below code

  // List of indices to send
  std::vector<std::size_t> my_entities;

  // Remove local cells from my_entities to reduce communication
  if (dim == D)
  {
    // In order to fill vector my_entities...
    // build and populate a local set for non-local cells
    std::set<std::size_t> set_of_my_entities(entity_indices.begin(),
                                             entity_indices.end());

    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map
        = mesh.topology().shared_entities(D);

    // FIXME: This can be made more efficient by exploiting fact that
    //        set is sorted
    // Remove local cells from set_of_my_entities to reduce communication
    for (std::size_t j = 0; j < global_entity_indices.size(); ++j)
    {
      if (sharing_map.find(j) == sharing_map.end())
        set_of_my_entities.erase(global_entity_indices[j]);
    }
    // Copy entries from set_of_my_entities to my_entities
    my_entities = std::vector<std::size_t>(set_of_my_entities.begin(),
                                           set_of_my_entities.end());
  }
  else
    my_entities = entity_indices;

  // FIXME: handle case when my_entities.empty()
  // assert(!my_entities.empty());

  // Prepare data structures for send/receive
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_proc = MPI::size(mpi_comm);
  const std::size_t proc_num = MPI::rank(mpi_comm);
  const std::size_t max_recv = MPI::max(mpi_comm, my_entities.size());
  std::vector<std::size_t> off_process_entities(max_recv);

  // Send and receive data
  for (std::size_t k = 1; k < num_proc; ++k)
  {
    const std::size_t src = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;
    MPI::send_recv(mpi_comm, my_entities, dest, off_process_entities, src);

    const std::size_t recv_entity_count = off_process_entities.size();

    // Check if this process owns received entities, and if so
    // store local index
    std::vector<std::size_t> my_hosted_entities;
    {
      // Build a temporary map hosting global_entity_indices
      std::map<std::size_t, std::size_t> map_of_global_entity_indices;
      for (std::size_t j = 0; j < global_entity_indices.size(); j++)
        map_of_global_entity_indices[global_entity_indices[j]] = j;

      for (std::size_t j = 0; j < recv_entity_count; j++)
      {
        // Check if this process hosts 'received_entity'
        const std::size_t received_entity = off_process_entities[j];
        std::map<std::size_t, std::size_t>::const_iterator it
            = map_of_global_entity_indices.find(received_entity);
        if (it != map_of_global_entity_indices.end())
        {
          const std::size_t local_index = it->second;
          my_hosted_entities.push_back(received_entity);
          my_hosted_entities.push_back(local_index);
        }
      }
    }

    // Send/receive hosted cells
    const std::size_t max_recv_host_proc
        = MPI::max(mpi_comm, my_hosted_entities.size());
    std::vector<std::size_t> host_processes(max_recv_host_proc);
    MPI::send_recv(mpi_comm, my_hosted_entities, src, host_processes, dest);

    const std::size_t recv_hostproc_count = host_processes.size();
    for (std::size_t j = 0; j < recv_hostproc_count; j += 2)
    {
      const std::size_t global_index = host_processes[j];
      const std::size_t local_index = host_processes[j + 1];
      processes[global_index].insert({dest, local_index});
    }

    // FIXME: Do later for efficiency
    // Remove entries from entities (from my_entities) that cannot
    // reside on more processes (i.e., cells)
  }

  // Sanity check
  const std::set<std::size_t> test_set(my_entities.begin(), my_entities.end());
  const std::size_t number_expected = test_set.size();
  if (number_expected != processes.size())
    throw std::runtime_error("Sanity check failed");

  return processes;
}
//-----------------------------------------------------------------------------
std::unordered_map<std::int32_t,
                   std::vector<std::pair<std::int32_t, std::int32_t>>>
DistributedMeshTools::compute_shared_entities(const Mesh& mesh, std::size_t d)
{
  LOG(INFO) << "Compute shared mesh entities of dimension" << d;
  common::Timer timer("Computed shared mesh entities");

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const int comm_size = MPI::size(mpi_comm);

  // Return empty set if running in serial
  if (MPI::size(mpi_comm) == 1)
  {
    return std::unordered_map<
        std::int32_t, std::vector<std::pair<std::int32_t, std::int32_t>>>();
  }

  // Initialize entities of dimension d
  mesh.create_entities(d);

  // Number entities (globally)
  number_entities(mesh, d);

  // Get shared entities to processes map
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = mesh.topology().shared_entities(d);

  // Get local-to-global indices map
  const std::vector<std::int64_t>& global_indices_map
      = mesh.topology().global_indices(d);

  // Global-to-local map for each process
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, std::size_t>>
      global_to_local;

  // Pack global indices for sending to sharing processes
  std::vector<std::vector<std::size_t>> send_indices(comm_size);
  std::vector<std::vector<std::size_t>> local_sent_indices(comm_size);
  for (auto shared_entity = shared_entities.cbegin();
       shared_entity != shared_entities.cend(); ++shared_entity)
  {
    // Local index
    const std::int32_t local_index = shared_entity->first;

    // Global index
    assert(local_index < (std::int32_t)global_indices_map.size());
    std::size_t global_index = global_indices_map[local_index];

    // Destination process
    const std::set<std::int32_t>& sharing_processes = shared_entity->second;

    // Pack data for sending and build global-to-local map
    for (auto dest = sharing_processes.cbegin();
         dest != sharing_processes.cend(); ++dest)
    {
      send_indices[*dest].push_back(global_index);
      local_sent_indices[*dest].push_back(local_index);
      global_to_local[*dest].insert({global_index, local_index});
    }
  }

  std::vector<std::vector<std::size_t>> recv_entities;
  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Clear send data
  send_indices.clear();
  send_indices.resize(comm_size);

  // Determine local entities indices for received global entity indices
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    // Get process number of neighbour
    const std::size_t sending_proc = p;

    if (recv_entities[p].size() > 0)
    {
      // Get global-to-local map for neighbour process
      std::unordered_map<
          std::size_t,
          std::unordered_map<std::size_t, std::size_t>>::const_iterator it
          = global_to_local.find(sending_proc);
      assert(it != global_to_local.end());
      const std::unordered_map<std::size_t, std::size_t>&
          neighbour_global_to_local
          = it->second;

      // Build vector of local indices
      const std::vector<std::size_t>& global_indices_recv = recv_entities[p];
      for (std::size_t i = 0; i < global_indices_recv.size(); ++i)
      {
        // Global index
        const std::size_t global_index = global_indices_recv[i];

        // Find local index corresponding to global index
        std::unordered_map<std::size_t, std::size_t>::const_iterator
            n_global_to_local
            = neighbour_global_to_local.find(global_index);

        assert(n_global_to_local != neighbour_global_to_local.end());
        const std::size_t my_local_index = n_global_to_local->second;
        send_indices[sending_proc].push_back(my_local_index);
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Build map
  std::unordered_map<std::int32_t,
                     std::vector<std::pair<std::int32_t, std::int32_t>>>
      shared_local_indices_map;

  // Loop over data received from each process
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    if (recv_entities[p].size() > 0)
    {
      // Process that shares entities
      const std::size_t proc = p;

      // Local indices on sharing process
      const std::vector<std::size_t>& neighbour_local_indices
          = recv_entities[p];

      // Local indices on this process
      const std::vector<std::size_t>& my_local_indices = local_sent_indices[p];

      // Check that sizes match
      assert(my_local_indices.size() == neighbour_local_indices.size());

      for (std::size_t i = 0; i < neighbour_local_indices.size(); ++i)
      {
        shared_local_indices_map[my_local_indices[i]].push_back(
            {proc, neighbour_local_indices[i]});
      }
    }
  }

  return shared_local_indices_map;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::init_facet_cell_connections(Mesh& mesh)
{
  // Topological dimension
  const int D = mesh.topology().dim();

  // Initialize entities of dimension d
  mesh.create_entities(D - 1);

  // Initialise local facet-cell connections.
  mesh.create_connectivity(D - 1, D);

  // Global numbering
  number_entities(mesh, D - 1);

  // Calculate the number of global cells attached to each facet
  // essentially defining the exterior surface
  // FIXME: should this be done earlier, e.g. at partitioning stage
  // when dual graph is built?

  // Create vector to hold number of cells connected to each
  // facet. Initially copy over from local values.

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_global_neighbors(
      mesh.num_entities(D - 1));

  std::map<std::int32_t, std::set<std::int32_t>>& shared_facets
      = mesh.topology().shared_entities(D - 1);

  // Check if no ghost cells
  if (mesh.topology().ghost_offset(D) == mesh.topology().size(D))
  {
    // Copy local values
    assert(mesh.topology().connectivity(D - 1, D));
    auto connectivity = mesh.topology().connectivity(D - 1, D);
    for (auto& f : mesh::MeshRange<mesh::Facet>(mesh))
      num_global_neighbors[f.index()] = connectivity->size(f.index());

    // All shared facets must have two cells, if no ghost cells
    for (auto f_it = shared_facets.begin(); f_it != shared_facets.end(); ++f_it)
      num_global_neighbors[f_it->first] = 2;
  }
  else
  {
    // With ghost cells, shared facets may be on an external edge, so
    // need to check connectivity with the cell owner.

    const std::int32_t mpi_size = MPI::size(mesh.mpi_comm());
    std::vector<std::vector<std::size_t>> send_facet(mpi_size);
    std::vector<std::vector<std::size_t>> recv_facet(mpi_size);

    // Map shared facets
    std::map<std::size_t, std::size_t> global_to_local_facet;

    const std::vector<std::int32_t>& cell_owners = mesh.topology().cell_owner();
    const std::int32_t ghost_offset_c = mesh.topology().ghost_offset(D);
    const std::int32_t ghost_offset_f = mesh.topology().ghost_offset(D - 1);
    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_f
        = mesh.topology().shared_entities(D - 1);
    const auto& global_facets = mesh.topology().global_indices(D - 1);
    assert(mesh.topology().connectivity(D - 1, D));
    auto connectivity = mesh.topology().connectivity(D - 1, D);
    for (auto& f :
         mesh::MeshRange<MeshEntity>(mesh, D - 1, mesh::MeshRangeType::ALL))
    {
      // Insert shared facets into mapping
      if (sharing_map_f.find(f.index()) != sharing_map_f.end())
        global_to_local_facet.insert({global_facets[f.index()], f.index()});

      // Copy local values
      const int n_cells = connectivity->size(f.index());
      num_global_neighbors[f.index()] = n_cells;

      if ((f.index() >= ghost_offset_f) and n_cells == 1)
      {
        // Singly attached ghost facet - check with owner of attached
        // cell
        assert(f.entities(D)[0] >= ghost_offset_c);
        const int owner = cell_owners[f.entities(D)[0] - ghost_offset_c];
        send_facet[owner].push_back(global_facets[f.index()]);
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_facet, recv_facet);

    // Convert received global facet index into number of attached cells
    // and return to sender
    std::vector<std::vector<std::size_t>> send_response(mpi_size);
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (auto r = recv_facet[p].begin(); r != recv_facet[p].end(); ++r)
      {
        auto map_it = global_to_local_facet.find(*r);
        assert(map_it != global_to_local_facet.end());
        // const mesh::Facet local_facet(mesh, map_it->second);
        // const int n_cells = local_facet.num_entities(D);
        const int n_cells = connectivity->size(map_it->second);
        send_response[p].push_back(n_cells);
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_response, recv_facet);

    // Insert received result into same facet that it came from
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (std::size_t i = 0; i < recv_facet[p].size(); ++i)
      {
        auto f_it = global_to_local_facet.find(send_facet[p][i]);
        assert(f_it != global_to_local_facet.end());
        num_global_neighbors[f_it->second] = recv_facet[p][i];
      }
    }
  }

  assert(mesh.topology().connectivity(D - 1, D));
  mesh.topology().connectivity(D - 1, D)->set_global_size(num_global_neighbors);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<double>(mpi_comm, values,
                                                  global_indices);
}
//-----------------------------------------------------------------------------
Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
             Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<std::complex<double>>(
      mpi_comm, values, global_indices);
}
//-----------------------------------------------------------------------------
