// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2008-11-13

#include <cstring>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMesh::XMLMesh(Mesh& mesh) : XMLObject(), _mesh(mesh), state(OUTSIDE), f(0), a(0)
{
  // Do nothing
  
}
//-----------------------------------------------------------------------------
XMLMesh::~XMLMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLMesh::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      read_mesh(name, attrs);
      state = INSIDE_MESH;
    }
    
    break;

  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      read_vertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
    {
      read_cells(name, attrs);
      state = INSIDE_CELLS;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "data") == 0 )
    {
      state = INSIDE_DATA;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "coordinates") == 0 )
    {
      state = INSIDE_COORDINATES;
    }

    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertex") == 0 )
      read_vertex(name, attrs);

    break;
    
  case INSIDE_CELLS:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "interval") == 0 )
      read_interval(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "triangle") == 0 )
      read_triangle(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0 )
      read_tetrahedron(name, attrs);
    
    break;

  case INSIDE_DATA:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0 )
    {
      read_meshFunction(name, attrs);
      state = INSIDE_MESH_FUNCTION;
    }
    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
    {
      read_array(name, attrs);
      state = INSIDE_ARRAY;
    }

    break;

  case INSIDE_COORDINATES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      state = INSIDE_VECTOR;
    }

    break;

  case INSIDE_MESH_FUNCTION:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "entity") == 0 )
      read_meshEntity(name, attrs);

    break;

  case INSIDE_ARRAY:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "element") == 0 )
      read_arrayElement(name, attrs);

    break;

  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "entry") == 0 )
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      close_mesh();
      state = DONE;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      state = INSIDE_MESH;    
    }

    break;

  case INSIDE_CELLS:
	 
    if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_DATA:

    if ( xmlStrcasecmp(name, (xmlChar *) "data") == 0 )
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_COORDINATES:

    if ( xmlStrcasecmp(name, (xmlChar *) "coordinates") == 0 )
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_MESH_FUNCTION:

    if ( xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0 )
    {
      state = INSIDE_DATA;
    }

    break;

  case INSIDE_ARRAY:

    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
    {
      state = INSIDE_DATA;
    }

    break;

  case INSIDE_VECTOR:

    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      state = INSIDE_COORDINATES;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool XMLMesh::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parse_string(name, attrs, "celltype");
  uint gdim = parseUnsignedInt(name, attrs, "dim");
  
  // Create cell type to get topological dimension
  CellType* cell_type = CellType::create(type);
  uint tdim = cell_type->dim();
  delete cell_type;

  // Open mesh for editing
  editor.open(_mesh, CellType::string2type(type), tdim, gdim);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_vertices = parseUnsignedInt(name, attrs, "size");

  // Set number of vertices
  editor.init_vertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_cells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_cells = parseUnsignedInt(name, attrs, "size");

  // Set number of vertices
  editor.init_cells(num_cells);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parseUnsignedInt(name, attrs, "index");
  
  // Handle differently depending on geometric dimension
  switch ( _mesh.geometry().dim() )
  {
  case 1:
    {
      double x = parse_real(name, attrs, "x");
      editor.add_vertex(v, x);
    }
    break;
  case 2:
    {
      double x = parse_real(name, attrs, "x");
      double y = parse_real(name, attrs, "y");
      editor.add_vertex(v, x, y);
    }
    break;
  case 3:
    {
      double x = parse_real(name, attrs, "x");
      double y = parse_real(name, attrs, "y");
      double z = parse_real(name, attrs, "z");
      editor.add_vertex(v, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_interval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 1 )
    error("Mesh entity (interval) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  
  // Add cell
  editor.add_cell(c, v0, v1);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_triangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 2 )
    error("Mesh entity (triangle) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  
  // Add cell
  editor.add_cell(c, v0, v1, v2);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_tetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 3 )
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  uint v3 = parseUnsignedInt(name, attrs, "v3");
  
  // Add cell
  editor.add_cell(c, v0, v1, v2, v3);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_meshFunction(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  const std::string id = parse_string(name, attrs, "name");
  const std::string type = parse_string(name, attrs, "type");
  const uint dim = parseUnsignedInt(name, attrs, "dim");
  const uint size = parseUnsignedInt(name, attrs, "size");

  // Only uint supported at this point
  if (strcmp(type.c_str(), "uint") != 0)
    error("Only uint-valued mesh data is currently supported.");

  // Check size
  _mesh.init(dim);
  if (_mesh.size(dim) != size)
    error("Wrong number of values for MeshFunction, expecting %d.", _mesh.size(dim));

  // Register data
  f = _mesh.data().create_mesh_function(id);
  dolfin_assert(f);
  f->init(_mesh, dim);

  // Set all values to zero
  *f = 0;
}
//-----------------------------------------------------------------------------
void XMLMesh::read_array(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  const std::string id = parse_string(name, attrs, "name");
  const std::string type = parse_string(name, attrs, "type");
  const uint size = parseUnsignedInt(name, attrs, "size");

  // Only uint supported at this point
  if (strcmp(type.c_str(), "uint") != 0)
    error("Only uint-valued mesh data is currently supported.");

  // Register data
  a = _mesh.data().create_array(id, size);
  dolfin_assert(a);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_meshEntity(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parseUnsignedInt(name, attrs, "index");

  // Read and set value
  dolfin_assert(f);
  dolfin_assert(index < f->size());
  const uint value = parseUnsignedInt(name, attrs, "value");
  f->set(index, value);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_arrayElement(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parseUnsignedInt(name, attrs, "index");

  // Read and set value
  dolfin_assert(a);
  dolfin_assert(index < a->size());
  const uint value = parseUnsignedInt(name, attrs, "value");
  (*a)[index] = value;
}
//-----------------------------------------------------------------------------
void XMLMesh::close_mesh()
{
  editor.close(false);
}
//-----------------------------------------------------------------------------
