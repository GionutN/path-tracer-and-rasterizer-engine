#pragma once

#include <vector>
#include <array>
#include <string>
#include <map>
#include <set>

#include "mesh.h"
#include "shape.h"
#include "model.h"


// TODO:
// catching errors is needed, a ioniq class for standard exceptions

class scene
{
public:
	// represents the structure that needs to be sent to the gpu for rendering
	struct gpu_packet
	{
		UINT num_drawcalls[mesh::type::NUMTYPES] = {};
		UINT num_tri_meshes = 0;

		struct tri_mesh {
			vertex* vertices;
			UINT* indices;
			UINT num_indices;
			UINT num_vertices;
		} *tri_meshes;

		struct tri_mesh_drawcall
		{
			iqmat transform;
			UINT mesh_id;
		} *tri_mesh_dcs;
		struct sphere_drawcall
		{
			iqvec center;
			float radius;
		} *sphere_dcs;

		// TODO: add cylinders, parabolas and toruses
	};

	struct model_comparator
	{
		bool operator()(const model* a, const model* b) const {
			if (a->get_mesh_name() != b->get_mesh_name()) {
				return a->get_mesh_name() < b->get_mesh_name();
			}

			// tie braker to allow models with the same name
			return a < b;
		}
	};

public:
	scene();

	const std::set<model*, model_comparator>& get_models() const { return m_sorted_by_mesh_name; }

	void add(const mesh& m);
	void add_mesh(const std::string& name, const mesh& m);
	void rename_mesh(const std::string& old_name, const std::string& new_name);
	void delete_mesh(const std::string& name);
	const mesh& get_mesh(const std::string& name) const { return m_meshes.at(name); };
	mesh& get_mesh(const std::string& name) { return m_meshes[name]; };

	void add_model(const std::string& name, const model& m);
	void rename_model(const std::string& old_name, const std::string& new_name);
	void delete_model(const std::string& name);
	const model& get_model(const std::string& name) const { return m_models_x.at(name); };
	model& get_model(const std::string& name) { return m_models_x[name]; };
	void change_model_mesh(const std::string& model_name, const std::string& new_mesh_name);

	inline bool modified() const { return m_modified; }

	gpu_packet build_packet() const;
	void free_packet(gpu_packet* pkt) const;

private:
	std::array<UINT, mesh::type::NUMTYPES> m_model_types;

	mutable bool m_modified = true;	// if the scene changes, update the gpu packet

	// new scene data members
	std::map<std::string, mesh> m_meshes;	// will be used for instancing
	std::map<std::string, model> m_models_x;	// make this a map from a pair<model_name, mesh_name> to model
	std::set<model*, model_comparator> m_sorted_by_mesh_name;	// used for sorting the models based on the mesh's name

};
