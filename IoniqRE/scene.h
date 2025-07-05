#pragma once

#include <vector>
#include <array>
#include <string>
#include <unordered_map>

#include "mesh.h"
#include "shape.h"
#include "model.h"

template<typename T>
class named_object_map
{
public:
	named_object_map() = default;

	void add(const std::string& name, const T& obj)
	{
		auto result = m_map.emplace(name, obj);  // does nothing if the key already exists
		if (!result.second) {
			// log that there already is an object with the same name
		}
	}
	void modify(const std::string& name, const T& obj) {
		m_map[name] = obj;
	}
	void rename(const std::string& old_name, const std::string& new_name) {
		typename std::unordered_map<std::string, T>::iterator iter = m_map.find(old_name);
		if (iter == m_map.end()) {
			// the element does not exist
			return;
		}

		T obj = std::move(iter->second);
		m_map.erase(iter);
		m_map[new_name] = obj;
	}
	void remove(const std::string& name) {
		typename std::unordered_map<std::string, T>::iterator iter = m_map.find(name);
		if (iter == m_map.end()) {
			return;
		}

 		m_map.erase(iter);
	}
	std::pair<std::unordered_map<std::string, T>::iterator, bool> get(const std::string& name)
	{
		typename std::unordered_map<std::string, T>::iterator iter = m_map.find(name);
		return { iter, iter == m_map.end() ? false : true };
	}
	std::pair<std::unordered_map<std::string, T>::const_iterator, bool> get(const std::string& name) const
	{
		typename std::unordered_map<std::string, T>::const_iterator iter = m_map.find(name);
		return { iter, iter == m_map.cend() ? false : true };
	}

	std::unordered_map<std::string, T>::iterator begin() noexcept {
		return m_map.begin();
	}
	std::unordered_map<std::string, T>::const_iterator begin() const noexcept {
		return m_map.begin();
	}
	std::unordered_map<std::string, T>::const_iterator cbegin() const noexcept {
		return m_map.cbegin();
	}
	std::unordered_map<std::string, T>::iterator end() noexcept {
		return m_map.end();
	}
	std::unordered_map<std::string, T>::const_iterator end() const noexcept {
		return m_map.end();
	}
	std::unordered_map<std::string, T>::const_iterator cend() const noexcept {
		return m_map.cend();
	}

	size_t size() const {
		return m_map.size();
	}

private:
	std::unordered_map<std::string, T> m_map;

};

class scene
{
public:
	// represents the structure that needs to be sent to the gpu for rendering
	struct gpu_packet
	{
		vertex* vertices;
		UINT* indices;
		UINT* model_types;
	};

	struct gpu_packet_x
	{
		UINT num_drawcalls[mesh::type::NUMTYPES];
		UINT num_tri_meshes;

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

public:
	scene();

	const std::vector<mesh>& meshes() const { return m_models; }

	void add(const mesh& m);
	inline bool modified() const { return m_modified; }
	gpu_packet build_packet() const;
	gpu_packet_x build_packet_x() const;
	void free_packet_x(gpu_packet_x* pkt) const;

private:
	std::vector<mesh> m_models;
	std::array<UINT, mesh::type::NUMTYPES> m_model_types;

	size_t m_vertices = 0;	// total number of vertices
	size_t m_indices = 0;	// total number of indices
	mutable bool m_modified = true;	// if the scene changes, update the gpu packet

	// new scene data members
	named_object_map<mesh> m_meshes;	// will be used for instancing
	named_object_map<model> m_models_x;

};
