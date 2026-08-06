/**
 * @file fk.cpp
 * @brief Implementation of RobotFK – forward kinematics from MuJoCo XML.
 *
 * Parses a MuJoCo MJCF XML file to extract the kinematic tree (joint axes,
 * body translations, rest rotations) and recursively computes world-frame
 * positions and orientations via `DoFK()`.
 *
 * The XML parsing uses the header-only `xml.h` library (MIT, Vlad Krupinskii).
 * `XML_H_IMPLEMENTATION` is defined here (and only here) to pull in the
 * implementation.
 *
 * Joint angle indexing uses `isaaclab_to_mujoco` from policy_parameters.hpp
 * to map the IsaacLab-ordered input array to the MuJoCo kinematic tree order.
 */

#include "fk.hpp"
#include "../include/math_utils.hpp"
#include "../include/policy_parameters.hpp"

#include <sstream>
#include <array>
#include <string_view>

#define XML_H_IMPLEMENTATION 1
#include "xml.h"

/// RAII wrapper for the xml.h parse tree – ensures xml_node_free on scope exit.
struct XMLOwner
{
    XMLOwner(const std::string &xmlfile)
    {
        root_ = xml_parse_file(xmlfile.c_str());
        if(!root_)
        {
            throw std::runtime_error("Couldn't parse xml file " + xmlfile);
        }
    }

    ~XMLOwner()
    {
        if(root_)
        {
            xml_node_free(root_);
        }
    }

    XMLNode *root_ = nullptr;
};

namespace {

constexpr std::array<std::string_view, 31> kA3SdkJointNames = {
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint"};

int A3SdkIndexForJointName(const char* name) noexcept {
    if (!name) return -1;
    const std::string_view value{name};
    for (std::size_t i = 0; i < kA3SdkJointNames.size(); ++i) {
        if (value == kA3SdkJointNames[i]) return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

RobotFK::RobotFK(const std::string &xmlfile)
{
    XMLOwner xml(xmlfile);
    XMLNode *world_body = xml_node_find_by_path(xml.root_, "mujoco/worldbody", true);
    if(!world_body)
    {
        throw std::runtime_error("couldn't find worldbody node in " + xmlfile);
    }

    XMLNode *root_body = xml_node_find_tag(world_body, "body", true);
    AddNode(root_body);
}

int RobotFK::FindBodyIndex(const std::string &name) const
{
    for (size_t i = 0; i < node_names_.size(); ++i)
    {
        if (node_names_[i] == name) return static_cast<int>(i);
    }
    return -1;
}
    
void RobotFK::AddNode(XMLNode *node)
{
    auto nodename = xml_node_attr(node, "name");
    nodename = nodename ? nodename : "<unnamed>";
    node_names_.push_back(nodename);

    size_t node_idx = node_children_.size();
    node_children_.emplace_back();

    XMLNode *joint = xml_node_find_tag(node, "joint", true);
    if(!joint)
    {
        throw std::runtime_error("Couldn't find joint for node " + std::string(nodename));
    }
    std::array<double, 3> axis = {0.0, 0.0, 0.0};
    const char *axis_attr = xml_node_attr(joint, "axis");
    if(axis_attr)
    {
        std::istringstream iss(axis_attr);
        iss >> axis[0];
        iss >> axis[1];
        iss >> axis[2];
        if(iss.fail())
        {
            throw std::runtime_error("couldn't parse axis for node '" + std::string(nodename) + "'");
        }
    }
    axes_.push_back(axis);
    a3_joint_indices_.push_back(
        A3SdkIndexForJointName(xml_node_attr(joint, "name")));

    std::array<double, 3> pos = {0.0, 0.0, 0.0};
    const char *pos_attr = xml_node_attr(node, "pos");
    if(pos_attr)
    {
        std::istringstream iss(pos_attr);
        iss >> pos[0];
        iss >> pos[1];
        iss >> pos[2];
        if(iss.fail())
        {
            throw std::runtime_error("couldn't parse pos for node '" + std::string(nodename) + "'");
        }
    }
    translations_.push_back(pos);

    std::array<double, 4> quat = {1.0, 0.0, 0.0, 0.0};
    const char *quat_attr = xml_node_attr(node, "quat");
    if(quat_attr)
    {
        std::istringstream iss(quat_attr);
        iss >> quat[0];
        iss >> quat[1];
        iss >> quat[2];
        iss >> quat[3];
        if(iss.fail())
        {
            throw std::runtime_error("couldn't parse pos for node '" + std::string(nodename) + "'");
        }
    }
    rest_rotations_.push_back(quat);

    for (size_t i = 0; i < node->children->len; i++)
    {
        XMLNode *child = (XMLNode *)node->children->data[i];
        if (strcmp(child->tag, "body") == 0)
        {
            size_t child_idx = node_children_.size();
            AddNode(child);
            node_children_[node_idx].push_back(child_idx);
        }
    }
}

void RobotFK::FKChildren(
    std::array<double, 3> *positions_world,
    std::array<double, 4> *rotations_world,
    int parent_idx,
    const double *joint_angles
) const
{
    for(auto child_idx : node_children_[parent_idx])
    {
        positions_world[child_idx] = quat_rotate(rotations_world[parent_idx], translations_[child_idx]);
        positions_world[child_idx][0] += positions_world[parent_idx][0];
        positions_world[child_idx][1] += positions_world[parent_idx][1];
        positions_world[child_idx][2] += positions_world[parent_idx][2];

        std::array<double, 4> child_rot = quat_from_angle_axis(
            joint_angles[isaaclab_to_mujoco[child_idx-1]],
            quat_rotate(rest_rotations_[child_idx], axes_[child_idx])
        );

        rotations_world[child_idx] = quat_mul(
            rotations_world[parent_idx],
            quat_mul(
                child_rot,
                rest_rotations_[child_idx]
            )
        );

        FKChildren(
            positions_world,
            rotations_world,
            child_idx,
            joint_angles
        );
    }
}

void RobotFK::FKChildrenA3(
    std::array<double, 3> *positions_world,
    std::array<double, 4> *rotations_world,
    int parent_idx,
    const double *joint_angles_31
) const
{
    for (auto child_idx : node_children_[parent_idx])
    {
        positions_world[child_idx] = quat_rotate(rotations_world[parent_idx], translations_[child_idx]);
        positions_world[child_idx][0] += positions_world[parent_idx][0];
        positions_world[child_idx][1] += positions_world[parent_idx][1];
        positions_world[child_idx][2] += positions_world[parent_idx][2];

        // MuJoCo body traversal is tree order, not actuator/SDK order. Select
        // the measured angle through the joint-name mapping built at parse
        // time.
        const int sdk_idx = a3_joint_indices_[child_idx];
        const double q = (sdk_idx >= 0 && sdk_idx < 31)
            ? joint_angles_31[sdk_idx] : 0.0;
        std::array<double, 4> child_rot = quat_from_angle_axis(
            q, quat_rotate(rest_rotations_[child_idx], axes_[child_idx]));
        rotations_world[child_idx] = quat_mul(
            rotations_world[parent_idx],
            quat_mul(child_rot, rest_rotations_[child_idx]));

        FKChildrenA3(positions_world, rotations_world, child_idx, joint_angles_31);
    }
}

void RobotFK::DoFK(
    std::array<double, 3> *positions_world,
    std::array<double, 4> *rotations_world,
    const std::array<double, 3> &root_translation,
    const std::array<double, 4> &root_rotation,
    const double *joint_angles
) const
{
    positions_world[0] = root_translation;
    rotations_world[0] = root_rotation;

    FKChildren(positions_world, rotations_world, 0, joint_angles);
}

void RobotFK::DoFKA3(
    std::array<double, 3> *positions_world,
    std::array<double, 4> *rotations_world,
    const std::array<double, 3> &root_translation,
    const std::array<double, 4> &root_rotation,
    const double *joint_angles_31
) const
{
    positions_world[0] = root_translation;
    rotations_world[0] = root_rotation;
    FKChildrenA3(positions_world, rotations_world, 0, joint_angles_31);
}
