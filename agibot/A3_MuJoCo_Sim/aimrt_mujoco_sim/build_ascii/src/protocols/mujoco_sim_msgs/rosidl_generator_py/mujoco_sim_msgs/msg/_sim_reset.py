# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mujoco_sim_msgs:msg/SimReset.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SimReset(type):
    """Metaclass of message 'SimReset'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'MODE_ABSOLUTE': 0,
        'MODE_KEYFRAME': 1,
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('mujoco_sim_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'mujoco_sim_msgs.msg.SimReset')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__sim_reset
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__sim_reset
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__sim_reset
            cls._TYPE_SUPPORT = module.type_support_msg__msg__sim_reset
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__sim_reset

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

            from geometry_msgs.msg import Twist
            if Twist.__class__._TYPE_SUPPORT is None:
                Twist.__class__.__import_type_support__()

            from sensor_msgs.msg import JointState
            if JointState.__class__._TYPE_SUPPORT is None:
                JointState.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'MODE_ABSOLUTE': cls.__constants['MODE_ABSOLUTE'],
            'MODE_KEYFRAME': cls.__constants['MODE_KEYFRAME'],
        }

    @property
    def MODE_ABSOLUTE(self):
        """Message constant 'MODE_ABSOLUTE'."""
        return Metaclass_SimReset.__constants['MODE_ABSOLUTE']

    @property
    def MODE_KEYFRAME(self):
        """Message constant 'MODE_KEYFRAME'."""
        return Metaclass_SimReset.__constants['MODE_KEYFRAME']


class SimReset(metaclass=Metaclass_SimReset):
    """
    Message class 'SimReset'.

    Constants:
      MODE_ABSOLUTE
      MODE_KEYFRAME
    """

    __slots__ = [
        '_header',
        '_mode',
        '_keyframe_id',
        '_set_base',
        '_pelvis_pose',
        '_set_base_twist',
        '_pelvis_twist',
        '_set_joints',
        '_joint_state',
        '_zero_all_velocities',
        '_clear_ctrl',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'mode': 'uint8',
        'keyframe_id': 'int32',
        'set_base': 'boolean',
        'pelvis_pose': 'geometry_msgs/Pose',
        'set_base_twist': 'boolean',
        'pelvis_twist': 'geometry_msgs/Twist',
        'set_joints': 'boolean',
        'joint_state': 'sensor_msgs/JointState',
        'zero_all_velocities': 'boolean',
        'clear_ctrl': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Twist'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'JointState'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.mode = kwargs.get('mode', int())
        self.keyframe_id = kwargs.get('keyframe_id', int())
        self.set_base = kwargs.get('set_base', bool())
        from geometry_msgs.msg import Pose
        self.pelvis_pose = kwargs.get('pelvis_pose', Pose())
        self.set_base_twist = kwargs.get('set_base_twist', bool())
        from geometry_msgs.msg import Twist
        self.pelvis_twist = kwargs.get('pelvis_twist', Twist())
        self.set_joints = kwargs.get('set_joints', bool())
        from sensor_msgs.msg import JointState
        self.joint_state = kwargs.get('joint_state', JointState())
        self.zero_all_velocities = kwargs.get('zero_all_velocities', bool())
        self.clear_ctrl = kwargs.get('clear_ctrl', bool())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.mode != other.mode:
            return False
        if self.keyframe_id != other.keyframe_id:
            return False
        if self.set_base != other.set_base:
            return False
        if self.pelvis_pose != other.pelvis_pose:
            return False
        if self.set_base_twist != other.set_base_twist:
            return False
        if self.pelvis_twist != other.pelvis_twist:
            return False
        if self.set_joints != other.set_joints:
            return False
        if self.joint_state != other.joint_state:
            return False
        if self.zero_all_velocities != other.zero_all_velocities:
            return False
        if self.clear_ctrl != other.clear_ctrl:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'mode' field must be an unsigned integer in [0, 255]"
        self._mode = value

    @builtins.property
    def keyframe_id(self):
        """Message field 'keyframe_id'."""
        return self._keyframe_id

    @keyframe_id.setter
    def keyframe_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'keyframe_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'keyframe_id' field must be an integer in [-2147483648, 2147483647]"
        self._keyframe_id = value

    @builtins.property
    def set_base(self):
        """Message field 'set_base'."""
        return self._set_base

    @set_base.setter
    def set_base(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'set_base' field must be of type 'bool'"
        self._set_base = value

    @builtins.property
    def pelvis_pose(self):
        """Message field 'pelvis_pose'."""
        return self._pelvis_pose

    @pelvis_pose.setter
    def pelvis_pose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'pelvis_pose' field must be a sub message of type 'Pose'"
        self._pelvis_pose = value

    @builtins.property
    def set_base_twist(self):
        """Message field 'set_base_twist'."""
        return self._set_base_twist

    @set_base_twist.setter
    def set_base_twist(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'set_base_twist' field must be of type 'bool'"
        self._set_base_twist = value

    @builtins.property
    def pelvis_twist(self):
        """Message field 'pelvis_twist'."""
        return self._pelvis_twist

    @pelvis_twist.setter
    def pelvis_twist(self, value):
        if __debug__:
            from geometry_msgs.msg import Twist
            assert \
                isinstance(value, Twist), \
                "The 'pelvis_twist' field must be a sub message of type 'Twist'"
        self._pelvis_twist = value

    @builtins.property
    def set_joints(self):
        """Message field 'set_joints'."""
        return self._set_joints

    @set_joints.setter
    def set_joints(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'set_joints' field must be of type 'bool'"
        self._set_joints = value

    @builtins.property
    def joint_state(self):
        """Message field 'joint_state'."""
        return self._joint_state

    @joint_state.setter
    def joint_state(self, value):
        if __debug__:
            from sensor_msgs.msg import JointState
            assert \
                isinstance(value, JointState), \
                "The 'joint_state' field must be a sub message of type 'JointState'"
        self._joint_state = value

    @builtins.property
    def zero_all_velocities(self):
        """Message field 'zero_all_velocities'."""
        return self._zero_all_velocities

    @zero_all_velocities.setter
    def zero_all_velocities(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'zero_all_velocities' field must be of type 'bool'"
        self._zero_all_velocities = value

    @builtins.property
    def clear_ctrl(self):
        """Message field 'clear_ctrl'."""
        return self._clear_ctrl

    @clear_ctrl.setter
    def clear_ctrl(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'clear_ctrl' field must be of type 'bool'"
        self._clear_ctrl = value
