# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ros2_plugin_proto:srv/RosRpcWrapper.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RosRpcWrapper_Request(type):
    """Metaclass of message 'RosRpcWrapper_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ros2_plugin_proto')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ros2_plugin_proto.srv.RosRpcWrapper_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ros_rpc_wrapper__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ros_rpc_wrapper__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ros_rpc_wrapper__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ros_rpc_wrapper__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ros_rpc_wrapper__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RosRpcWrapper_Request(metaclass=Metaclass_RosRpcWrapper_Request):
    """Message class 'RosRpcWrapper_Request'."""

    __slots__ = [
        '_serialization_type',
        '_context',
        '_data',
    ]

    _fields_and_field_types = {
        'serialization_type': 'string',
        'context': 'sequence<string>',
        'data': 'sequence<octet>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.UnboundedString()),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('octet')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.serialization_type = kwargs.get('serialization_type', str())
        self.context = kwargs.get('context', [])
        self.data = kwargs.get('data', [])

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
        if self.serialization_type != other.serialization_type:
            return False
        if self.context != other.context:
            return False
        if self.data != other.data:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def serialization_type(self):
        """Message field 'serialization_type'."""
        return self._serialization_type

    @serialization_type.setter
    def serialization_type(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'serialization_type' field must be of type 'str'"
        self._serialization_type = value

    @builtins.property
    def context(self):
        """Message field 'context'."""
        return self._context

    @context.setter
    def context(self, value):
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, str) for v in value) and
                 True), \
                "The 'context' field must be a set or sequence and each value of type 'str'"
        self._context = value

    @builtins.property
    def data(self):
        """Message field 'data'."""
        return self._data

    @data.setter
    def data(self, value):
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, bytes) for v in value) and
                 True), \
                "The 'data' field must be a set or sequence and each value of type 'bytes'"
        self._data = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_RosRpcWrapper_Response(type):
    """Metaclass of message 'RosRpcWrapper_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ros2_plugin_proto')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ros2_plugin_proto.srv.RosRpcWrapper_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ros_rpc_wrapper__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ros_rpc_wrapper__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ros_rpc_wrapper__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ros_rpc_wrapper__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ros_rpc_wrapper__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RosRpcWrapper_Response(metaclass=Metaclass_RosRpcWrapper_Response):
    """Message class 'RosRpcWrapper_Response'."""

    __slots__ = [
        '_code',
        '_serialization_type',
        '_data',
    ]

    _fields_and_field_types = {
        'code': 'int64',
        'serialization_type': 'string',
        'data': 'sequence<octet>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('octet')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.code = kwargs.get('code', int())
        self.serialization_type = kwargs.get('serialization_type', str())
        self.data = kwargs.get('data', [])

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
        if self.code != other.code:
            return False
        if self.serialization_type != other.serialization_type:
            return False
        if self.data != other.data:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def code(self):
        """Message field 'code'."""
        return self._code

    @code.setter
    def code(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'code' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'code' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._code = value

    @builtins.property
    def serialization_type(self):
        """Message field 'serialization_type'."""
        return self._serialization_type

    @serialization_type.setter
    def serialization_type(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'serialization_type' field must be of type 'str'"
        self._serialization_type = value

    @builtins.property
    def data(self):
        """Message field 'data'."""
        return self._data

    @data.setter
    def data(self, value):
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, bytes) for v in value) and
                 True), \
                "The 'data' field must be a set or sequence and each value of type 'bytes'"
        self._data = value


class Metaclass_RosRpcWrapper(type):
    """Metaclass of service 'RosRpcWrapper'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ros2_plugin_proto')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ros2_plugin_proto.srv.RosRpcWrapper')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__ros_rpc_wrapper

            from ros2_plugin_proto.srv import _ros_rpc_wrapper
            if _ros_rpc_wrapper.Metaclass_RosRpcWrapper_Request._TYPE_SUPPORT is None:
                _ros_rpc_wrapper.Metaclass_RosRpcWrapper_Request.__import_type_support__()
            if _ros_rpc_wrapper.Metaclass_RosRpcWrapper_Response._TYPE_SUPPORT is None:
                _ros_rpc_wrapper.Metaclass_RosRpcWrapper_Response.__import_type_support__()


class RosRpcWrapper(metaclass=Metaclass_RosRpcWrapper):
    from ros2_plugin_proto.srv._ros_rpc_wrapper import RosRpcWrapper_Request as Request
    from ros2_plugin_proto.srv._ros_rpc_wrapper import RosRpcWrapper_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
