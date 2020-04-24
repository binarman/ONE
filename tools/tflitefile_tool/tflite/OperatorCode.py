# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onert_tflite

import flatbuffers


class OperatorCode(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsOperatorCode(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OperatorCode()
        x.Init(buf, n + offset)
        return x

    # OperatorCode
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OperatorCode
    def BuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # OperatorCode
    def CustomCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OperatorCode
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1


def OperatorCodeStart(builder):
    builder.StartObject(3)


def OperatorCodeAddBuiltinCode(builder, builtinCode):
    builder.PrependInt8Slot(0, builtinCode, 0)


def OperatorCodeAddCustomCode(builder, customCode):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(customCode), 0)


def OperatorCodeAddVersion(builder, version):
    builder.PrependInt32Slot(2, version, 1)


def OperatorCodeEnd(builder):
    return builder.EndObject()
