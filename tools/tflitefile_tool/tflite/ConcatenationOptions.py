# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onert_tflite

import flatbuffers


class ConcatenationOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConcatenationOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConcatenationOptions()
        x.Init(buf, n + offset)
        return x

    # ConcatenationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConcatenationOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConcatenationOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ConcatenationOptionsStart(builder):
    builder.StartObject(2)


def ConcatenationOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)


def ConcatenationOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def ConcatenationOptionsEnd(builder):
    return builder.EndObject()
