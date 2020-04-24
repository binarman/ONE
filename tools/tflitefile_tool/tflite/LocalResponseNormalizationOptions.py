# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onert_tflite

import flatbuffers


class LocalResponseNormalizationOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLocalResponseNormalizationOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LocalResponseNormalizationOptions()
        x.Init(buf, n + offset)
        return x

    # LocalResponseNormalizationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LocalResponseNormalizationOptions
    def Radius(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # LocalResponseNormalizationOptions
    def Bias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LocalResponseNormalizationOptions
    def Alpha(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LocalResponseNormalizationOptions
    def Beta(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0


def LocalResponseNormalizationOptionsStart(builder):
    builder.StartObject(4)


def LocalResponseNormalizationOptionsAddRadius(builder, radius):
    builder.PrependInt32Slot(0, radius, 0)


def LocalResponseNormalizationOptionsAddBias(builder, bias):
    builder.PrependFloat32Slot(1, bias, 0.0)


def LocalResponseNormalizationOptionsAddAlpha(builder, alpha):
    builder.PrependFloat32Slot(2, alpha, 0.0)


def LocalResponseNormalizationOptionsAddBeta(builder, beta):
    builder.PrependFloat32Slot(3, beta, 0.0)


def LocalResponseNormalizationOptionsEnd(builder):
    return builder.EndObject()
