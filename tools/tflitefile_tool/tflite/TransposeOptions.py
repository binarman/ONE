# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onert_tflite

import flatbuffers


class TransposeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTransposeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TransposeOptions()
        x.Init(buf, n + offset)
        return x

    # TransposeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def TransposeOptionsStart(builder):
    builder.StartObject(0)


def TransposeOptionsEnd(builder):
    return builder.EndObject()
