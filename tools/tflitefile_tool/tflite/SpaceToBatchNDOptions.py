# automatically generated by the FlatBuffers compiler, do not modify

# namespace: onert_tflite

import flatbuffers


class SpaceToBatchNDOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSpaceToBatchNDOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SpaceToBatchNDOptions()
        x.Init(buf, n + offset)
        return x

    # SpaceToBatchNDOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SpaceToBatchNDOptionsStart(builder):
    builder.StartObject(0)


def SpaceToBatchNDOptionsEnd(builder):
    return builder.EndObject()
